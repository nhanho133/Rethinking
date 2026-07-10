import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

# from datasets.sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr, cosine_lr_pergroup
import argparse
import subprocess
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
import random
from PIL import Image
# from torchvision.transforms import ToPILImage

from torch.utils.data import DataLoader, random_split
import pandas as pd

from datasets_config.datasets_config import dataset_mapping
from caption_chunking import chunking
# from clip_adapter import Adapter
from sampling import star_bar_long_text_split
from pathlib import Path

import os
import torch
import torch.distributed as dist

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return distributed, rank, world_size, local_rank


class CLIP_Clean_Train():
    def __init__(self, args):
        # --- Model & optimizer setup ---
        self.base_model = args.base_model

        if args.adapter:
            print("adapter turn on:", args.adapter)

        # self.model, _ = longclip.load_from_clip(
        #     self.base_model,
        #     device='cpu',
        #     download_root=args.download_root,
        #     adapter = args.adapter
        # )

        self.distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_main = (self.rank == 0)
        # === LLM2CLIP text-teacher variant: keep ViT-B/16 visual, swap text encoder for a
        # frozen Llama-3-8B-CC whose embeddings are looked up from a precomputed cache. ===
        self.is_llm2clip = args.base_model in ("llm2clip_frozen", "llm2clip_text", "llm2clip_released")
        if self.is_llm2clip:
            from model.model_llm2clip import (LLM2CLIPTextTeacher, LLM2CLIPReleasedTeacher,
                                              TextEmbeddingCache)
            if args.base_model == "llm2clip_released":
                # faithful 'fully LLM2CLIP': released ViT-L-336 (frozen) + their pretrained
                # TextProj adapter (trained). Loss ablation platform (vanilla vs ours).
                print(f"[Init] LLM2CLIP RELEASED (ViT-L-336, freeze_visual={args.freeze_visual}); "
                      f"ckpt: {args.llm2clip_model_path}; text cache: {args.text_cache_path}")
                self.model = LLM2CLIPReleasedTeacher(
                    model_path=args.llm2clip_model_path, log_scale=args.log_scale, device="cpu",
                    freeze_visual=args.freeze_visual)
            else:
                freeze_visual = (args.base_model == "llm2clip_frozen")
                print(f"[Init] LLM2CLIP text-teacher (freeze_visual={freeze_visual}); "
                      f"text cache: {args.text_cache_path}")
                loss_w = (1.0, args.w_pos, args.w_longer, args.w_hinge)
                print(f"[Init] loss_w (w_long,w_pos,w_longer,w_hinge) = {loss_w}")
                self.model = LLM2CLIPTextTeacher(
                    clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                    freeze_visual=freeze_visual, log_scale=args.log_scale, device="cpu",
                    adapter_type=args.adapter_type, loss_w=loss_w,
                )
            self.text_cache = TextEmbeddingCache(args.text_cache_path, device=self.device)
            self.model.train()
            self.model = self.model.to(self.device)
        # === NHÁNH A: có init_ckpt -> load giống eval_all.py ===
        elif getattr(args, "init_ckpt", None):
            ckpt_path = str(Path(args.init_ckpt).expanduser())
            print(f"[Init] Loading LongCLIP from checkpoint: {ckpt_path}")
            # giống hệt eval_all.load_longclip nhưng để trên CPU
            self.model, _ = longclip.load(ckpt_path, device="cpu")
            # logit_scale lúc này lấy luôn từ ckpt, KHÔNG override
            self.model.train()
            self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
            self.model = self.model.to(self.device)
        else:
            # === NHÁNH B: không có init_ckpt -> load từ CLIP gốc như trước ===
            print(f"[Init] Loading LongCLIP from CLIP backbone: {self.base_model}")
            self.model, _ = longclip.load_from_clip(
                self.base_model,
                device='cpu',
                download_root=args.download_root,
                adapter=args.adapter
            )
            # chỉ set logit_scale khi train từ CLIP gốc
            self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
            self.model.train()
            self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
            self.model = self.model.to(self.device)

        # --without_hinge_loss / --weight_hinge_loss (test case 3, PDF 9/7/2026): applies to
        # both branches (model_longclip.py and model_llm2clip.py both read self.model.loss_w).
        # Must run BEFORE the DDP wrap below: setting attributes on a DDP-wrapped object sets
        # them on the wrapper, not on `.module`, so the underlying forward() would never see it.
        self.use_hinge_loss = not args.without_hinge_loss
        hinge_weight = args.weight_hinge_loss if args.weight_hinge_loss is not None else args.w_hinge
        if hasattr(self.model, "loss_w"):
            # Apply ALL loss weights from args uniformly, so --w_pos/--w_longer/--w_hinge work on
            # BOTH the llm2clip_text and llm2clip_released models (the released model never had its
            # loss_w set from args otherwise -> those flags would silently be ignored on it).
            self.model.loss_w = (1.0, args.w_pos, args.w_longer, hinge_weight)
        print(f"[Init] use_hinge_loss={self.use_hinge_loss}, loss_w={getattr(self.model,'loss_w',None)}")

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True)

        self.batch_size    = args.batch_size
        self.num_epoch     = args.epochs
        self.lr            = args.lr
        self.weight_decay  = args.weight_decay
        self.warmup_length = args.warmup_length

        if self.is_llm2clip:
            # Two lr groups: the pretrained ViT-B/16 at the (small) base lr, the randomly-init
            # 4096->512 text adapter (+ logit_scale) at a much higher lr, since a new head needs
            # to move fast while the backbone must not be blown up. Stage 1 confirmed the adapter
            # wants ~1e-3.
            self.adapter_lr = args.adapter_lr
            _is_visual = lambda n: ("visual" in n) or ("vision_model" in n)
            visual_params = [p for n, p in self.model.named_parameters()
                             if p.requires_grad and _is_visual(n)]
            other_params = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and not _is_visual(n)]
            groups = []
            if visual_params:
                groups.append({"params": visual_params, "lr": self.lr})
            groups.append({"params": other_params, "lr": self.adapter_lr})
            self.optimizer = optim.AdamW(groups, weight_decay=self.weight_decay)
            print(f"[optim] llm2clip param groups: visual(lr={self.lr}, n={len(visual_params)}) "
                  f"+ adapter(lr={self.adapter_lr}, n={len(other_params)})")
        else:
            self.optimizer = optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad), # self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        self.scaler = GradScaler()

        # --- Logging & checkpoints ---
        if args.exp_name == "auto":
            self.logdir = (
                f"longclip/lr={self.lr}_wd={self.weight_decay}"
                f"_wl={self.warmup_length}_logs={args.log_scale}_64xb"
            )
        else:
            self.logdir = args.exp_name
        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir) if self.is_main else None

        self.results = []
        if args.train_data not in dataset_mapping.keys():
            raise ValueError(
                f"Invalid train_data: {args.train_data}. "
                f"Allowed values are: {', '.join(dataset_mapping.keys())}"
            )
        self.train_data = args.train_data
        self.use_learnable_mps = args.learnable_mps
        self.max_num_short_texts = args.max_num_short_texts
        # self.use_sparsemax = args.use_sparsemax
        self.naive_sampling = args.naive_sampling
        self.loss_mode = getattr(args, "loss_mode", "full")
        self.grad_accum_steps = getattr(args, "grad_accum_steps", 1)
        print(f"\n Train model:{self.base_model} -- use learnable mps:{self.use_learnable_mps}")
        print(f"Number of bars:{self.max_num_short_texts }")
        print(f"Batch size:{args.batch_size} \n")
        # print(f"use sparsemax:{self.use_sparsemax}")
        # print(f"naive sampling:{self.naive_sampling}")
    def split_into_detail_captions(self, text_long):
        sentences = [
            p.strip()
            for p in text_long.split('.')
            if p.strip()
            and len(p.strip()) >= 18
            and not (len(p.strip()) == 1 and p.strip().isalpha())
        ]
        return sentences

    def chunk_random(self, sentences, min_len=1, max_len=5, seed=None):
        if seed is not None:
            random.seed(seed)
        chunks = []
        i = 0
        while i < len(sentences):
            take = random.randint(min_len, max_len)
            chunk_text = ". ".join(sentences[i:i+take])
            chunks.append(chunk_text.strip())
            i += take
        return chunks

    def extract_image_patches(self, images, patch_size=16):
        B, C, H, W = images.shape
        patches = images.unfold(2, patch_size, patch_size)\
                         .unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        return patches.permute(0, 2, 1, 3, 4)
    
    def shuffle_words(self, phrase, rng):
        words = phrase.split()
        rng.shuffle(words)
        return " ".join(words)

    def make_base_longer(self, chunks, rng):
        n = len(chunks)
        if n <= 1:
            return [chunks[0]] if n == 1 else []
        out = []
        for i in range(n):
            partner = rng.randrange(n - 1)
            if partner >= i:
                partner += 1
            
            a, b = sorted([i, partner])  # đảm bảo thứ tự index tăng dần
            out.append((chunks[a].strip() + " " + chunks[b].strip()).strip())
        return out

    def make_neg_details_for_batch(self, chunks_all, chunk_longer_all, rng, shuffle_rate=1):
        """
        chunks_all[j] = list các base captions (c_1..c_K) của sample j
        output: neg_captions_all[j][t] = chunks_all[j][t] + d^-  (d^- lấy từ sample khác)
        """
        B = len(chunks_all)
        neg_captions_all = []

        for cur_id in range(B):
            cur_chunks = chunk_longer_all[cur_id]
            neg_list = []

            for t in range(len(cur_chunks)):  # mỗi chunk đều có 1 negative
                # pick other sample
                other_id = rng.randrange(B - 1)
                if other_id >= cur_id:
                    other_id += 1
                other_chunks = chunks_all[other_id]

                # pick 1 "unit" từ sample khác (tránh lấy other_chunks[0] nếu đó là base quá chung)
                pick_idx = rng.randrange(len(other_chunks))
                neg_unit = other_chunks[pick_idx].strip()

                # paper: shuffle rate để tránh neg quá "fluent" gây false negative
                # (họ báo 90% là tốt nhất) :contentReference[oaicite:4]{index=4}
                if rng.random() < shuffle_rate:
                    neg_unit = self.shuffle_words(neg_unit, rng)

                neg_caption = f"{cur_chunks[t].strip()} {neg_unit}".strip()
                neg_list.append(neg_caption)

            neg_captions_all.append(neg_list)

        return neg_captions_all



    def train_epoch(self, dataloader, epoch, start_iter=0, finegrained_loss=False, learnable_mps=True, naive_sampling=False):
        running_loss = 0.0
        num_batches = len(dataloader)
        
        for i, (images, texts, short_text, _, _) in enumerate(tqdm(dataloader)):
            step = num_batches * epoch + i
            if step < start_iter:
                continue

            # fixed sampling seed for mp_loss
            num_seed=42
            random.seed(num_seed)


            # num_det = 5  # số câu muốn lấy
            # sample_size = num_det
            tokenized_caps = []

            chunks_all = []

            
            """ STAR BAR STRATEGY """
            for j, long_text in enumerate(texts):
                split_cap = self.split_into_detail_captions(long_text)
                
                if len(split_cap) < self.max_num_short_texts:
                    # new_star = random.randint(1, len(split_cap))                    
                    # chunks = star_bar_long_text_split(split_cap, new_star, num_seed)
                    
                    chunks = star_bar_long_text_split(split_cap, len(split_cap), num_seed)
                    
                    while len(chunks) < self.max_num_short_texts:
                        new_star = random.randint(1, len(split_cap))
                        new_chunks = star_bar_long_text_split(split_cap, new_star, num_seed)
                        chunks.append(random.choice(new_chunks))
                else:
                    chunks = star_bar_long_text_split(split_cap, self.max_num_short_texts, num_seed)
                """END STRATEGY"""
                if not chunks:
                    chunks = [short_text[j]]
                # import pdb
                # pdb.set_trace()
                chunks_all.append(chunks)
                # tokenized_caps is unused by the model forward; skip building it for the
                # llm2clip branch (its text comes from the embedding cache, not token ids).
                if not self.is_llm2clip:
                    caps = [
                        longclip.tokenize(dc, truncate=True).to(self.device)
                        for dc in (chunks or [short_text[j]])
                    ]
                    tokenized_caps.append(caps)
            rng_batch = random.Random(num_seed + 999)
            # neg_details_all = self.make_neg_details_for_batch(chunks_all, rng_batch)
            chunks_longer_all = []
            for j in range(len(texts)):
                rng = random.Random(num_seed + 1000 + j)
                chunks_longer_all.append(self.make_base_longer(chunks_all[j], rng))
            
            # neg_details_all = self.make_neg_details_for_batch(chunks_all, chunks_longer_all, rng_batch)
            # neg_list = [longclip.tokenize(neg, truncate=True).cuda() for neg in neg_details_all]
            neg_list = None

            if self.is_llm2clip:
                # text side = frozen Llama-3-8B-CC embeddings looked up from the precomputed
                # cache; each chunks_all[j]/chunks_longer_all[j] is a list of K strings -> [K, 4096]
                long_tokens = self.text_cache.lookup(list(texts))   # [B, 4096]
                if self.loss_mode == "vanilla":
                    # ClipLoss baseline needs only the full caption; skip sub-caption lookups.
                    pos_list = pos_longer_list = None
                else:
                    pos_list = [self.text_cache.lookup(pos) for pos in chunks_all]
                    pos_longer_list = [self.text_cache.lookup(pos_longer) for pos_longer in chunks_longer_all]
            else:
                pos_list = [longclip.tokenize(pos, truncate=True).to(self.device) for pos in chunks_all]
                pos_longer_list = [longclip.tokenize(pos_longer, truncate=True).cuda() for pos_longer in chunks_longer_all]
                long_tokens = longclip.tokenize(texts, truncate=True).cuda()
            # short_token = longclip.tokenize(short_text, truncate=True).cuda()
 
            images = images.cuda(non_blocking=True)

            accum = max(self.grad_accum_steps, 1)
            is_accum_boundary = ((i + 1) % accum == 0) or (i == num_batches - 1)
            self.scheduler(step)  # smooth per-batch; only optimizer.step() is gated by accum
            if i % accum == 0:
                self.optimizer.zero_grad()

            with autocast():
                use_hinge=False
                # loss = self.model(images, 
                #                   long_tokens, 
                #                   tokenized_caps,
                #                   learnable_mps= self.use_learnable_mps
                #                   )
                # loss, L_long, L_pos, L_pos_longer, L_neg = self.model(images, 
                #                   long_tokens, 
                #                   tokenized_caps,
                #                   learnable_mps= self.use_learnable_mps,
                #                   text_pos = pos_list,
                #                   text_neg = neg_list,
                #                   text_pos_longer= pos_longer_list
                #                   )
                # loss_mode=="vanilla" -> LLM2CLIP's ClipLoss baseline: only L_long (image <->
                # full caption), no multi-positive sub-captions, no specificity hinge.
                if self.loss_mode == "vanilla":
                    _text_pos, _text_pos_longer, use_hinge = None, None, False
                else:
                    _text_pos, _text_pos_longer, use_hinge = pos_list, pos_longer_list, self.use_hinge_loss
                loss, L_long, L_pos, L_longer, L_hinge, logit_scale, tau = self.model(images,
                                  long_tokens,
                                  tokenized_caps,
                                  learnable_mps= self.use_learnable_mps,
                                  text_pos = _text_pos,
                                  text_neg = neg_list,
                                  text_pos_longer= _text_pos_longer,
                                  use_hinge=use_hinge
                                  )
            if use_hinge:
                if i % 50 == 0:
                    print(f"Epoch {epoch} ▶ "
                        f"total_loss: {loss:.4f} ▶ "
                        f"L_long: {L_long:.4f} ▶ "
                        # f"L_pos_longer: {L_pos_longer:.4f} ▶ "
                        f"L_pos: {L_pos:.4f} ▶ "
                        f"L_longer: {L_longer:.4f} ▶ "
                        f"L_hinge: {L_hinge:.4f} ▶ "
                        f"logit scale: {logit_scale:.4f} --- tau: {tau:.4f}"
                        )
            else:
                if i % 50 == 0:
                    print(f"Epoch {epoch} ▶ "
                        f"Loss: {loss:.4f}")
                        # f"Finegrained: {finegrained_loss:.4f}, MP: {mp_loss:.4f}")

            if self.is_main and self.writer is not None:
                self.writer.add_scalar("train/total_loss", loss.item(), step)
                self.writer.add_scalar("train/L_long", L_long.item(), step)
                if use_hinge:
                    self.writer.add_scalar("train/L_pos", L_pos.item(), step)
                    self.writer.add_scalar("train/L_longer", L_longer.item(), step)
                    self.writer.add_scalar("train/L_hinge", L_hinge.item(), step)
                self.writer.add_scalar("train/logit_scale", logit_scale.item(), step)

            # Scale down so accumulated gradients over `accum` micro-batches sum to the same
            # magnitude as a single full-batch backward (effective batch = batch_size * accum).
            self.scaler.scale(loss / accum).backward()
            if is_accum_boundary:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # running_loss           += loss.item()
            loss_for_log = loss.detach()
            if self.distributed:
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
                loss_for_log = loss_for_log / self.world_size

            running_loss           += loss_for_log.item()

        avg_loss               = running_loss / num_batches
        if self.is_main:
            print(f"Epoch {epoch} ▶ Avg Loss: {avg_loss:.4f}, ")
        return avg_loss
        # avg_loss               = running_loss / num_batches
        # avg_finegrained_loss   = running_loss_finegrained / num_batches
        # avg_mp_loss            = running_loss_mp_loss / num_batches

        # print(f"Epoch {epoch} ▶ Avg Loss: {avg_loss:.4f}, "
        #       f"Finegrained: {avg_finegrained_loss:.4f}, MP: {avg_mp_loss:.4f}")
        # return avg_loss, avg_finegrained_loss, avg_mp_loss

    @torch.no_grad()
    def test_epoch(self, dataloader):
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        correct = total = 0
        for batch in tqdm(dataloader):
            images = batch[0].cuda(non_blocking=True)
            texts  = batch[1]

            # encode
            im_feats = model.encode_image(images)
            im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)

            txt = longclip.tokenize(texts, truncate=True).cuda()
            txt_feats = model.encode_text(txt)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            sims = txt_feats @ im_feats.T
            preds = sims.argmax(dim=1)
            correct += (preds == torch.arange(sims.size(0), device=preds.device)).sum().item()
            total   += sims.size(0)

        acc = correct / total if total > 0 else 0.0
        print(f"▶ Test Accuracy: {acc:.4%}")
        return acc

    @torch.no_grad()
    def test_epoch_ver2(self, dataloader):
        # chuyển model về eval và chọn đúng object nếu dùng DDP
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        im_feats_list   = []
        txt_feats_list  = []
        txt1_feats_list = []
        txt2_feats_list = []
        txt3_feats_list = []
        txt4_feats_list = []

        for images, texts, _ in tqdm(dataloader):
            # --- image features ---
            images = images.cuda(non_blocking=True)
            im_feats = model.encode_image(images)
            im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)
            im_feats_list.append(im_feats)

            # --- full-text features ---
            tokens_full = longclip.tokenize(texts, truncate=True).cuda()
            feats_full  = model.encode_text(tokens_full)
            feats_full  = feats_full / feats_full.norm(dim=-1, keepdim=True)
            txt_feats_list.append(feats_full)

            # --- split thành câu ---
            caps = [self.split_into_detail_captions(t) for t in texts]
            first_texts  = [c[0] if len(c) > 0 else "" for c in caps]
            second_texts = [c[1] if len(c) > 1 else "" for c in caps]
            third_texts  = [c[2] if len(c) > 2 else "" for c in caps]
            fourth_texts = [c[3] if len(c) > 3 else "" for c in caps]

            # câu đầu tiên
            tokens1 = longclip.tokenize(first_texts, truncate=True).cuda()
            feats1  = model.encode_text(tokens1)
            feats1  = feats1 / feats1.norm(dim=-1, keepdim=True)
            txt1_feats_list.append(feats1)

            # câu thứ hai
            tokens2 = longclip.tokenize(second_texts, truncate=True).cuda()
            feats2  = model.encode_text(tokens2)
            feats2  = feats2 / feats2.norm(dim=-1, keepdim=True)
            txt2_feats_list.append(feats2)

            # câu thứ ba
            tokens3 = longclip.tokenize(third_texts, truncate=True).cuda()
            feats3  = model.encode_text(tokens3)
            feats3  = feats3 / feats3.norm(dim=-1, keepdim=True)
            txt3_feats_list.append(feats3)

            # câu thứ tư
            tokens4 = longclip.tokenize(fourth_texts, truncate=True).cuda()
            feats4  = model.encode_text(tokens4)
            feats4  = feats4 / feats4.norm(dim=-1, keepdim=True)
            txt4_feats_list.append(feats4)

        # --- concat tất cả ---
        im_feats_all   = torch.cat(im_feats_list,   dim=0)  # [N, D]
        txt_full_all   = torch.cat(txt_feats_list,  dim=0)  # [N, D]
        txt1_all       = torch.cat(txt1_feats_list, dim=0)  # [N, D]
        txt2_all       = torch.cat(txt2_feats_list, dim=0)  # [N, D]
        txt3_all       = torch.cat(txt3_feats_list, dim=0)  # [N, D]
        txt4_all       = torch.cat(txt4_feats_list, dim=0)  # [N, D]

        # --- ma trận cosine similarity ---
        sims_full = txt_full_all @ im_feats_all.T  # [N, N]
        sims1     = txt1_all     @ im_feats_all.T
        sims2     = txt2_all     @ im_feats_all.T
        sims3     = txt3_all     @ im_feats_all.T
        sims4     = txt4_all     @ im_feats_all.T

        # --- target index ---
        N = sims_full.size(0)
        target = torch.arange(N, device=sims_full.device)

        # Full Text → Image
        acc_t2i = (sims_full.argmax(dim=1) == target).sum().item() / N
        # Image → Full Text
        acc_i2t = (sims_full.argmax(dim=0) == target).sum().item() / N

        # First Sentence → Image
        acc_first  = (sims1.argmax(dim=1) == target).sum().item() / N
        # Second Sentence → Image
        acc_second = (sims2.argmax(dim=1) == target).sum().item() / N
        # Third Sentence → Image
        acc_third  = (sims3.argmax(dim=1) == target).sum().item() / N
        # Fourth Sentence → Image
        acc_fourth = (sims4.argmax(dim=1) == target).sum().item() / N

        print(f"▶ Full Text→Image:     {acc_t2i:.4%}")
        print(f"▶ Image→Full Text:     {acc_i2t:.4%}")
        print(f"▶ 1st Sentence→Image:  {acc_first:.4%}")
        print(f"▶ 2nd Sentence→Image:  {acc_second:.4%}")
        print(f"▶ 3rd Sentence→Image:  {acc_third:.4%}")
        print(f"▶ 4th Sentence→Image:  {acc_fourth:.4%}")

        return acc_t2i, acc_i2t, acc_first, acc_second, acc_third, acc_fourth
    
    
    @torch.no_grad()
    def test_epoch_ver5(self, dataloader, epoch):
        # 1) Eval mode & unwrap DDP
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # 2) Collect features
        im_feats_list    = []
        im_paths_list    = []
        long_feats_list  = []
        txt_feats_lists  = {j: [] for j in range(4)}
        detail_texts_list = {j: [] for j in range(4)}
        
        for images, long_texts, _, image_paths, _ in tqdm(dataloader, desc="Extracting features"):
            # import pdb
            # pdb.set_trace()
            im_paths_list.extend(image_paths)

            # Image features
            feats_i = model.encode_image(images.cuda(non_blocking=True))
            feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
            im_feats_list.append(feats_i)

            # Full-text features
            if self.is_llm2clip:
                feats_long = model.encode_text(self.text_cache.lookup(list(long_texts)))
            else:
                feats_long = model.encode_text(longclip.tokenize(long_texts, truncate=True).cuda())
            feats_long  = feats_long / feats_long.norm(dim=-1, keepdim=True)
            long_feats_list.append(feats_long)

            # Split long_texts into detail captions
            caps = [self.split_into_detail_captions(t) for t in long_texts]
            # import pdb
            # pdb.set_trace()
            for j in range(4):
                detail_texts_list[j].extend([c[j] if j < len(c) else "" for c in caps])

            # Encode each detail caption group
            for j in range(4):
                detail_batch = detail_texts_list[j][-len(long_texts):]
                if self.is_llm2clip:
                    fts = model.encode_text(self.text_cache.lookup(detail_batch))
                else:
                    fts = model.encode_text(longclip.tokenize(detail_batch, truncate=True).cuda())
                txt_feats_lists[j].append(fts / fts.norm(dim=-1, keepdim=True))

        # 3) Concatenate all features
        im_feats_all  = torch.cat(im_feats_list,   dim=0)
        long_all      = torch.cat(long_feats_list, dim=0)
        txt_all_lists = [torch.cat(txt_feats_lists[j], dim=0) for j in range(4)]

        # 4) Compute similarity matrices (Text→Image)
        sims_t2i = {
            "long": long_all @ im_feats_all.T,
            **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(4)}
        }
        sims_i2t_long = im_feats_all @ long_all.T

        # 5) Compute Recall@K
        N = im_feats_all.size(0)
        target = torch.arange(N, device=im_feats_all.device)
        ks = [1, 5, 10, 25, 50]
        acc = {}

        # Text→Image (full + details)
        for name, sims in sims_t2i.items():
            for k in ks:
                topk_inds = sims.topk(k, dim=1).indices  # [N, k]
                hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
                acc[f"{name}_t2i_R{k}"] = hits

        # Image→Text (full)
        for k in ks:
            topk_inds = sims_i2t_long.topk(k, dim=1).indices
            hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"long_i2t_R{k}"] = hits

        # 6) Print summary
        print("\n—— Retrieval Metrics ——")
        for k in ks:
            print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
        for j in range(4):
            for k in ks:
                print(f"▶ Detail {j+1} → Image @ {k:2}: {acc[f'detail_{j+1}_t2i_R{k}']:.4%}")
        for k in ks:
            print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
        print("—" * 40)

        # 7) Return metrics dict
        return acc

    

    def train(self, resume=False, warmup_length=200):
        # deterministic 80/20 split
        torch.manual_seed(42)

        print("tạo train test")

        # DocciDataset uses model_name only to fetch the CLIP image transform. The released
        # LLM2CLIP uses ViT-L/14@336px; the other llm2clip variants keep ViT-B/16.
        if self.base_model == "llm2clip_released":
            preprocess_model = "ViT-L/14@336px"
        elif self.is_llm2clip:
            preprocess_model = "ViT-B/16"
        else:
            preprocess_model = self.base_model
        train_ds, test_ds = dataset_mapping[self.train_data](model_name=preprocess_model)
        # train_loader = DataLoader(
        #     train_ds,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=32,
        #     pin_memory=True
        # )
        # test_loader = DataLoader(
        #     test_ds,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=32,
        #     pin_memory=True
        # )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.distributed else None
        test_sampler = DistributedSampler(test_ds, shuffle=False) if self.distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=32,
            pin_memory=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
        )
        print("tạo train test xong")

        if self.is_llm2clip:
            # per-group cosine so the ViT (1e-6) and adapter (1e-3) each decay from their own base
            self.scheduler = cosine_lr_pergroup(
                self.optimizer,
                warmup_length=warmup_length,
                steps=self.num_epoch * len(train_loader),
            )
        else:
            self.scheduler = cosine_lr(
                self.optimizer,
                base_lr=self.lr,
                warmup_length=warmup_length,
                steps=self.num_epoch * len(train_loader)
            )
        # Specify which epoch to save ckpt for each dataset
        if self.train_data=="docci":
            save_ckpt_list = [2, 3, 4, 5, 6]
        elif self.train_data=="xopenv1":
            save_ckpt_list = [4, 5, 6, 8, 9]
        elif self.train_data=="sharedgpt4v":
            save_ckpt_list = [0, 1, 2, 3, 4, 5, 6, 8]
        elif self.train_data=="artpedia":
            save_ckpt_list = [1, 2, 3, 5, 6]
        elif self.train_data=="dreamlip_cc3m":
            save_ckpt_list = [2, 3, 4, 5, 6]
        elif self.train_data=="sharegpt4v_coco":
            save_ckpt_list = [2, 3, 4, 5, 6]
        else:
            raise ValueError(
                f" You should specify which epoch to save checkpoint to: {args.train_data}. "
                f"Allowed data are: {', '.join(dataset_mapping.keys())}"
            )
        # For long runs, the hardcoded lists above only cover early epochs -> extend
        # with periodic saves (every 3 epochs) plus the final epoch so late-run
        # checkpoints (where the best result may land) aren't silently dropped.
        if self.num_epoch > max(save_ckpt_list) + 1:
            save_ckpt_list = sorted(
                set(save_ckpt_list)
                | set(range(max(save_ckpt_list), self.num_epoch, 3))
                | {self.num_epoch - 1}
            )

        for epoch in range(self.num_epoch):
            if self.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # train
            self.model.train()
            avg_loss = self.train_epoch(train_loader, epoch, naive_sampling=self.naive_sampling)

            # save checkpoint
            now = datetime.now().strftime("%m-%d--%H_%M_%S_")
            if self.is_llm2clip:
                ckpt_name = f"{args.base_model}-B16-{now}-{epoch}.pt"
            elif args.base_model=="ViT-B/16":
                # ckpt_name = f"B16-longclip-{now}-{epoch}_art.pt"
                ckpt_name = f"B16-longclip-{now}-{epoch}.pt"
            else:
                ckpt_name= f"L14-longclip-{now}-{epoch}.pt"
            # ckpt_name = f"Propose-b16-longclip-{now}-share4v-no.pt"\
            if epoch in save_ckpt_list and self.is_main:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), os.path.join(self.ckptdir, ckpt_name))
            # if epoch == self.num_epoch - 1 or epoch == 5:
            #     ckpt_name = f"Propose-b16-longclip-{now}-dci-both-mlp-lr-{epoch}.pt"
            #     # ckpt_name = f"Propose-b16-longclip-{now}-share4v-no.pt"
            #     torch.save(self.model.state_dict(), os.path.join(self.ckptdir, ckpt_name))

            # optional external eval
            eval_script = os.path.join("..", "eval", "classification", "imagenet", "imagenet.py")
            if os.path.isfile(eval_script):
                try:
                    out = subprocess.check_output(
                        ["python", eval_script],
                        cwd=os.path.dirname(eval_script),
                        stderr=subprocess.STDOUT
                    )
                    print("=== External Eval ===")
                    print(out.decode().splitlines()[-1])
                    print("=====================")
                except subprocess.CalledProcessError as e:
                    print("External eval error:", e.output.decode())
            else:
                print(">> Skip external eval: script not found.")

            acc = self.test_epoch_ver5(test_loader, epoch)
            
            # log
            # if self.use_finegrained_loss:
            #     self.results.append({
            #         "epoch":        epoch,
            #         "avg_loss":     avg_loss,
            #         "avg_finegrained_loss":    avg_fg,
            #         "avg_mp_loss":   avg_mp,
            #         "test_acc_t2i": acc[f'long_t2i_R{1}'],
            #         "test_acc_i2t": acc[f'long_i2t_R{1}']
            #     })
            # else:
            self.results.append({
                "epoch":        epoch,
                "avg_loss":     avg_loss,
                "test_acc_t2i": acc[f'long_t2i_R{1}'],
                "test_acc_i2t": acc[f'long_i2t_R{1}']
            })
            pd.DataFrame(self.results)\
            .to_excel(os.path.join(self.logdir, "epoch_results.xlsx"),
                        index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-B/16",
                        help="CLIP Base Model, or 'llm2clip_frozen'/'llm2clip_text' for the "
                             "frozen-Llama-3-8B-CC text-teacher variant")
    # parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument("--text_cache_path", type=str,
                        default="/cm/shared/chautvh_second/Nhan_folder/work/docci_cache.pt",
                        help="Precomputed LLM2Vec text-embedding cache (for llm2clip_* base models)")
    parser.add_argument("--adapter_lr", type=float, default=1e-3,
                        help="lr for the llm2clip text adapter (random-init head); the ViT uses --lr")
    parser.add_argument("--w_pos", type=float, default=1.0, help="loss weight for L_pos (llm2clip_* only)")
    parser.add_argument("--w_longer", type=float, default=1.0, help="loss weight for L_longer (llm2clip_* only)")
    parser.add_argument("--w_hinge", type=float, default=1.0, help="loss weight for L_hinge (llm2clip_* only)")
    parser.add_argument("--without_hinge_loss", action="store_true",
                        help="Disable L_hinge entirely (use_hinge=False in model forward). "
                             "Ablation flag, applies to both longclip and llm2clip_* branches.")
    parser.add_argument("--weight_hinge_loss", type=float, default=None,
                        help="Override weight for L_hinge (alias of --w_hinge, applies to both "
                             "longclip and llm2clip_* branches). Ignored if --without_hinge_loss is set.")
    parser.add_argument("--adapter_type", type=str, default="linear", choices=["linear", "mlp"],
                        help="llm2clip text adapter: 'linear' (light, best on DOCCI) or 'mlp' (faithful LLM2CLIP TextProj)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Micro-batches to accumulate before an optimizer step (stabilizes "
                             "gradients; does NOT increase in-batch contrastive negatives -- "
                             "raise --batch_size for that).")
    parser.add_argument("--loss_mode", type=str, default="full", choices=["full", "vanilla"],
                        help="full = L_long + DreamLIP multi-positive + SPECS hinge (ours); "
                             "vanilla = L_long only (LLM2CLIP's ClipLoss baseline)")
    parser.add_argument("--llm2clip_model_path", type=str,
                        default="/cm/shared/chautvh_second/Nhan_folder/ckpts/ViT-L-336",
                        help="Path to the released LLM2CLIP model (for --base_model llm2clip_released)")
    parser.add_argument("--freeze_visual", action="store_true",
                        help="Freeze the ViT-L visual tower (llm2clip_released). Default False on "
                             "H100 = train full visual (LLM2CLIP's real stage-2). Set on small GPU.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per gpu.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--resume", default=False, action='store_true', help="resume training from checkpoint.")
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    parser.add_argument("--use_sparsemax", action="store_true", help="Enable sparsemax in loss")
    

    parser.add_argument("--train_data", type=str,default="docci", help="Which data to train")
    parser.add_argument("--learnable_mps", action="store_true", help="Enable finegrained loss. Default will be False")
    parser.add_argument("--max_num_short_texts", type=int, default=5, help="Max number of splitting short texts")
    parser.add_argument("--adapter", action="store_true", help="Enable adapter. Default will be False")

    parser.add_argument("--naive_sampling", action="store_true", help="Split 1 sentence per subcaptions")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Path to LongCLIP checkpoint để init weight trước khi train"
    )


    args = parser.parse_args()
    print("DDP Done")

    trainer = CLIP_Clean_Train(args=args)
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
