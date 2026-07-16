import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
import pandas as pd  # For saving results to Excel
import random
import os
from PIL import Image
# from torchvision.transforms import ToPILImage


import os
import subprocess
import random
from datetime import datetime

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader, random_split
from datasets_config import dataset_mapping

# import longclip  # hoặc tên module tokenize của bạn

class CLIP_Clean_Train():
    def __init__(self, args):
        # --- Model & optimizer setup ---
        self.base_model = args.base_model
        self.model, _ = longclip.load_from_clip(
            self.base_model,
            device='cpu',
            download_root=args.download_root
        )
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
        self.model = self.model.cuda()

        self.batch_size    = args.batch_size
        self.num_epoch     = args.epochs
        self.lr            = args.lr
        self.weight_decay  = args.weight_decay
        self.warmup_length = args.warmup_length

        self.optimizer = optim.AdamW(
            self.model.parameters(),
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
        self.writer = SummaryWriter(self.logdir)

        self.results = []
        if args.train_data not in dataset_mapping.keys():
            raise ValueError(
                f"Invalid train_data: {args.train_data}. "
                f"Allowed values are: {', '.join(dataset_mapping.keys())}"
            )
        self.train_data = args.train_data
        self.use_finegrained_loss= args.use_finegrained
        self.use_learnable_mps = args.learnable_mps
        print(f"Train model:{self.base_model} -- use finegrained:{self.use_finegrained_loss} -- use learnable mps:{self.use_learnable_mps}")
    

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

    def train_epoch(self, dataloader, epoch, start_iter=0, finegrained_loss=False, learnable_mps=True):
        running_loss = running_loss_finegrained = running_loss_mp_loss = 0.0
        num_batches = len(dataloader)

        for i, (images, texts, short_text, _, _) in enumerate(tqdm(dataloader)):

            step = num_batches * epoch + i
            if step < start_iter:
                continue

            # prepare detail captions
            # tokenized_detail_caps = [
            #     [longclip.tokenize(dc, truncate=True).cuda()
            #      for dc in self.split_into_detail_captions(text)[:20]]
            #     for text in texts
            # ]
            # import pdb
            # pdb.set_trace()
            # tokenized_detail_caps = [
            #     [dc #longclip.tokenize(dc, truncate=True).cuda()
            #      for dc in (self.split_into_detail_captions(text)[:20] or [short_text[j]])
            #     ] for j, text in enumerate(texts)
            # ]

            # fixed sampling seed for mp_loss
            import math
            num_seed=42
            random.seed(num_seed)


            num_det = 5  # số câu muốn lấy
            sample_size = num_det
            tokenized_caps = []
            for j, text in enumerate(texts):
                split_cap = self.split_into_detail_captions(text)
                chunks =self.chunk_random(split_cap, min_len=1, max_len=3)
                if len(chunks) < num_det:
                    i=1
                    while len(chunks) < num_det:
                        new_chunks = self.chunk_random(split_cap, min_len=1, max_len=3, seed=num_seed+i)
                        if len(new_chunks)==0:
                            import pdb
                            pdb.set_trace()
                        chunks.append(random.choice(new_chunks))
                        i=i+1 
                else:
                    chunks = random.sample(chunks, num_det)

                caps = [
                    longclip.tokenize(dc, truncate=True).cuda()
                    for dc in (chunks or [short_text[j]])
                ]
                tokenized_caps.append(caps)

                # nếu không đủ sample_size thì nhân bản để đạt ít nhất sample_size
                # import pdb
                # pdb.set_trace()
                # if len(caps) < num_det:
                #     import pdb
                #     pdb.set_trace()
                #     multiplier = math.ceil(sample_size / len(caps))
                #     sample = (caps * multiplier)[:sample_size]
                #     sample = caps
                #     tokenized_detail_caps.append(sample)
                # else:
                #     sample = random.sample(caps, num_det)
                #     tokenized_detail_caps.append(sample)


            long_texts_token = longclip.tokenize(texts, truncate=True).cuda()
            # short_token = longclip.tokenize(short_text, truncate=True).cuda()
 
            images = images.cuda(non_blocking=True)

            self.scheduler(step)
            self.optimizer.zero_grad()

            with autocast():
                if not finegrained_loss:
                    # loss = self.model(
                    #     images,
                    #     short_token,
                    #     tokenized_detail_caps,
                    #     # tokenized_detail_caps_sample
                    # )

                    loss = self.model(
                        images,
                        long_texts_token,
                        tokenized_caps,
                    )
                else:
                    finegrained_loss, mp_loss = self.model(
                        images,
                        # texts_token,
                        # short_token,
                        tokenized_caps,
                        # tokenized_detail_caps_sample
                    )
                    loss = finegrained_loss + mp_loss

            if i % 50 == 0:
                print(f"Epoch {epoch} ▶ "
                    f"Loss: {loss:.4f}")
                    # f"Finegrained: {finegrained_loss:.4f}, MP: {mp_loss:.4f}")

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss           += loss.item()
            # running_loss_finegrained += finegrained_loss.item()
            # running_loss_mp_loss   += mp_loss.item()
        avg_loss               = running_loss / num_batches
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
    def test_epoch_ver3(self, dataloader, epoch):
        # 1) chuyển model về eval và chọn đúng object nếu dùng DDP
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # 2) gom features và giữ paths + caption strings
        im_feats_list, im_paths_list = [], []
        long_feats_list = []
        txt_feats_lists = {j: [] for j in range(4)}
        long_texts_list = []
        detail_texts_list = {j: [] for j in range(4)}

        for images, long_texts, _, image_paths in tqdm(dataloader, desc="Extracting features"):
            im_paths_list.extend(image_paths)

            # image features
            feats_i = model.encode_image(images.cuda(non_blocking=True))
            feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
            im_feats_list.append(feats_i)

            # full-text features
            tokens_long = longclip.tokenize(long_texts, truncate=True).cuda()
            feats_long  = model.encode_text(tokens_long)
            feats_long  = feats_long / feats_long.norm(dim=-1, keepdim=True)
            long_feats_list.append(feats_long)

            # split long_texts thành detail captions
            caps = [self.split_into_detail_captions(t) for t in long_texts]
            for j in range(4):
                # lấy câu thứ j nếu có, else ""
                detail_texts_list[j].extend([c[j] if j < len(c) else "" for c in caps])

            # tính feature cho từng câu con
            for j in range(4):
                toks = longclip.tokenize(detail_texts_list[j][-len(long_texts):], truncate=True).cuda()
                fts  = model.encode_text(toks)
                txt_feats_lists[j].append(fts / fts.norm(dim=-1, keepdim=True))

            long_texts_list.extend(long_texts)

        # 3) concat features
        im_feats_all  = torch.cat(im_feats_list,   dim=0)  # [N, D]
        long_all      = torch.cat(long_feats_list, dim=0)  # [N, D]
        txt_all_lists = [torch.cat(txt_feats_lists[j], dim=0) for j in range(4)]

        # 4) similarity matrices text->image
        sims_t2i = {
            "long": long_all @ im_feats_all.T,
            **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(4)}
        }

        # 5) tạo thư mục kết quả
        root  = os.path.join(self.logdir, f"retrieval_results_{epoch}")
        types = list(sims_t2i.keys())
        for t in types:
            os.makedirs(os.path.join(root, t), exist_ok=True)

        N, topk = im_feats_all.size(0), 10
        texts_dict = {"long": long_texts_list, **{f"detail_{j+1}": detail_texts_list[j] for j in range(4)}}

        # 6) gộp ảnh Top-10 + ground-truth thành 1 ảnh chung
        for t in types:
            inds = sims_t2i[t].topk(topk, dim=1).indices  # [N, topk]
            folder = os.path.join(root, t)
            for i in tqdm(range(N), desc=f"Compositing {t}"):
                # Ảnh đúng (ground-truth) luôn đầu tiên
                true_img = Image.open(im_paths_list[i]).convert("RGB")
                retrieved_paths = [im_paths_list[idx] for idx in inds[i].tolist()]

                # Mở và chuẩn hoá kích thước tất cả ảnh về cùng height
                imgs = [true_img]
                for p in retrieved_paths:
                    imgs.append(Image.open(p).convert("RGB"))
                h_min = min(im.height for im in imgs)
                thumbs = [im.resize((int(im.width * h_min / im.height), h_min), Image.LANCZOS) for im in imgs]

                # Tạo canvas và ghép
                total_w = sum(im.width for im in thumbs)
                comp = Image.new("RGB", (total_w, h_min))
                x_offset = 0
                for im in thumbs:
                    comp.paste(im, (x_offset, 0))
                    x_offset += im.width

                # Lưu ảnh composite và file caption
                comp.save(os.path.join(folder, f"{i}.png"))
                txt_path = os.path.join(folder, f"{i}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f_txt:
                    f_txt.write(texts_dict[t][i])

        # 7) tính Acc cả 2 chiều
        target = torch.arange(N, device=im_feats_all.device)
        acc = {}

        # text -> image
        acc_t2i = (sims_t2i["long"].argmax(dim=1) == target).sum().item() / N
        acc["long_t2i"] = acc_t2i
        for j in range(4):
            key = f"detail_{j+1}"
            acc[f"{key}_t2i"] = (sims_t2i[key].argmax(dim=1) == target).sum().item() / N

        # image -> text
        sims_i2t_long = im_feats_all @ long_all.T
        acc_i2t = (sims_i2t_long.argmax(dim=1) == target).sum().item() / N
        acc["long_i2t"] = acc_i2t

        # 8) in ra các metrics bạn cần
        print(f"▶ Full Text→Image:     {acc_t2i:.4%}")
        print(f"▶ Image→Full Text:     {acc_i2t:.4%}")
        print(f"▶ 1st Sentence→Image:  {acc['detail_1_t2i']:.4%}")
        print(f"▶ 2nd Sentence→Image:  {acc['detail_2_t2i']:.4%}")
        print(f"▶ 3rd Sentence→Image:  {acc['detail_3_t2i']:.4%}")
        print(f"▶ 4th Sentence→Image:  {acc['detail_4_t2i']:.4%}")

        # 9) trả về list metrics (tuỳ bạn sắp xếp thứ tự)
        return [
            acc_t2i,
            acc_i2t,
            acc["detail_1_t2i"],
            acc["detail_2_t2i"],
            acc["detail_3_t2i"],
            acc["detail_4_t2i"],
        ]
    
    @torch.no_grad()
    def test_epoch_ver4(self, dataloader, epoch):
        # 1) chuyển model về eval và chọn đúng object nếu dùng DDP
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # 2) gom features và giữ paths + caption strings
        im_feats_list    = []
        im_paths_list    = []
        long_feats_list  = []
        txt_feats_lists  = {j: [] for j in range(4)}
        detail_texts_list = {j: [] for j in range(4)}

        for images, long_texts, _, image_paths, _ in tqdm(dataloader, desc="Extracting features"):
            im_paths_list.extend(image_paths)

            # image features
            feats_i = model.encode_image(images.cuda(non_blocking=True))
            feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
            im_feats_list.append(feats_i)

            # full-text features
            tokens_long = longclip.tokenize(long_texts, truncate=True).cuda()
            feats_long  = model.encode_text(tokens_long)
            feats_long  = feats_long / feats_long.norm(dim=-1, keepdim=True)
            long_feats_list.append(feats_long)

            # split long_texts thành detail captions
            caps = [self.split_into_detail_captions(t) for t in long_texts]
            for j in range(4):
                detail_texts_list[j].extend([c[j] if j < len(c) else "" for c in caps])

            # tính feature cho từng câu con
            for j in range(4):
                toks = longclip.tokenize(detail_texts_list[j][-len(long_texts):], truncate=True).cuda()
                fts  = model.encode_text(toks)
                txt_feats_lists[j].append(fts / fts.norm(dim=-1, keepdim=True))

        # 3) concat features
        im_feats_all  = torch.cat(im_feats_list,   dim=0)
        long_all      = torch.cat(long_feats_list, dim=0)
        txt_all_lists = [torch.cat(txt_feats_lists[j], dim=0) for j in range(4)]

        # 4) similarity matrices text->image
        sims_t2i = {
            "long": long_all @ im_feats_all.T,
            **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(4)}
        }

        # # 5) cache thumbnails to speed up IO + resize once
        # target_h = 200
        # thumb_cache = {}
        # for p in set(im_paths_list):
        #     im = Image.open(p).convert("RGB")
        #     w = int(im.width * target_h / im.height)
        #     thumb_cache[p] = im.resize((w, target_h), Image.BILINEAR)

        # # 6) tạo thư mục kết quả chỉ cho detail captions (bỏ "long")
        # root = os.path.join("/home/ubuntu/shared/hieu.tq/test/images/docci_mul", f"retrieval_results_{epoch}")
        # save_types = [t for t in sims_t2i.keys() if t != "long"]
        # for t in save_types:
        #     os.makedirs(os.path.join(root, t), exist_ok=True)

        N, topk = im_feats_all.size(0), 10

        # # 7) gộp ảnh Top-10 + ground-truth + hiển thị điểm tương đồng (skip ground-truth)
        # font = ImageFont.load_default()
        # for t in save_types:
        #     topk_vals, topk_inds = sims_t2i[t].topk(topk, dim=1)
        #     folder = os.path.join(root, t)
        #     for i in tqdm(range(N), desc=f"Compositing {t}"):
        #         true_idx = i
        #         true_score = sims_t2i[t][i, i].item()
        #         retrieved = topk_inds[i].tolist()
        #         retrieved_scores = topk_vals[i].tolist()

        #         # paths and scores lists
        #         paths  = [im_paths_list[true_idx]] + [im_paths_list[idx] for idx in retrieved]
        #         scores = [true_score] + retrieved_scores

        #         thumbs = [thumb_cache[p] for p in paths]

        #         # build composite
        #         total_w = sum(im.width for im in thumbs)
        #         comp = Image.new("RGB", (total_w, target_h))
        #         x = 0
        #         for im_thumb in thumbs:
        #             comp.paste(im_thumb, (x, 0))
        #             x += im_thumb.width

        #         # downscale by half
        #         new_w, new_h = total_w // 2, target_h // 2
        #         comp = comp.resize((new_w, new_h), Image.BILINEAR)

        #         # draw similarity only on retrieved (skip idx=0)
        #         draw = ImageDraw.Draw(comp)
        #         x = 0
        #         for idx_thumb, (score, im_thumb) in enumerate(zip(scores, thumbs)):
        #             w_thumb = im_thumb.width // 2
        #             if idx_thumb == 0:
        #                 x += w_thumb
        #                 continue
        #             text = f"{score:.2f}"
        #             bbox = draw.textbbox((0, 0), text, font=font)
        #             tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        #             draw.rectangle([x, new_h - th, x + tw, new_h], fill=(0, 0, 0))
        #             draw.text((x, new_h - th), text, fill=(255, 255, 255), font=font)
        #             x += w_thumb

        #         # save image and caption
        #         out_img = os.path.join(folder, f"{i}.jpg")
        #         comp.save(out_img, format='JPEG', optimize=True, quality=85)
        #         idx_detail = int(t.split('_')[1]) - 1
        #         txt_path = os.path.join(folder, f"{i}.txt")
        #         with open(txt_path, 'w', encoding='utf-8') as f_txt:
        #             f_txt.write(detail_texts_list[idx_detail][i])

        # 8) tính metrics (giữ nguyên)
        target = torch.arange(N, device=im_feats_all.device)
        acc = {}
        acc_t2i = (sims_t2i['long'].argmax(dim=1) == target).sum().item() / N
        acc['long_t2i'] = acc_t2i
        for j in range(4):
            key = f"detail_{j+1}"
            acc[f"{key}_t2i"] = (sims_t2i[key].argmax(dim=1) == target).sum().item() / N
        sims_i2t_long = im_feats_all @ long_all.T
        acc_i2t = (sims_i2t_long.argmax(dim=1) == target).sum().item() / N
        acc['long_i2t'] = acc_i2t

        print(f"▶ Full Text→Image:     {acc_t2i:.4%}")
        print(f"▶ Image→Full Text:     {acc_i2t:.4%}")
        for j in range(4):
            print(f"▶ {j+1}th Sentence→Image:  {acc[f'detail_{j+1}_t2i']:.4%}")

        # 9) trả về list metrics
        return [
            acc_t2i,
            acc_i2t,
            acc['detail_1_t2i'],
            acc['detail_2_t2i'],
            acc['detail_3_t2i'],
            acc['detail_4_t2i'],
        ]
    

    def train(self, resume=False, warmup_length=200):
        # deterministic 80/20 split
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)

        # ckpt_path = "/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/longclip/lr=1e-06_wd=0.01_wl=200_logs=4.6052_64xb/ckpt/Propose-b16-longclip-04-23--03_58_18_-local05_2for_tau007.pt"
        # print(f"Loading checkpoint from {ckpt_path} ...")
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # # Nếu bạn chỉ lưu state_dict của model, dùng:
        # self.model.load_state_dict(state_dict)
        # # Nếu bạn lưu cả optimizer và scheduler state, ví dụ:
        # # ckpt = torch.load(ckpt_path, map_location="cpu")
        # # self.model.load_state_dict(ckpt["model_state_dict"])
        # # self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # # self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # print("Checkpoint loaded. Resuming training...")

        print("tạo train test")

        

        train_ds, test_ds = dataset_mapping[self.train_data]()


        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True
        )

        print("tạo train test xong")

        self.scheduler = cosine_lr(
            self.optimizer,
            base_lr=self.lr,
            warmup_length=warmup_length,
            steps=self.num_epoch * len(train_loader)
        )
        finegrained_loss=False
        for epoch in range(self.num_epoch):
            # train
            self.model.train()
            if self.use_finegrained_loss:
                avg_loss, avg_fg, avg_mp = self.train_epoch(train_loader, epoch, finegrained_loss=True)
            else:
                avg_loss = self.train_epoch(train_loader, epoch)

            # save checkpoint
            now = datetime.now().strftime("%m-%d--%H_%M_%S_")
            if args.base_model=="ViT-B/16":
                ckpt_name = f"B16-longclip-{now}-{epoch}.pt"
            else:
                ckpt_name= f"L14-longclip-{now}-{epoch}.pt"
            # ckpt_name = f"Propose-b16-longclip-{now}-share4v-no.pt"
            if epoch in [6, 8, 9, 10]:
                torch.save(self.model.state_dict(), os.path.join(self.ckptdir, ckpt_name))
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

            # test after epoch
            acc_t2i, acc_i2t, acc_first, acc_second, acc_third, acc_fourth = self.test_epoch_ver4(test_loader, epoch)
            
            # log
            if finegrained_loss:
                self.results.append({
                    "epoch":        epoch,
                    "avg_loss":     avg_loss,
                    "avg_finegrained_loss":    avg_fg,
                    "avg_mp_loss":   avg_mp,
                    "test_acc_t2i": acc_t2i,
                    "test_acc_i2t": acc_i2t
                })
            else:
                self.results.append({
                    "epoch":        epoch,
                    "avg_loss":     avg_loss,
                    "test_acc_t2i": acc_t2i,
                    "test_acc_i2t": acc_i2t
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
    parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per gpu.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--resume", default=False, action='store_true', help="resume training from checkpoint.")
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    parser.add_argument("--is_sparsemax", default=False, help="Using sparsemax in loss")
    parser.add_argument("--train_data", type=str,default="docci", help="Which data to train")
    parser.add_argument("--use_finegrained", action="store_true", help="Enable finegrained loss. Default will be False")
    parser.add_argument("--learnable_mps", action="store_true", help="Enable finegrained loss. Default will be False")


    args = parser.parse_args()
    print("DDP Done")

    trainer = CLIP_Clean_Train(args=args)
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
