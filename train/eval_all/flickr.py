# # ===== Zero-shot retrieval eval (Flickr30k) - dùng model & preprocess từ cell 1 =====
# import os, json
# import pandas as pd
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# from types import SimpleNamespace
# from tqdm import tqdm

# import sys
# sys.path.append('../..')
# from model import longclip  # dùng tokenizer của LongCLIP

# # -----------------------------
# # Dataset: dùng preprocess do bạn truyền vào (không tự clip.load)
# # -----------------------------
# class Flickr30kPairDataset(Dataset):
#     """
#     Mỗi hàng là 1 (image, caption) sample. Mặc định lấy caption[0] từ cột 'raw' (list captions).
#     Trả về: image_tensor, caption_long, caption_short, img_path, split
#     """
#     def __init__(
#         self,
#         csv_path: str,
#         root_dir: str,
#         preprocess,               # bắt buộc: transform từ cell 1
#         split: str = None,        # 'train' | 'val' | 'test' | None (lấy tất)
#         max_items: int = None,
#     ):
#         super().__init__()
#         self.preprocess = preprocess

#         df = pd.read_csv(csv_path)
#         if split is not None:
#             df = df[df["split"] == split]
#         if max_items is not None:
#             df = df.iloc[:max_items]
#         df = df.reset_index(drop=True)

#         samples = []
#         for _, row in df.iterrows():
#             img_path = os.path.join(root_dir, "flickr30k-images", row["filename"])
#             caps = json.loads(row["raw"])
#             cap0 = caps[0].strip() if isinstance(caps, list) and len(caps) > 0 else str(row.get("sentence", "")).strip()
#             samples.append({
#                 "img_path": img_path,
#                 "caption": cap0,
#                 "split": row["split"],
#                 "img_id": row.get("img_id", None),
#             })
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         image = Image.open(item["img_path"]).convert("RGB")
#         image_tensor = self.preprocess(image)
#         caption_long = item["caption"]
#         caption_short = item["caption"]
#         return image_tensor, caption_long, caption_short, item["img_path"], item["split"]


# # -----------------------------
# # Helpers: encode wrappers & tách câu chi tiết
# # -----------------------------
# def _encode_image(model, images):
#     out = model.encode_image(images)
#     return out[0] if isinstance(out, (tuple, list)) else out

# def _encode_text(model, tokens):
#     out = model.encode_text(tokens)
#     return out[0] if isinstance(out, (tuple, list)) else out

# def split_into_detail_captions(text: str, max_details: int = 4):
#     """
#     Tách caption dài thành tối đa `max_details` câu ngắn.
#     Mặc định: tách theo dấu câu . ! ? ; : hoặc xuống dòng. Nếu thiếu thì pad "".
#     Bạn có thể thay bằng splitter riêng của bạn nếu có.
#     """
#     if not isinstance(text, str):
#         text = str(text)
#     # tách thô, giữ nội dung ngắn gọn
#     import re
#     parts = [p.strip() for p in re.split(r'[\.!\?;:\n]+', text) if p.strip()]
#     parts = parts[:max_details]
#     while len(parts) < max_details:
#         parts.append("")
#     return parts


# # -----------------------------
# # Evaluation core (thay cho trainer.test_epoch)
# # -----------------------------
# def test_epoch_with_model(model, dataloader, device, max_details: int = 4):
#     model.eval()
#     model = model.to(device)

#     im_feats_list     = []
#     im_paths_list     = []
#     long_feats_list   = []
#     txt_feats_lists   = {j: [] for j in range(max_details)}

#     with torch.no_grad():
#         for images, long_texts, _, image_paths, _ in tqdm(dataloader, desc="Extracting features"):
#             im_paths_list.extend(image_paths)
#             images = images.to(device, non_blocking=True)

#             # Image features
#             feats_i = _encode_image(model, images)
#             feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
#             im_feats_list.append(feats_i)

#             # Full-text features
#             tokens_long = longclip.tokenize(list(long_texts), truncate=True).to(device)
#             feats_long = _encode_text(model, tokens_long)
#             feats_long = feats_long / feats_long.norm(dim=-1, keepdim=True)
#             long_feats_list.append(feats_long)

#             # Detail captions (per batch)
#             caps_batch = [split_into_detail_captions(t, max_details=max_details) for t in long_texts]
#             for j in range(max_details):
#                 texts_j = [c[j] if j < len(c) else "" for c in caps_batch]
#                 toks_j = longclip.tokenize(texts_j, truncate=True).to(device)
#                 fts_j = _encode_text(model, toks_j)
#                 fts_j = fts_j / fts_j.norm(dim=-1, keepdim=True)
#                 txt_feats_lists[j].append(fts_j)

#     # Concat all features
#     im_feats_all   = torch.cat(im_feats_list, dim=0)
#     long_all       = torch.cat(long_feats_list, dim=0)
#     txt_all_lists  = [torch.cat(txt_feats_lists[j], dim=0) for j in range(max_details)]

#     # Similarity matrices
#     sims_t2i = {
#         "long":    long_all @ im_feats_all.T,
#         **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(max_details)}
#     }
#     sims_i2t_long = im_feats_all @ long_all.T

#     # Recall@K
#     N = im_feats_all.size(0)
#     target = torch.arange(N, device=im_feats_all.device)
#     ks = [1, 5, 25, 50]

#     acc = {}
#     # Text → Image
#     for name, sims in sims_t2i.items():
#         for k in ks:
#             topk_inds = sims.topk(k, dim=1).indices
#             hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
#             acc[f"{name}_t2i_R{k}"] = hits

#     # Image → Text (full)
#     for k in ks:
#         topk_inds = sims_i2t_long.topk(k, dim=1).indices
#         hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
#         acc[f"long_i2t_R{k}"] = hits

#     # Print summary
#     print("\n—— Test Epoch Metrics ——")
#     for k in ks:
#         print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
#     for k in ks:
#         print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
#     print("—" * 30)
#     return acc


# # -----------------------------
# # Runner cho notebook
# # -----------------------------
# def run_flickr30k_eval(
#     model,
#     preprocess,
#     csv_path="/home/ubuntu/shared/hieu.tq/flickr30k/flickr_annotations_30k.csv",
#     root_dir="/home/ubuntu/shared/hieu.tq/flickr30k",
#     split="test",
#     batch_size=64,
#     num_workers=32,
#     device=None,
#     max_items=None,
#     max_details=4,
# ):
#     """
#     Dùng model & preprocess đã có (cell 1). Không cần CLIP_Clean_Train hay load ckpt.
#     """
#     assert preprocess is not None, "`preprocess` must be provided."

#     # ✅ Quan trọng: reset default device về CPU để DataLoader/transform không đòi CUDA generator
#     try:
#         # Chỉ có từ PyTorch 2.0+, nếu không có thì bỏ qua
#         import torch
#         if hasattr(torch, "set_default_device"):
#             torch.set_default_device("cpu")
#     except Exception:
#         pass

#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#     test_ds = Flickr30kPairDataset(
#         csv_path=csv_path,
#         root_dir=root_dir,
#         preprocess=preprocess,
#         split=split,
#         max_items=max_items,
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=(num_workers > 0),
#     )

#     print(f"Số mẫu trong test set flick30k: {len(test_loader.dataset)}")
#     print(f"Số batch trong test loader: {len(test_loader)}")

#     metrics = test_epoch_with_model(model, test_loader, device=device, max_details=max_details)
#     print(f"Individual metrics: {metrics}")
#     return metrics

# ===== Flickr30k retrieval eval (multi-caption-per-image, cached image feats) =====
import os, sys, json
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Cho phép import module model/longclip ở repo gốc
sys.path.append('../..')
from model import longclip  # dùng tokenizer của LongCLIP

# ────────────────────────────────────────────────────────────────────────────────
# Dataset: MỖI CAPTION là 1 sample (giống COCO multi-caption)
# Trả về: image_tensor, caption (str), image_id (str)
# ────────────────────────────────────────────────────────────────────────────────
class Flickr30kAllCaptionsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        preprocess,               # bắt buộc: transform từ cell 1 / model.preprocess
        split: Optional[str] = "test",  # 'train' | 'val' | 'test' | None (lấy tất)
        max_items: Optional[int] = None,
    ):
        super().__init__()
        assert preprocess is not None, "`preprocess` must be provided."
        self.preprocess = preprocess

        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df["split"] == split].copy()

        samples = []
        for _, row in df.iterrows():
            filename = str(row["filename"])
            img_path = os.path.join(root_dir, "flickr30k-images", filename)

            # id ảnh: ưu tiên cột 'img_id' nếu có, nếu không dùng stem của filename
            img_id = str(row["img_id"]) if "img_id" in row and pd.notna(row["img_id"]) \
                    else Path(filename).stem

            # cột 'raw' là list JSON các caption
            caps = []
            try:
                caps = json.loads(row["raw"]) if pd.notna(row["raw"]) else []
                if not isinstance(caps, list):
                    caps = [str(caps)]
            except Exception:
                # fallback: nếu không parse được, thử cột 'sentence'
                sent = row.get("sentence", "")
                caps = [str(sent)] if pd.notna(sent) and str(sent).strip() else []

            # thêm từng caption thành 1 sample
            for cap in caps:
                cap = str(cap).replace("\n", " ").strip()
                if not cap:
                    continue
                samples.append({
                    "img_path": img_path,
                    "image_id": img_id,
                    "caption": cap,
                })

        if max_items is not None:
            samples = samples[: max_items]

        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["img_path"]).convert("RGB")
        return self.preprocess(image), s["caption"], s["image_id"]


# ────────────────────────────────────────────────────────────────────────────────
# Encode helpers: an toàn nếu model.encode_* trả tuple (feats, extras)
# ────────────────────────────────────────────────────────────────────────────────
def _encode_image(model, images):
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens):
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out


# ────────────────────────────────────────────────────────────────────────────────
# Evaluation (multi-caption-per-image safe, cache image features)
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_flickr30k_multicap(model, dataloader, device="cuda"):
    model.eval()
    model = model.to(device)

    def _cpu_fp32(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to("cpu", dtype=torch.float32)

    img_feat_cache: Dict[str, torch.Tensor] = {}
    txt_feats_parts: List[torch.Tensor] = []
    cap2img: List[str] = []

    for imgs, caps, img_ids in tqdm(dataloader, desc="Extracting features"):
        # 1) Encode ảnh (chỉ ảnh mới)
        new_ids, new_imgs = [], []
        for _img, _id in zip(imgs, img_ids):
            if _id not in img_feat_cache:
                new_ids.append(_id); new_imgs.append(_img)
        if new_imgs:
            batch_imgs = torch.stack(new_imgs).to(device, non_blocking=True)
            feats_i = F.normalize(_encode_image(model, batch_imgs), dim=-1)
            feats_i = _cpu_fp32(feats_i)
            for _id, _f in zip(new_ids, feats_i):
                img_feat_cache[_id] = _f

        # 2) Encode text
        tokens = longclip.tokenize(list(caps), truncate=True).to(device)
        feats_t = F.normalize(_encode_text(model, tokens), dim=-1)
        txt_feats_parts.append(_cpu_fp32(feats_t))
        cap2img.extend(img_ids)

    # Gộp tensor trên CPU FP32
    img_ids_list = list(img_feat_cache.keys())
    img_feats = torch.stack([img_feat_cache[k] for k in img_ids_list]).to(torch.float32)  # [I, D]
    txt_feats = torch.cat(txt_feats_parts, dim=0).to(torch.float32)                       # [M, D]

    # Similarity (CPU FP32)
    sims_t2i = txt_feats @ img_feats.T   # [M, I]
    sims_i2t = img_feats @ txt_feats.T   # [I, M]

    # Recall@K (multi-caption safe)
    ks = (1, 5, 10, 25, 50)
    recalls_t2i, recalls_i2t = [], []

    # T→I: đúng nếu ảnh top-k có image_id == cap2img[i]
    for k in ks:
        topk = sims_t2i.topk(k, dim=1).indices  # [M, k]
        correct = 0
        for i, row in enumerate(topk):
            if any(img_ids_list[j] == cap2img[i] for j in row.tolist()):
                correct += 1
        recalls_t2i.append(correct / len(cap2img))

    # I→T: đúng nếu top-k có ÍT NHẤT MỘT caption thuộc ảnh đó
    imgid2capidx: Dict[str, List[int]] = {}
    for cap_idx, iid in enumerate(cap2img):
        imgid2capidx.setdefault(iid, []).append(cap_idx)

    for k in ks:
        topk = sims_i2t.topk(k, dim=1).indices  # [I, k]
        correct = 0
        for i, row in enumerate(topk):
            iid = img_ids_list[i]
            caps_of_img = imgid2capidx.get(iid, [])
            if any(idx in caps_of_img for idx in row.tolist()):
                correct += 1
        recalls_i2t.append(correct / len(img_ids_list))

    return {"T→I": dict(zip(ks, recalls_t2i)),
            "I→T": dict(zip(ks, recalls_i2t))}


# ────────────────────────────────────────────────────────────────────────────────
# Runner cho notebook (giống COCO)
# ────────────────────────────────────────────────────────────────────────────────
def run_flickr30k_eval_allcaps(
    model,
    preprocess=None,
    csv_path="/home/ubuntu/shared/hieu.tq/flickr30k/flickr_annotations_30k.csv",
    root_dir="/home/ubuntu/shared/hieu.tq/flickr30k",
    split="test",
    batch_size=64,
    num_workers=16,
    device=None,
    max_items=None,  # giới hạn tổng số CAPTIONS (không phải số ảnh)
):
    """
    Đánh giá Flickr30k trên TẤT CẢ captions (5/cái ảnh).
    Dùng model (và preprocess) đã load ở cell 1. Nếu preprocess=None → lấy model.preprocess.
    """
    if preprocess is None:
        preprocess = getattr(model, "preprocess", None)
    assert preprocess is not None, "Please provide `preprocess` or attach `model.preprocess`."

    # ✅ Reset default device về CPU để DataLoader/transform không đòi CUDA generator (nếu có)
    try:
        import torch as _torch
        if hasattr(_torch, "set_default_device"):
            _torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = Flickr30kAllCaptionsDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        preprocess=preprocess,
        split=split,
        max_items=max_items,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Thống kê: số captions & số ảnh duy nhất
    n_caps = len(ds)
    # Duyệt nhanh để đếm ảnh duy nhất mà không đọc lại file
    img_ids = set()
    for _, _, iid in dl:  # chỉ lấy ids từ batch 1 (nhanh)
        img_ids.update(list(iid))
    n_imgs_est = len(img_ids)

    print(f"Flickr30k ({split}) — {n_caps:,} captions (ước tính ~{n_imgs_est:,} images)")
    metrics = evaluate_flickr30k_multicap(model, dl, device=device)

    print("\nRecall@K FLICKR")
    for mode, d in metrics.items():
        print(f"  {mode:3s} : " + "  ".join(f"R@{k} = {v*100:.2f}%" for k, v in d.items()))
    return metrics

