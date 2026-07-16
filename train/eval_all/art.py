# eval_artpedia_retrieval.py
# -*- coding: utf-8 -*-
"""
ArtPedia retrieval evaluation (docci-like format).
Mirrors the DOCCI eval style you shared: same metrics, same tokenizer (LongCLIP),
and the same "detail_{j}" split of long captions into short sentences.

Usage (from a notebook / script):
    from eval_artpedia_retrieval import run_artpedia_eval
    metrics = run_artpedia_eval(model, preprocess, split="test", batch_size=64, num_workers=32)

Requirements:
    - artpedia_docci_like.py available on PYTHONPATH with class `ArtPediaDataset`
    - LongCLIP tokenizer available as: from model_sail import longclip  (fallback: from model import longclip)
"""
import os
from typing import Optional, List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# LongCLIP tokenizer (same pattern as in your DOCCI cell)
try:
    from model_sail import longclip  # type: ignore
except Exception:
    from model import longclip  # type: ignore

# ArtPedia dataset in "docci-like" format (as provided)
# import sys
# sys.path.append('..')
# from art import ArtPediaDataset

# artpedia_docci_like.py
import os
import re
import json
import hashlib
import unicodedata
from urllib.parse import unquote
from typing import List, Optional, Tuple, Dict

from PIL import Image, ImageFile
import torch.utils.data as data
import clip

# ---- robust image loading (giống bạn đang dùng) ----
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# =========================
# Helpers
# =========================
def _strip_accents(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def _first_sentence(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[\.!\?])\s+", text, maxsplit=1)
    first = (parts[0] if parts else text).strip()
    if first and first[-1] not in ".!?":
        first += "."
    return first

def _norm_split_art(name: str) -> str:
    """
    Chuẩn hoá split của ArtPedia về {train, val, test}.
    """
    n = (name or "").strip().lower()
    if n in {"val", "valid", "validation", "dev", "qual_train"}:
        return "val"
    if n in {"test", "qual_test"}:
        return "test"
    return "train" if n == "train" else n

def _load_records(json_path: str) -> List[dict]:
    """
    Đọc JSON ArtPedia và trả về list[dict].
    Hỗ trợ:
      1) [ {...}, {...} ]
      2) { "data": [ {...} ] }
      3) { "2": {...}, "3": {...} }  (dict-of-dicts), tự chèn 'id'=key nếu thiếu.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]

    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return [r for r in raw["data"] if isinstance(r, dict)]

    if isinstance(raw, dict):
        out = []
        for k, v in raw.items():
            if isinstance(v, dict):
                if "id" not in v:
                    vv = v.copy()
                    vv["id"] = k
                    out.append(vv)
                else:
                    out.append(v)
        if out:
            return out

    raise ValueError(f"Unrecognized JSON format at '{json_path}'")

def _guess_image_path(item: dict, image_root: str) -> str:
    """
    Map ArtPedia record -> local image path.
    Ưu tiên id.jpg/jpeg/png/webp (vì thư mục images đặt theo số),
    fallback theo basename của img_url, rồi thử title+year.
    """
    # 0) theo id
    rid = item.get("id")
    if rid is None and "ID" in item:
        rid = item["ID"]
    if rid is not None:
        rid_str = str(rid).strip()
        if rid_str:
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
                p = os.path.join(image_root, rid_str + ext)
                if os.path.exists(p):
                    return p
            if rid_str.isdigit():
                rid_no_zero = str(int(rid_str))
                for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
                    p = os.path.join(image_root, rid_no_zero + ext)
                    if os.path.exists(p):
                        return p

    # 1) theo img_url
    url = item.get("img_url") or ""
    base = unquote(url.strip().split("/")[-1]) if url else ""
    base = base.strip()
    if base:
        for cand in [base, base.replace(" ", "_"), base.replace("%20", "_")]:
            p = os.path.join(image_root, cand)
            if os.path.exists(p):
                return p
        root, _ext = os.path.splitext(base)
        for ext2 in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
            p = os.path.join(image_root, root + ext2)
            if os.path.exists(p):
                return p

    # 2) theo title+year (fallback)
    title = (item.get("title") or "unknown").strip()
    year  = str(item.get("year") or "").strip()
    safe  = re.sub(r"[^A-Za-z0-9_\-]+", "_", _strip_accents(title))
    base2 = f"{safe}_{year}" if year else safe
    for ext2 in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
        p = os.path.join(image_root, base2 + ext2)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Missing image for id/url='{item.get('id') or item.get('img_url')}'")


def _make_caption_from_art(item: dict, include_title: bool = True, text_mode: str = "both") -> str:
    """
    Tạo caption 'dài' cho ArtPedia từ contextual + visual (giống phong cách bạn đang dùng).
    """
    visual = [ (s or "").replace("\n", " ").strip() for s in (item.get("visual_sentences") or []) if (s or "").strip() ]
    context = [ (s or "").replace("\n", " ").strip() for s in (item.get("contextual_sentences") or []) if (s or "").strip() ]

    if text_mode == "visual":
        parts = visual
    elif text_mode == "contextual":
        parts = context
    else:
        parts = visual + context

    if include_title:
        t = (item.get("title") or "").strip()
        if t:
            parts = [t + "."] + parts

    cap = " ".join(parts).strip()
    cap = re.sub(r"\s+", " ", cap)
    return cap

def _make_caption_short_from_art(item: dict) -> str:
    vs = item.get("visual_sentences", []) or []
    cs = item.get("contextual_sentences", []) or []
    candidate = (vs[0] if len(vs) > 0 else (cs[0] if len(cs) > 0 else ""))
    return _first_sentence(candidate)


# =========================
# ArtPedia loader (docci-like format)
# =========================
# Gợi ý đường dẫn ArtPedia
art_root   = "/home/ubuntu/shared/ARO/artpedia/"
json_path  = os.path.join(art_root, "artpedia_filtered.json")   # hoặc 'artpedia.json' nếu bạn muốn
image_root = os.path.join(art_root, "images")

class ArtPediaDataset(data.Dataset):
    def __init__(self, split: str, max_items: int = None):
        """
        Trả về format GIỐNG DocciDataset (ở project kia):
            __getitem__ -> (image_tensor, caption, caption_short, img_path, split)

        split: one of {'train','qual_train','test','qual_test'}
          - 'train'       -> ArtPedia split == 'train'
          - 'qual_train'  -> ArtPedia split == 'val'
          - 'test'        -> ArtPedia split in {'test','val'}  (mở rộng như docci 'test' gồm cả 'qual_test')
          - 'qual_test'   -> ArtPedia split == 'test'
        """
        print(f"[ArtPediaDocciLikeDataset] json path: {json_path}")
        valid = ('train', 'qual_train', 'test', 'qual_test')
        if split not in valid:
            raise ValueError(f"Unsupported split: {split}. Use one of {valid}.")

        # map split (docci-style) -> artpedia splits
        if split == 'train':
            target_splits = {'train'}
        elif split == 'qual_train':
            target_splits = {'val'}
        elif split == 'test':
            # target_splits = {'test', 'val'}
            target_splits = {'test'}
        else:  # 'qual_test'
            target_splits = {'test'}

        # load records
        all_items = _load_records(json_path)
        # filter by split
        self.samples: List[dict] = [d for d in all_items if _norm_split_art(d.get('split', 'train')) in target_splits]

        if max_items is not None:
            self.samples = self.samples[:max_items]

        # preprocess (giống docci)
        _, self.preprocess = clip.load("ViT-B/16")
        # _, self.preprocess = clip.load("ViT-L/14")

        self.image_root = image_root

        if len(self.samples) == 0:
            raise ValueError(f"[ArtPediaDocciLikeDataset] Empty after filtering split='{split}' -> {target_splits}")

        print(f"[ArtPediaDocciLikeDataset] split_map={target_splits} | kept={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        it = self.samples[idx]

        # caption dài & ngắn (giống pattern bạn hay dùng)
        caption = _make_caption_from_art(it, include_title=True, text_mode="both")
        # caption = _make_caption_from_art(it, include_title=True, text_mode="both")
        # caption = _make_caption_from_art(it, include_title=True, text_mode="visual")
        caption_short = _make_caption_short_from_art(it)

        # ảnh
        img_path = _guess_image_path(it, self.image_root)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        # trả về format y hệt docci ở project kia:
        # (image_tensor, caption, caption_short, img_path, item['split'])
        return image_tensor, caption, caption_short, img_path, it.get('split', 'train')


# -----------------------------
# DataLoader builder
# -----------------------------
def build_loader(dataset, batch_size: int, num_workers: int = 8) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# -----------------------------
# Helpers (same as your DOCCI cell)
# -----------------------------
def _encode_image(model, images):
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens):
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out

def split_into_detail_captions(text: str, max_details: int = 4):
    """
    Split long caption into <= max_details short sentences by . ! ? ; : or newlines.
    Pads with "" if fewer parts.
    """
    import re
    if not isinstance(text, str):
        text = str(text)
    parts = [p.strip() for p in re.split(r'[\.!\?;:\n]+', text) if p.strip()]
    parts = parts[:max_details]
    while len(parts) < max_details:
        parts.append("")
    return parts


# -----------------------------
# Evaluation core (generic, works for ArtPedia docci-like)
# -----------------------------
@torch.no_grad()
def _eval_epoch_generic(model, dataloader, device, max_details: int = 4):
    model.eval()
    model = model.to(device)

    im_feats_list   = []
    long_feats_list = []
    txt_feats_lists = {j: [] for j in range(max_details)}  # detail sentences

    for batch in tqdm(dataloader, desc="Extracting features (ArtPedia)"):
        # Expected batch layout (docci-like):
        #   images, long_texts, short_texts, img_path, split
        images, long_texts = batch[0], batch[1]
        images = images.to(device, non_blocking=True)

        # Image features
        feats_i = _encode_image(model, images)
        feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
        im_feats_list.append(feats_i)

        # Full-text (long) features
        tokens_long = longclip.tokenize(list(long_texts), truncate=True).to(device)
        feats_long = _encode_text(model, tokens_long)
        feats_long = feats_long / feats_long.norm(dim=-1, keepdim=True)
        long_feats_list.append(feats_long)

        # Detail captions from long text
        caps_batch = [split_into_detail_captions(t, max_details=max_details) for t in long_texts]
        for j in range(max_details):
            texts_j = [c[j] if j < len(c) else "" for c in caps_batch]
            toks_j = longclip.tokenize(texts_j, truncate=True).to(device)
            fts_j = _encode_text(model, toks_j)
            fts_j = fts_j / fts_j.norm(dim=-1, keepdim=True)
            txt_feats_lists[j].append(fts_j)

    # Concatenate
    im_feats_all  = torch.cat(im_feats_list, dim=0)
    long_all      = torch.cat(long_feats_list, dim=0)
    txt_all_lists = [torch.cat(txt_feats_lists[j], dim=0) for j in range(max_details)]

    # Similarities
    sims_t2i = {
        "long":    long_all @ im_feats_all.T,
        **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(max_details)}
    }
    sims_i2t_long = im_feats_all @ long_all.T

    # Recall@K (1-1 mapping image<->caption by row index)
    N = im_feats_all.size(0)
    target = torch.arange(N, device=im_feats_all.device)
    ks = [1, 5, 25, 50]

    acc = {}
    # Text→Image (full + detail)
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

    # Print summary in the same style as your DOCCI cell
    print("\n—— ArtPedia (docci-like) Metrics ——")
    for k in ks:
        print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
    for k in ks:
        print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
    print("—" * 30)
    return acc


# -----------------------------
# Public runner (mirrors your run_docci_eval)
# -----------------------------
def run_artpedia_eval(
    model,
    preprocess,  # kept for API parity; ArtPediaDataset already applies its own preprocess internally
    split: str = "test",  # 'train' | 'qual_train' | 'test' (test+val) | 'qual_test' (test)
    batch_size: int = 64,
    num_workers: int = 32,
    device: Optional[str] = None,
    max_items: Optional[int] = None,
    max_details: int = 4,
):
    """
    Uses the 'model' & 'preprocess' from your main cell.
    Mirrors the DOCCI runner but evaluates on ArtPedia docci-like dataset.

    Notes:
      * ArtPediaDataset already returns (image_tensor, caption_long, caption_short, img_path, split)
      * The 'preprocess' argument is kept to match your API, but is not used here since the dataset applies its own.
    """
    # Ensure default device is CPU for transforms/DataLoader workers
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = ArtPediaDataset(split=split, max_items=max_items)
    dl = build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    print(f"ArtPedia split: {split}")
    print(f"Num samples: {len(ds)}")
    print(f"Num batches: {len(dl)}")

    metrics = _eval_epoch_generic(model, dl, device=device, max_details=max_details)
    print(f"Individual metrics: {metrics}")
    return metrics


# -----------------------------
# Optional: quick smoke test (requires a valid model & artpedia paths)
# -----------------------------
if __name__ == "__main__":
    print("This module exposes run_artpedia_eval(model, preprocess, ...).")
    print("Import it and call from your training/eval notebook.")
