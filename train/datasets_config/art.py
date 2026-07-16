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
        # caption = _make_caption_from_art(it, include_title=True, text_mode="visual")
        caption_short = _make_caption_short_from_art(it)

        # ảnh
        img_path = _guess_image_path(it, self.image_root)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        # trả về format y hệt docci ở project kia:
        # (image_tensor, caption, caption_short, img_path, item['split'])
        return image_tensor, caption, caption_short, img_path, it.get('split', 'train')


# =========================
# Quick test
# =========================
if __name__ == "__main__":
    ds = ArtPediaDataset(split="test", max_items=5)
    print("len:", len(ds))
    x = ds[0]
    print(type(x), len(x))
    print("caption_short:", x[2][:120])
    print("path:", x[3])
    print("orig split:", x[4])
