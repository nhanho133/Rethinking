# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# PixmoDataset: dataloader giống DocciDataset nhưng cho cấu trúc:
#   <pixmo_root>/
#     README.md
#     data/
#     images/
#     pixmo_manifest.jsonl     # JSON Lines: {image_url, image_path, caption, transcripts, ...}

# - Chia 80/20 theo seed (deterministic).
# - Hỗ trợ gọi split như Docci: train/qual_train/test/qual_test.
# - Trả về tuple: (image_tensor, caption, caption_short, img_path, split)

# Ví dụ dùng:
#     from torch.utils.data import DataLoader
#     ds_train = PixmoDataset(
#         root="/home/ubuntu/shared/hieu.tq/molmo/pixmo-cap",
#         split="train", seed=123, max_items=None, base_model="ViT-B/16"
#     )
#     ds_val   = PixmoDataset(
#         root="/home/ubuntu/shared/hieu.tq/molmo/pixmo-cap",
#         split="qual_test", seed=123
#     )
#     train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
#     val_loader   = DataLoader(ds_val,   batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
# """

# import json
# import os
# import random
# from pathlib import Path
# from typing import List, Dict, Optional

# from PIL import Image
# import clip
# import torch.utils.data as data
# from PIL import Image, ImageFile, UnidentifiedImageError
# ImageFile.LOAD_TRUNCATED_IMAGES = True


# def _read_jsonl(path: str) -> List[Dict]:
#     items = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#                 items.append(obj)
#             except Exception:
#                 # bỏ qua dòng lỗi
#                 continue
#     return items


# class PixmoDataset(data.Dataset):
#     def __init__(
#         self,
#         root: str,
#         split: str,
#         seed: int = 42,
#         max_items: Optional[int] = None,
#         base_model: str = "ViT-B/16",
#         manifest_name: str = "pixmo_manifest.jsonl",
#         filter_status_ok: bool = True,
#         strict_exists: bool = True,
#     ):
#         """
#         root: thư mục gốc của pixmo-cap (chứa images/ và pixmo_manifest.jsonl)
#         split: 'train' | 'qual_train' | 'test' | 'qual_test'
#         seed : seed cho phép chia 80/20 ổn định
#         max_items: giới hạn số mẫu sau khi đã chọn split
#         base_model: backbone để lấy CLIP preprocess
#         filter_status_ok: nếu True thì chỉ nhận các bản ghi status == 'ok'
#         strict_exists: nếu True thì chỉ nhận ảnh có file tồn tại
#         """
#         valid_splits = ("train", "qual_train", "test", "qual_test")
#         if split not in valid_splits:
#             raise ValueError(f"Unsupported split: {split}. Use one of {valid_splits}.")

#         self.root = Path(root)
#         self.manifest_path = self.root / manifest_name

#         if not self.manifest_path.is_file():
#             raise FileNotFoundError(f"Not found: {self.manifest_path}")

#         # 1) Load manifest JSONL
#         full_data = _read_jsonl(str(self.manifest_path))

#         # 2) Lọc theo status và tồn tại file
#         cleaned = []
#         for d in full_data:
#             if filter_status_ok and d.get("status") not in (None, "", "ok"):
#                 continue
#             # image_path trong manifest là kiểu "images/<sha1>.jpg"
#             rel_path = d.get("image_path")
#             if not isinstance(rel_path, str) or not rel_path:
#                 continue
#             img_path = self.root / rel_path
#             if strict_exists and not img_path.is_file():
#                 continue

#             caption = (d.get("caption") or "").replace("\n", " ").strip()
#             cleaned.append({
#                 "image": rel_path,               # relative to root
#                 "caption": caption,
#                 "transcripts": d.get("transcripts", []),
#                 "source_url": d.get("image_url", ""),
#             })

#         if not cleaned:
#             raise RuntimeError("No usable records after filtering. Check paths/status.")

#         # 3) Chia 80/20 theo seed (deterministic)
#         #    - train_indices = 80%
#         #    - val_indices   = 20% (gắn nhãn 'qual_test')
#         N = len(cleaned)
#         idxs = list(range(N))
#         rng = random.Random(seed)
#         rng.shuffle(idxs)

#         n_train = int(0.8 * N)
#         train_set = set(idxs[:n_train])
#         val_set = set(idxs[n_train:])

#         # Map kiểu Docci
#         if split in ("train", "qual_train"):
#             target_split_name = "train"
#             chosen = [cleaned[i] for i in sorted(train_set)]
#         else:  # split in ("test", "qual_test")
#             target_split_name = "qual_test"
#             chosen = [cleaned[i] for i in sorted(val_set)]

#         # 4) Nếu cần giới hạn số mẫu
#         if max_items is not None:
#             chosen = chosen[:max_items]

#         self.samples = chosen
#         self.target_split_name = target_split_name

#         # 5) Load CLIP preprocess once
#         _, self.preprocess = clip.load(base_model)

#         # giữ root để join path khi __getitem__
#         self.image_root = self.root

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]

#         caption = item["caption"]
#         # caption ngắn = câu đầu (fallback: nếu không có dấu chấm, dùng caption luôn)
#         if "." in caption:
#             caption_short = caption.split(".")[0].strip() + "."
#         else:
#             caption_short = caption

#         img_path = os.path.join(str(self.image_root), item["image"])
#         image = Image.open(img_path).convert("RGB")
#         image_tensor = self.preprocess(image)

#         # trả về giống DocciDataset
#         return image_tensor, caption, caption_short, img_path, self.target_split_name

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # đọc được nhiều JPEG/GIF bị cắt

import clip
import torch.utils.data as data


def _read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    return items


class PixmoDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        seed: int = 42,
        max_items: Optional[int] = None,
        base_model: str = "ViT-B/16",
        # base_model: str = "ViT-L/14",
        manifest_name: str = "pixmo_manifest.jsonl",
        filter_status_ok: bool = True,
        strict_exists: bool = True,
        on_error: str = "skip",           # 'skip' | 'raise'
        max_skip_attempts: int = 10,      # số lần thử thay mẫu nếu ảnh hỏng
        min_filesize_bytes: int = 512,    # lọc file quá nhỏ (thường là lỗi)
    ):
        """
        - split: 'train' | 'qual_train' | 'test' | 'qual_test' (như Docci)
        - 80/20 chia theo seed (deterministic)
        - on_error='skip': nếu ảnh hỏng, dataset tự chọn mẫu khác thay thế (giữ batch đủ)
        """
        valid_splits = ("train", "qual_train", "test", "qual_test")
        if split not in valid_splits:
            raise ValueError(f"Unsupported split: {split}. Use one of {valid_splits}.")

        self.root = Path(root)
        self.manifest_path = self.root / manifest_name
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Not found: {self.manifest_path}")

        self.on_error = on_error
        self.max_skip_attempts = max_skip_attempts

        # 1) Load manifest
        full_data = _read_jsonl(str(self.manifest_path))

        # 2) Lọc bản ghi usable
        cleaned = []
        images_dir = self.root  # image_path trong manifest đã là 'images/<sha1>.<ext>'
        for d in full_data:
            if filter_status_ok and d.get("status") not in (None, "", "ok"):
                continue
            rel_path = d.get("image_path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            img_path = images_dir / rel_path
            if strict_exists and not img_path.is_file():
                continue
            # lọc file quá nhỏ (hay gặp khi 404/HTML)
            try:
                if img_path.is_file() and img_path.stat().st_size < min_filesize_bytes:
                    continue
            except Exception:
                continue

            caption = (d.get("caption") or "").replace("\n", " ").strip()
            cleaned.append({
                "image": rel_path,
                "caption": caption,
                "transcripts": d.get("transcripts", []),
                "source_url": d.get("image_url", ""),
            })

        if not cleaned:
            raise RuntimeError("No usable records after filtering. Check paths/status.")

        # 3) Split 80/20 theo seed
        N = len(cleaned)
        idxs = list(range(N))
        rng = random.Random(seed)
        rng.shuffle(idxs)
        n_train = int(0.8 * N)
        train_set = set(idxs[:n_train])
        val_set = set(idxs[n_train:])

        if split in ("train", "qual_train"):
            self.target_split_name = "train"
            chosen = [cleaned[i] for i in sorted(train_set)]
        else:  # 'test' | 'qual_test'
            self.target_split_name = "qual_test"
            chosen = [cleaned[i] for i in sorted(val_set)]

        if max_items is not None:
            chosen = chosen[:max_items]

        self.samples = chosen
        self.image_root = self.root

        # 4) CLIP preprocess
        _, self.preprocess = clip.load(base_model)

        # 5) Sổ tay ảnh hỏng để tránh lặp (tuỳ chọn)
        self._bad_indices = set()

    def __len__(self):
        return len(self.samples)

    def _load_image_rgb(self, path: str) -> Image.Image:
        # cố gắng mở "an toàn"
        img = Image.open(path)
        img.load()  # nạp đầy đủ buffer (giúp ảnh truncated)
        return img.convert("RGB")

    def __getitem__(self, idx):
        # Thử tối đa max_skip_attempts nếu ảnh lỗi
        attempt = 0
        cur_idx = idx
        while attempt < self.max_skip_attempts:
            item = self.samples[cur_idx]
            caption = item["caption"]
            caption_short = caption.split(".")[0].strip() + "." if "." in caption else caption
            img_path = os.path.join(str(self.image_root), item["image"])
            try:
                image = self._load_image_rgb(img_path)
                image_tensor = self.preprocess(image)
                return image_tensor, caption, caption_short, img_path, self.target_split_name
            except Exception:
                self._bad_indices.add(cur_idx)
                if self.on_error == "raise":
                    raise
                # Chọn một index khác ngẫu nhiên (tránh index đang lỗi)
                cur_idx = random.randrange(0, len(self.samples))
                attempt += 1

        # Nếu quá nhiều lần vẫn lỗi
        raise RuntimeError(f"Too many broken images encountered near index {idx}.")
