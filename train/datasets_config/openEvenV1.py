# openeventv1_dataset.py
import os
import csv
import random
from pathlib import Path
from typing import Optional, List

from PIL import Image
import torch.utils.data as data

# Optional: load lazily để không tốn thời gian nhiều lần
_CLIP_PREPROCESS = None

def _get_clip_preprocess(model_name: str = "ViT-B/16"):
# def _get_clip_preprocess(model_name: str = "ViT-L/14"):
    global _CLIP_PREPROCESS
    if _CLIP_PREPROCESS is None:
        import clip
        _, _CLIP_PREPROCESS = clip.load(model_name)
    return _CLIP_PREPROCESS


class OpenEventV1Dataset(data.Dataset):
    """
    Trả về tuple cùng format với DocciDataset:
      (image_tensor, caption, caption_short, img_path, split)

    - root_dir: thư mục chứa gt_train.csv và thư mục ảnh (vd: '.../OpenEvenv1/train/Train Set')
    - csv_name: tên file csv (mặc định 'gt_train.csv')
    - images_dir: tên thư mục ảnh bên trong root_dir (mặc định 'train_images_compressed90')
    - split: 'train' hoặc 'test' (chia 80/20 theo train_ratio)
    - train_ratio: tỉ lệ train (mặc định 0.8)
    - seed: để chia deterministic
    - max_items: giới hạn số mẫu ở split đã chọn
    - clip_model / preprocess: chọn model CLIP hoặc truyền sẵn preprocess
    """
    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 csv_name: str = "gt_train.csv",
                 images_dir: str = "train_images_compressed90",
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 max_items: Optional[int] = None,
                 model_name: str = "ViT-B/16",
                #  clip_model: str = "ViT-L/14",
                 preprocess=None):
        assert split in ("train", "test"), "split phải là 'train' hoặc 'test'."
        assert 0.0 < train_ratio < 1.0, "train_ratio phải nằm trong (0,1)."

        self.root_dir = Path(root_dir)
        self.csv_path = self.root_dir / csv_name
        self.images_dir = self.root_dir / images_dir
        self.split = split

        if preprocess is None:
            self.preprocess = _get_clip_preprocess(model_name)
        else:
            self.preprocess = preprocess

        # Đọc toàn bộ CSV (có thể có caption nhiều dòng => mở với newline='')
        rows = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Chuẩn hóa key theo header: image_index, caption, retrieved_article_id, retrieved_image_id
                rows.append({
                    "image_index": r["image_index"].strip(),
                    "caption": r["caption"],
                    "retrieved_article_id": r.get("retrieved_article_id", "").strip(),
                    "retrieved_image_id": r.get("retrieved_image_id", "").strip(),
                })

        # Sắp xếp cố định rồi shuffle bằng seed để chia 80/20 ổn định
        rows.sort(key=lambda x: x["image_index"])
        rng = random.Random(seed)
        rng.shuffle(rows)

        n_total = len(rows)
        n_train = int(round(train_ratio * n_total))
        train_rows = rows[:n_train]
        test_rows  = rows[n_train:]

        self.samples = train_rows if split == "train" else test_rows

        if max_items is not None:
            self.samples = self.samples[:max_items]

        # Kiểm tra thư mục ảnh tồn tại
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Không thấy thư mục ảnh: {self.images_dir}")

    def __len__(self):
        return len(self.samples)

    def _find_image_path(self, image_index: str) -> Path:
        # Mặc định ảnh là .jpg; thử thêm các đuôi khác nếu cần
        candidates: List[Path] = [
            self.images_dir / f"{image_index}.jpg",
            self.images_dir / f"{image_index}.jpeg",
            self.images_dir / f"{image_index}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        # Nếu không tìm được, ném lỗi rõ ràng
        raise FileNotFoundError(
            f"Không tìm thấy file ảnh cho image_index={image_index} "
            f"trong {self.images_dir} (đã thử .jpg/.jpeg/.png)"
        )

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image_index = item["image_index"]

        # Caption: giữ nguyên, chỉ thay \n -> space để ổn định như Docci
        caption_full = (item["caption"] or "").replace("\r\n", "\n").replace("\n", " ").strip()
        # Câu ngắn: lấy câu đầu (đơn giản theo dấu '.')
        if "." in caption_full:
            caption_short = caption_full.split(".")[0].strip() + "."
        else:
            caption_short = caption_full

        img_path = self._find_image_path(image_index)
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        # Trả về như Docci: (image_tensor, caption, caption_short, img_path, split)
        return image_tensor, caption_full, caption_short, str(img_path), self.split
