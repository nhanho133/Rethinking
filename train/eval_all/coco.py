# ===== COCO 2017 retrieval eval (multi-caption-per-image, cached image feats) =====
import os, json
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Nếu longclip nằm ở module khác, bạn có thể đổi import cho phù hợp
import sys
sys.path.append("../..")
try:
    from model import longclip
except ImportError:
    from model import longclip  # fallback

# ────────────────────────────────────────────────────────────────────────────────
# Dataset: dùng preprocess do bạn truyền từ cell 1 (hoặc lấy từ model.preprocess)
# ────────────────────────────────────────────────────────────────────────────────
class COCODataset(Dataset):
    """
    Trả về: image_tensor, caption (str), image_id (str)
    """
    def __init__(self,
                 data_root: str,
                 split: str = "val",
                 preprocess=None,
                 max_items: Optional[int] = None):
        assert split in ("train", "val")
        assert preprocess is not None, "`preprocess` is required (pass từ cell 1 hoặc model.preprocess)."

        ann_file = Path(data_root) / "annotations" / f"captions_{split}2017.json"
        img_dir  = Path(data_root) / f"{split}2017"

        with open(ann_file, "r", encoding="utf8") as fp:
            coco = json.load(fp)

        id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        self.samples = [
            {
                "image_id": str(ann["image_id"]),
                "file_name": id2file[ann["image_id"]],
                "caption": ann["caption"].replace("\n", " ")
            }
            for ann in coco["annotations"]
        ]
        if max_items is not None:
            self.samples = self.samples[: max_items]

        self.img_root   = img_dir
        self.preprocess = preprocess

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.img_root / s["file_name"]
        image = Image.open(img_path).convert("RGB")
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
def evaluate_coco(model, dataloader, device="cuda"):
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
            feats_i = _cpu_fp32(feats_i)                       # <<< FIX: ép về CPU FP32
            for _id, _f in zip(new_ids, feats_i):
                img_feat_cache[_id] = _f

        # 2) Encode text
        tokens = longclip.tokenize(list(caps), truncate=True).to(device)
        feats_t = F.normalize(_encode_text(model, tokens), dim=-1)
        feats_t = _cpu_fp32(feats_t)                           # <<< FIX: ép về CPU FP32
        txt_feats_parts.append(feats_t)
        cap2img.extend(img_ids)

    # Gộp tensor (đảm bảo FP32)
    img_ids_list = list(img_feat_cache.keys())
    img_feats = torch.stack([img_feat_cache[k] for k in img_ids_list]).to(torch.float32)  # [I, D]
    txt_feats = torch.cat(txt_feats_parts, dim=0).to(torch.float32)                       # [M, D]

    # Similarity (CPU FP32)
    sims_t2i = txt_feats @ img_feats.T   # [M, I]
    sims_i2t = img_feats @ txt_feats.T   # [I, M]

    # Recall@K
    ks = (1, 5, 10)
    recalls_t2i, recalls_i2t = [], []

    # text → image: ảnh đúng là ảnh có image_id == cap2img[i]
    for k in ks:
        topk = sims_t2i.topk(k, dim=1).indices  # [M, k] (chỉ số trong img_ids_list)
        correct = 0
        for i, row in enumerate(topk):
            if any(img_ids_list[j] == cap2img[i] for j in row):
                correct += 1
        recalls_t2i.append(correct / len(cap2img))

    # image → text: đúng nếu top-k có ÍT NHẤT MỘT caption thuộc ảnh đó
    imgid2capidx: Dict[str, List[int]] = {}
    for idx, iid in enumerate(cap2img):
        imgid2capidx.setdefault(iid, []).append(idx)

    for k in ks:
        topk = sims_i2t.topk(k, dim=1).indices  # [I, k] (chỉ số caption)
        correct = 0
        for i, row in enumerate(topk):
            iid = img_ids_list[i]
            if any(idx in imgid2capidx[iid] for idx in row.tolist()):
                correct += 1
        recalls_i2t.append(correct / len(img_ids_list))

    return {"T→I": dict(zip(ks, recalls_t2i)),
            "I→T": dict(zip(ks, recalls_i2t))}

# ────────────────────────────────────────────────────────────────────────────────
# Runner cho notebook
# ────────────────────────────────────────────────────────────────────────────────
def run_coco_eval(
    model,
    preprocess=None,
    data_root="/home/ubuntu/shared/ShareGPT4V/data/coco/images/coco2017",
    split="val",
    batch_size=64,
    num_workers=8,
    device=None,
    max_items=None,
):
    """
    Dùng model (và preprocess) đã load ở cell 1.
    Nếu `preprocess` = None, sẽ thử lấy `model.preprocess`.
    """
    if preprocess is None:
        preprocess = getattr(model, "preprocess", None)
    assert preprocess is not None, "Please provide `preprocess` or attach `model.preprocess`."

    # ✅ Quan trọng: reset default device về CPU để DataLoader/transform không đòi CUDA generator
    try:
        # Chỉ có từ PyTorch 2.0+, nếu không có thì bỏ qua
        import torch
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = COCODataset(
        data_root=data_root,
        split=split,
        preprocess=preprocess,
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

    # Số captions & ảnh duy nhất
    n_caps = len(ds)
    n_imgs = len({s["image_id"] for s in ds.samples})
    print(f"Dataset: {n_caps:,} captions – {n_imgs:,} images")

    metrics = evaluate_coco(model, dl, device=device)

    print("\nRecall@k Coco")
    for mode, d in metrics.items():
        print(f"  {mode:3s} : " + "  ".join(f"R@{k} = {v*100:.2f}%" for k, v in d.items()))
    return metrics
