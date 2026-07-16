# ===== Urban1k retrieval eval — dùng model & preprocess từ cell 1 =====
import os
from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# LongCLIP tokenizer
import sys
sys.path.append("../..")
try:
    from model_sail import longclip
except ImportError:
    from model import longclip  # fallback


# -----------------------------
# Dataset: dùng preprocess bạn truyền vào
# -----------------------------
class Urban1kDataset(Dataset):
    """
    Urban1k: mỗi ảnh có 1 caption trùng tên file (không đuôi).
    Trả về: (image_tensor, caption, "None", "None", "None") để tương thích collate cũ.
    """
    def __init__(self, root_dir: str, preprocess, max_items: Optional[int] = None):
        self.image_dir = os.path.join(root_dir, "image")
        self.caption_dir = os.path.join(root_dir, "caption")
        assert os.path.isdir(self.image_dir), f"Not found: {self.image_dir}"
        assert os.path.isdir(self.caption_dir), f"Not found: {self.caption_dir}"

        self.ids = sorted(
            fname[:-4]
            for fname in os.listdir(self.caption_dir)
            if fname.endswith(".txt")
        )
        if max_items is not None:
            self.ids = self.ids[:max_items]

        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        item_id = self.ids[index]

        # Caption
        with open(os.path.join(self.caption_dir, f"{item_id}.txt"), "r", encoding="utf-8") as f:
            caption = f.read().strip().replace("\n", " ")

        # Image (.jpg theo cấu trúc Urban1k)
        img_path = os.path.join(self.image_dir, f"{item_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image_t = self.preprocess(image)

        return image_t, caption, "None", "None", "None"


def build_loader(dataset: Dataset, batch_size: int, num_workers: int = 8) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# -----------------------------
# Helpers & splitter
# -----------------------------
def _encode_image(model, images):
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens):
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out

def split_into_detail_captions(text: str, max_details: int = 4):
    """
    Tách caption thành tối đa `max_details` câu ngắn theo dấu . ! ? ; : hoặc xuống dòng.
    Pad "" nếu thiếu.
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
# Evaluation core (như test_epoch nhưng dùng trực tiếp `model`)
# -----------------------------
@torch.no_grad()
def test_epoch_with_model_urban1k(model, dataloader, device, max_details: int = 4):
    model.eval()
    model = model.to(device)

    im_feats_list     = []
    long_feats_list   = []
    txt_feats_lists   = {j: [] for j in range(max_details)}

    for batch in tqdm(dataloader, desc="Extracting features"):
        # Dataloader trả: (images, long_texts, _, _, _)
        images, long_texts = batch[0], batch[1]
        images = images.to(device, non_blocking=True)

        # Image features
        feats_i = _encode_image(model, images)
        feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
        im_feats_list.append(feats_i)

        # Full-text features
        tokens_long = longclip.tokenize(list(long_texts), truncate=True).to(device)
        feats_long = _encode_text(model, tokens_long)
        feats_long = feats_long / feats_long.norm(dim=-1, keepdim=True)
        long_feats_list.append(feats_long)

        # Detail captions (per batch)
        caps_batch = [split_into_detail_captions(t, max_details=max_details) for t in long_texts]
        for j in range(max_details):
            texts_j = [c[j] if j < len(c) else "" for c in caps_batch]
            toks_j = longclip.tokenize(texts_j, truncate=True).to(device)
            fts_j = _encode_text(model, toks_j)
            fts_j = fts_j / fts_j.norm(dim=-1, keepdim=True)
            txt_feats_lists[j].append(fts_j)

    # Concat
    im_feats_all   = torch.cat(im_feats_list, dim=0)
    long_all       = torch.cat(long_feats_list, dim=0)
    txt_all_lists  = [torch.cat(txt_feats_lists[j], dim=0) for j in range(max_details)]

    # Similarities
    sims_t2i = {
        "long":    long_all @ im_feats_all.T,
        **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(max_details)}
    }
    sims_i2t_long = im_feats_all @ long_all.T

    # Recall@K (1-1 mapping ảnh↔caption theo thứ tự)
    N = im_feats_all.size(0)
    target = torch.arange(N, device=im_feats_all.device)
    ks = [1, 5, 25, 50]

    acc = {}
    # Text→Image (full + detail)
    for name, sims in sims_t2i.items():
        for k in ks:
            topk_inds = sims.topk(k, dim=1).indices              # [N, k]
            hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"{name}_t2i_R{k}"] = hits

    # Image→Text (full)
    for k in ks:
        topk_inds = sims_i2t_long.topk(k, dim=1).indices
        hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
        acc[f"long_i2t_R{k}"] = hits

    # Print summary
    print("\n—— Urban1k Metrics ——")
    for k in ks:
        print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
    for k in ks:
        print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
    print("—" * 30)
    return acc


# -----------------------------
# Runner cho notebook
# -----------------------------
def run_urban1k_eval(
    model,
    preprocess,
    data_root="/home/ubuntu/shared/hieu.tq/Urban1k/Urban1k",
    batch_size=64,
    num_workers=32,
    device=None,
    max_items=None,
    max_details=4,
):
    """
    Dùng `model` & `preprocess` từ cell 1. Không dùng CLIP_Clean_Train.
    """
    assert preprocess is not None, "`preprocess` must be provided."

    # ✅ Quan trọng: reset default device về CPU để DataLoader/transform không đòi CUDA generator
    try:
        # Chỉ có từ PyTorch 2.0+, nếu không có thì bỏ qua
        import torch
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = Urban1kDataset(root_dir=data_root, preprocess=preprocess, max_items=max_items)
    dl = build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    print(f"Số mẫu Urban1k: {len(ds)}")
    print(f"Số batch: {len(dl)}")

    metrics = test_epoch_with_model_urban1k(model, dl, device=device, max_details=max_details)
    print(f"Individual metrics: {metrics}")
    return metrics
