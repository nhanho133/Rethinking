# eval_openeventv1.py
# -*- coding: utf-8 -*-
"""
OpenEventV1 retrieval eval — dùng model & preprocess đã load sẵn (giống phong cách DOCCI eval)
- Dataset: sử dụng OpenEventV1Dataset mà bạn đã viết trong openeventv1_dataset.py
- Tokenizer: ưu tiên model_sail.longclip / model.longclip; fallback sang model.finelip nếu không có longclip
- Metrics: R@{1,5,25,50} cho Text→Image (full + detail_1..max_details) và Image→Text (full)

Ví dụ dùng trong notebook:
    from model_sail import longclip  # hoặc từ chỗ bạn load model
    model, preprocess = longclip.load("ViT-B/16")
    from eval_openeventv1 import run_openevent_eval
    run_openevent_eval(
        model, preprocess,
        root_dir="/home/ubuntu/shared/OpenEvenv1/train/Train Set",
        csv_name="gt_train.csv",
        images_dir="train_images_compressed90",
        split="test",
        batch_size=64, num_workers=32, device="cuda:0",
        max_items=None, max_details=4
    )
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

# =========================
# Tokenizer resolver
# =========================
def _resolve_tokenize():
    """
    Trả về hàm tokenize theo thứ tự ưu tiên:
      1) model_sail.longclip.tokenize
      2) model.longclip.tokenize
      3) model.finelip.tokenize
    """
    try:
        from model_sail import longclip as _lc
        return _lc.tokenize, "longclip(model_sail)"
    except Exception:
        pass
    try:
        from model import longclip as _lc
        return _lc.tokenize, "longclip(model)"
    except Exception:
        pass
    try:
        from model import finelip as _fl
        return _fl.tokenize, "finelip(model)"
    except Exception:
        pass
    raise ImportError(
        "Không tìm thấy tokenizer. Hãy đảm bảo có `model_sail.longclip` "
        "hoặc `model.longclip` hoặc `model.finelip`, "
        "hoặc tự sửa `run_openevent_eval(..., tokenize_fn=...)` để truyền hàm tokenize."
    )

# =========================
# Import dataset lớp bạn đã viết
# =========================
import sys
sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/datasets_config")
sys.path.append("../..")
from openEvenV1 import OpenEventV1Dataset
# try:
#     import sys
#     sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/datasets")
#     from openEvenV1 import OpenEventV1Dataset
# except ImportError as e:
#     raise ImportError(
#         "Không import được OpenEventV1Dataset. Đảm bảo file 'openeventv1_dataset.py' nằm trong PYTHONPATH hiện tại."
#     ) from e


# =========================
# Dataloader builder
# =========================
def build_loader(dataset: Dataset, batch_size: int, num_workers: int = 8) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# =========================
# Helpers (giống DOCCI eval)
# =========================
def _encode_image(model, images: torch.Tensor) -> torch.Tensor:
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens: torch.Tensor) -> torch.Tensor:
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out


def split_into_detail_captions(text: str, max_details: int = 4) -> List[str]:
    """
    Cắt long caption thành <= max_details câu ngắn theo . ! ? ; : hoặc xuống dòng.
    Đệm "" nếu thiếu.
    """
    import re
    if not isinstance(text, str):
        text = str(text)
    parts = [p.strip() for p in re.split(r'[\.!\?;:\n]+', text) if p.strip()]
    parts = parts[:max_details]
    while len(parts) < max_details:
        parts.append("")
    return parts


# =========================
# Evaluation core
# =========================
@torch.no_grad()
def test_epoch_with_model_openevent(
    model,
    dataloader: DataLoader,
    device: str,
    tokenize_fn,
    max_details: int = 4,
) -> Dict[str, float]:
    model.eval()
    model = model.to(device)

    im_feats_list: List[torch.Tensor] = []
    long_feats_list: List[torch.Tensor] = []
    txt_feats_lists: Dict[int, List[torch.Tensor]] = {j: [] for j in range(max_details)}

    for batch in tqdm(dataloader, desc="Extracting OpenEventV1 features"):
        # batch format từ dataset: (image_tensor, caption_full, caption_short, img_path, split)
        images, long_texts = batch[0], batch[1]
        images = images.to(device, non_blocking=True)

        # Image features
        feats_i = _encode_image(model, images)
        feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
        im_feats_list.append(feats_i)

        # Full-text (long) features
        tokens_long = tokenize_fn(list(long_texts), truncate=True).to(device)
        feats_long = _encode_text(model, tokens_long)
        feats_long = feats_long / feats_long.norm(dim=-1, keepdim=True)
        long_feats_list.append(feats_long)

        # Detail captions
        caps_batch = [split_into_detail_captions(t, max_details=max_details) for t in long_texts]
        for j in range(max_details):
            texts_j = [c[j] if j < len(c) else "" for c in caps_batch]
            toks_j = tokenize_fn(texts_j, truncate=True).to(device)
            fts_j = _encode_text(model, toks_j)
            fts_j = fts_j / fts_j.norm(dim=-1, keepdim=True)
            txt_feats_lists[j].append(fts_j)

    # Concatenate all
    im_feats_all  = torch.cat(im_feats_list, dim=0)
    long_all      = torch.cat(long_feats_list, dim=0)
    txt_all_lists = [torch.cat(txt_feats_lists[j], dim=0) for j in range(max_details)]

    # Similarities
    sims_t2i = {
        "long": long_all @ im_feats_all.T,
        **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(max_details)},
    }
    sims_i2t_long = im_feats_all @ long_all.T

    # Recall@K (1-1 mapping)
    N = im_feats_all.size(0)
    target = torch.arange(N, device=im_feats_all.device)
    ks = [1, 5, 25, 50]

    acc: Dict[str, float] = {}
    # Text→Image (full + detail)
    for name, sims in sims_t2i.items():
        for k in ks:
            topk_inds = sims.topk(k, dim=1).indices  # [N, k]
            hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"{name}_t2i_R{k}"] = hits

    # Image→Text (full only)
    for k in ks:
        topk_inds = sims_i2t_long.topk(k, dim=1).indices
        hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
        acc[f"long_i2t_R{k}"] = hits

    # Print summary giống phong cách DOCCI/Urban1k
    print("\n—— OpenEventV1 Metrics ——")
    for k in ks:
        print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
    for j in range(max_details):
        key = f"detail_{j+1}"
        for k in ks:
            print(f"   • {key:9} → Image @ {k:2}: {acc[f'{key}_t2i_R{k}']:.4%}")
    for k in ks:
        print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
    print("—" * 30)
    return acc


# =========================
# Runner (giống DOCCI style)
# =========================
def run_openevent_eval(
    model,
    preprocess,
    *,
    root_dir: str,
    split: str = "test",               # 'train' hoặc 'test' (dataset của bạn đã chia 80/20 theo seed)
    csv_name: str = "gt_train.csv",
    images_dir: str = "train_images_compressed90",
    batch_size: int = 64,
    num_workers: int = 32,
    device: Optional[str] = None,
    max_items: Optional[int] = None,
    max_details: int = 4,
    tokenize_fn=None,                  # tùy chọn: truyền hàm tokenize riêng nếu không dùng longclip/finelip mặc định
):
    """
    Dùng 'model' & 'preprocess' đang có, build dataset OpenEventV1Dataset và tính metrics retrieval.
    """
    assert preprocess is not None, "`preprocess` must be provided."

    # Mặc định device
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve tokenizer nếu chưa truyền
    if tokenize_fn is None:
        tokenize_fn, tok_src = _resolve_tokenize()
        print(f"[Info] Using tokenizer from {tok_src}")

    # Dataset
    ds = OpenEventV1Dataset(
        root_dir=root_dir,
        split=split,
        train_ratio=0.8,
        csv_name=csv_name,
        images_dir=images_dir,
        preprocess=preprocess,
        max_items=max_items,
    )
    dl = build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    print(f"OpenEventV1 split: {split}")
    print(f"Num samples: {len(ds)}")
    print(f"Num batches: {len(dl)}")

    metrics = test_epoch_with_model_openevent(
        model, dl, device=device, tokenize_fn=tokenize_fn, max_details=max_details
    )
    print(f"Individual metrics: {metrics}")
    return metrics


# =========================
# (Optional) CLI đơn giản
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenEventV1 retrieval eval (giống DOCCI style)")
    parser.add_argument("--root_dir", required=True, type=str,
                        help="Thư mục chứa gt_train.csv và thư mục ảnh (vd: '.../OpenEvenv1/train/Train Set')")
    parser.add_argument("--csv_name", default="gt_train.csv", type=str)
    parser.add_argument("--images_dir", default="train_images_compressed90", type=str)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--max_items", default=None, type=int)
    parser.add_argument("--max_details", default=4, type=int)
    # parser.add_argument("--model_name", default="ViT-B/16", type=str,
    #                     help="Tên backbone nếu bạn tự load trong main (tuỳ môi trường).")
    parser.add_argument("--model_name", default="ViT-L/16", type=str,
                        help="Tên backbone nếu bạn tự load trong main (tuỳ môi trường).")

    args = parser.parse_args()

    # Lưu ý: phần CLI này chỉ minh hoạ. Thực tế bạn nên load model & preprocess từ code của bạn.
    # Dưới đây là load LongCLIP nếu bạn có sẵn:
    try:
        from model_sail import longclip
    except Exception:
        from model import longclip  # fallback

    model, preprocess = longclip.load("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/openv1_propose_train/ckpt/B16-longclip-10-31--05_40_24_-8.pt")
    run_openevent_eval(
        model, preprocess,
        root_dir=args.root_dir,
        csv_name=args.csv_name,
        images_dir=args.images_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_items=args.max_items,
        max_details=args.max_details,
    )
