# ===== DCI retrieval eval — dùng model & preprocess từ cell 1 =====
import os
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# LongCLIP tokenizer (same pattern as other cells)
try:
    from model_sail import longclip
except ImportError:
    from model import longclip  # fallback


# -----------------------------
# Dataset (pass in 'preprocess' from cell 1)
# -----------------------------
class JsonDCIDataset(Dataset):
    """
    Loads a flat JSON list: [{"filename": <path>, "caption": <str>}, ...]
    Returns tuple aligned with previous datasets:
      (image_tensor, caption_long, caption_short, img_path, split)
    """
    def __init__(self, json_path: str, preprocess, max_items: Optional[int] = None):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        if max_items is not None:
            samples = samples[:max_items]
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['filename']

        # Load + preprocess image
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        # Long + short captions
        caption = str(sample['caption']).replace("\n", " ").strip()
        cap0 = caption.split('.')[0].strip()
        caption_short = (cap0 + '.') if cap0 else ""

        return (
            image_tensor,     # [3, H, W]
            caption,          # long caption
            caption_short,    # short caption
            img_path,         # path
            "json"            # dummy split label
        )


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
# Helpers & sentence splitter (same as other cells)
# -----------------------------
def _encode_image(model, images):
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens):
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out

def split_into_detail_captions(text: str, max_details: int = 4):
    """
    Split into <= max_details short sentences by . ! ? ; : or newlines.
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
# Evaluation core (mirrors Urban1k/DOCCI)
# -----------------------------
@torch.no_grad()
def test_epoch_with_model_dci(model, dataloader, device, max_details: int = 4):
    model.eval()
    model = model.to(device)

    im_feats_list   = []
    long_feats_list = []
    txt_feats_lists = {j: [] for j in range(max_details)}

    for batch in tqdm(dataloader, desc="Extracting DCI features"):
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

        # Detail captions
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

    # Recall@K (1-1 mapping by row index)
    N = im_feats_all.size(0)
    target = torch.arange(N, device=im_feats_all.device)
    ks = [1, 5, 25, 50]

    acc = {}
    # Text→Image
    for name, sims in sims_t2i.items():
        for k in ks:
            topk_inds = sims.topk(k, dim=1).indices
            hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"{name}_t2i_R{k}"] = hits

    # Image→Text (full)
    for k in ks:
        topk_inds = sims_i2t_long.topk(k, dim=1).indices
        hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
        acc[f"long_i2t_R{k}"] = hits

    # Summary
    print("\n—— DCI Metrics ——")
    for k in ks:
        print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
    for k in ks:
        print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
    print("—" * 30)
    return acc


# -----------------------------
# Runner for notebook
# -----------------------------
def run_dci_eval(
    model,
    preprocess,
    json_path: str,           # path to the flat JSON file
    batch_size: int = 64,
    num_workers: int = 32,
    device: Optional[str] = None,
    max_items: Optional[int] = None,
    max_details: int = 4,
):
    """
    Uses 'model' & 'preprocess' from cell 1; mirrors other runners.
    """
    assert preprocess is not None, "`preprocess` must be provided."
    assert os.path.isfile(json_path), f"Not found: {json_path}"

    # Keep transforms on CPU for worker safety (if supported)
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = JsonDCIDataset(json_path=json_path, preprocess=preprocess, max_items=max_items)
    dl = build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    print(f"DCI JSON: {json_path}")
    print(f"Num samples: {len(ds)}")
    print(f"Num batches: {len(dl)}")

    metrics = test_epoch_with_model_dci(model, dl, device=device, max_details=max_details)
    print(f"Individual metrics: {metrics}")
    return metrics
