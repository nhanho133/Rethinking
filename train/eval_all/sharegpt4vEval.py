# ===== ShareGPT4V retrieval eval =====
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train")  # thư mục chứa sharegpt4v.py
sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/datasets")
sys.path.append("../..")

from sharegpt4v import share4v_val_dataset

# Import dataset đã có sẵn
from sharegpt4v import share4v_val_dataset  

# LongCLIP tokenizer
from model import longclip  # fallback


# -----------------------------
# Helpers
# -----------------------------
def _encode_image(model, images):
    out = model.encode_image(images)
    return out[0] if isinstance(out, (tuple, list)) else out

def _encode_text(model, tokens):
    out = model.encode_text(tokens)
    return out[0] if isinstance(out, (tuple, list)) else out

def split_into_detail_captions(text: str, max_details: int = 4):
    if not isinstance(text, str):
        text = str(text)
    parts = [p.strip() for p in re.split(r"[\.!\?;:\n]+", text) if p.strip()]
    parts = parts[:max_details]
    while len(parts) < max_details:
        parts.append("")
    return parts


# -----------------------------
# Evaluation core
# -----------------------------
@torch.no_grad()
def test_epoch_with_model_share4v(model, dataloader, device, max_details: int = 4):
    model.eval()
    model = model.to(device)

    im_feats_list   = []
    long_feats_list = []
    txt_feats_lists = {j: [] for j in range(max_details)}

    for images, long_texts, _, image_paths, _ in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)

        # Image features
        feats_i = _encode_image(model, images)
        feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
        im_feats_list.append(feats_i)

        # Full caption features
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

    # Concat
    im_feats_all   = torch.cat(im_feats_list, dim=0)
    long_all       = torch.cat(long_feats_list, dim=0)
    txt_all_lists  = [torch.cat(txt_feats_lists[j], dim=0) for j in range(max_details)]

    # Similarities
    sims_t2i = {
        "long": long_all @ im_feats_all.T,
        **{f"detail_{j+1}": txt_all_lists[j] @ im_feats_all.T for j in range(max_details)},
    }
    sims_i2t_long = im_feats_all @ long_all.T

    # Recall@K
    N = im_feats_all.size(0)
    target = torch.arange(N, device=im_feats_all.device)
    ks = [1, 5, 25, 50]

    acc = {}
    for name, sims in sims_t2i.items():
        for k in ks:
            topk_inds = sims.topk(k, dim=1).indices
            hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"{name}_t2i_R{k}"] = hits

    for k in ks:
        topk_inds = sims_i2t_long.topk(k, dim=1).indices
        hits = topk_inds.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
        acc[f"long_i2t_R{k}"] = hits

    print("\n—— ShareGPT4V Metrics ——")
    for k in ks:
        print(f"▶ Full Text → Image @ {k:2}: {acc[f'long_t2i_R{k}']:.4%}")
    for k in ks:
        print(f"▶ Image → Full Text @ {k:2}: {acc[f'long_i2t_R{k}']:.4%}")
    print("—" * 30)
    return acc


# -----------------------------
# Runner
# -----------------------------
def run_share4v_eval(
    model,
    batch_size=64,
    num_workers=32,
    device=None,
    max_details=4,
):
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = share4v_val_dataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=(num_workers > 0))

    print(f"Số mẫu ShareGPT4V val: {len(ds)}")
    print(f"Số batch: {len(dl)}")

    metrics = test_epoch_with_model_share4v(model, dl, device=device, max_details=max_details)
    print(f"Individual metrics: {metrics}")
    return metrics
