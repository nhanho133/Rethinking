"""
Zero-shot cross-dataset retrieval eval, matching the protocol DreamLIP/LLM2CLIP actually use to
report numbers in their papers: train on dataset A, evaluate retrieval on a DIFFERENT dataset
(Flickr30K / COCO) with NO fine-tuning on it. This is different from the in-domain
train-test-split eval used elsewhere in this project (test_epoch_ver5) -- that answers "does
this loss help fine-tuning work on data like what it saw," this answers "do the learned
features transfer to unseen data," which is what the papers' own tables report.

Simplification: uses ONE caption per image (Flickr30K/COCO both have 5 captions/image; using
just the first keeps this consistent with how the rest of this project's in-domain eval treats
a single long caption per image). This is NOT the exact 5-caption-per-image COCO protocol used
in published papers, so these numbers are indicative, not directly the same benchmark number.

Two-phase: (1) if no cache exists for the target dataset's captions, precompute Llama-3-8B-CC
embeddings for them (same frozen text teacher as everywhere else); (2) load a trained
llm2clip_text checkpoint and compute standard zero-shot T2I/I2T R@1/5/10.
"""
import argparse
import csv
import json
import os
import sys

import torch
from PIL import Image

sys.path.append(".")
sys.path.append("..")
from model.model_llm2clip import LLM2CLIPTextTeacher, TextEmbeddingCache
from precompute_llm2vec_embeddings import load_l2v
import clip


def load_flickr30k(csv_path, image_root, max_items=None, split="test"):
    """split: Karpathy-split column in flickr_annotations_30k.csv ('train'/'val'/'test',
    sizes 29000/1014/1000). Paper's "Flickr 1K test set" = split=="test" (1000 images) --
    NOT filtering here (the previous bug) silently evaluated on all 31k images instead."""
    items = []
    with open(csv_path, encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            caps = eval(row["raw"])  # stored as a python-list literal string
            items.append({
                "image": os.path.join(image_root, row["filename"]),
                "caption": caps[0].strip(),
            })
    if max_items:
        items = items[:max_items]
    return items


def load_coco_val(ann_path, image_root, max_items=None):
    data = json.load(open(ann_path, encoding="utf8"))
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    seen = set()
    items = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        items.append({
            "image": os.path.join(image_root, id_to_file[img_id]),
            "caption": ann["caption"].strip(),
        })
    if max_items:
        items = items[:max_items]
    return items


def ensure_cache(items, cache_path, batch_size=32):
    if os.path.exists(cache_path):
        print(f"[cache] found existing {cache_path}")
        return
    import hashlib
    captions = sorted({it["caption"] for it in items}, key=len)
    print(f"[cache] precomputing {len(captions)} caption embeddings -> {cache_path}")
    l2v = load_l2v(quant_4bit=True)
    cache = {}
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        embs = l2v.encode(batch, batch_size=len(batch), show_progress_bar=False, convert_to_tensor=True)
        embs = embs.to(torch.float16).cpu()
        for text, emb in zip(batch, embs):
            if not torch.isfinite(emb).all():
                emb = torch.zeros_like(emb)
            cache[hashlib.sha1(text.encode("utf8")).hexdigest()] = emb
        if i % (batch_size * 20) == 0:
            print(f"  [embed] {i}/{len(captions)}", flush=True)
    torch.save({"cache": cache, "dim": 4096}, cache_path)
    print(f"[cache] wrote {len(cache)} embeddings")
    del l2v
    torch.cuda.empty_cache()


def _remap_legacy_adapter_keys(sd):
    if "text_adapter.ln.weight" not in sd:
        return sd
    remap = {
        "text_adapter.ln.weight": "text_adapter.text_adaptor.0.weight",
        "text_adapter.ln.bias": "text_adapter.text_adaptor.0.bias",
        "text_adapter.proj.weight": "text_adapter.text_adaptor.1.weight",
        "text_adapter.proj.bias": "text_adapter.text_adaptor.1.bias",
    }
    return {remap.get(k, k): v for k, v in sd.items()}


@torch.no_grad()
def run_eval(ckpt_path, items, cache_path, device, batch_size=64, released=False):
    if released:
        # Microsoft's real released LLM2CLIP-Openai-L-14-336 (ViT-L-336 + their trained MLP
        # adapter), as a reference upper bound. Same cached-Llama-embedding text interface.
        from model.model_llm2clip import LLM2CLIPReleasedTeacher
        model = LLM2CLIPReleasedTeacher(
            model_path="/cm/shared/chautvh_second/Nhan_folder/ckpts/ViT-L-336", device="cpu")
        model = model.to(device).eval()
        _, preprocess = clip.load("ViT-L/14@336px", device="cpu")
    else:
        model = LLM2CLIPTextTeacher(clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                                    freeze_visual=False, device="cpu", adapter_type="linear")
        sd = _remap_legacy_adapter_keys(torch.load(ckpt_path, map_location="cpu"))
        model.load_state_dict(sd)
        model = model.to(device).eval()
        _, preprocess = clip.load("ViT-B/16", device="cpu")
    cache = TextEmbeddingCache(cache_path, device=device)

    im_feats, txt_feats = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(it["image"]).convert("RGB")) for it in batch]).to(device)
        f_i = model.encode_image(imgs); f_i = f_i / f_i.norm(dim=-1, keepdim=True)
        im_feats.append(f_i)
        t_emb = cache.lookup([it["caption"] for it in batch])
        f_t = model.encode_text(t_emb); f_t = f_t / f_t.norm(dim=-1, keepdim=True)
        txt_feats.append(f_t)
        if i % (batch_size * 10) == 0:
            print(f"  [encode] {i}/{len(items)}", flush=True)
    im_feats = torch.cat(im_feats, 0)
    txt_feats = torch.cat(txt_feats, 0)

    N = im_feats.size(0)
    target = torch.arange(N, device=device)
    sims_t2i = txt_feats @ im_feats.t()
    sims_i2t = im_feats @ txt_feats.t()
    ks = [1, 5, 10]
    acc = {}
    for name, sims in [("T2I", sims_t2i), ("I2T", sims_i2t)]:
        for k in ks:
            topk = sims.topk(k, dim=1).indices
            hits = topk.eq(target.unsqueeze(1)).any(dim=1).float().mean().item()
            acc[f"{name}_R{k}"] = hits
    return acc, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["flickr30k", "coco"], required=True)
    ap.add_argument("--ckpt", default="RELEASED")
    ap.add_argument("--released", action="store_true", help="Eval Microsoft's real LLM2CLIP-L-336 (reference)")
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "flickr30k":
        items = load_flickr30k(
            "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/flickr30k/flickr_annotations_30k.csv",
            "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/flickr30k/images/flickr30k-images",
            max_items=args.max_items,
        )
        cache_path = "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/flickr30k/llm2vec_cc_cache.pt"
    else:
        items = load_coco_val(
            "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/coco/annotations/captions_val2017.json",
            "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/coco/val2017",
            max_items=args.max_items,
        )
        cache_path = "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/coco/llm2vec_cc_cache.pt"

    print(f"[data] {len(items)} images from {args.dataset}")
    ensure_cache(items, cache_path)
    acc, n = run_eval(args.ckpt, items, cache_path, device, released=args.released)
    print(f"[RESULT] {args.dataset} zero-shot, ckpt={args.ckpt}, n={n}")
    for k, v in acc.items():
        print(f"[RESULT]   {k}: {v:.4%}")


if __name__ == "__main__":
    main()
