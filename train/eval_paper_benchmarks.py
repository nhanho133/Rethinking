"""
Zero-shot paper-protocol evaluation, matching the benchmark suite LLM2CLIP (arXiv:2411.04997)
Table 2 reports: Flickr30K (1K test, Karpathy split, ALL captions/image), COCO val2017 (5K,
ALL captions/image), Urban1K (1K, 1 caption/image), SG4V-1K (1K, 1 caption/image -- proxy list,
see load_sg4v1k_items docstring), DOCCI (test+qual_test split, 1 caption/image).

Loads a TRAINED CHECKPOINT (or the untouched Microsoft-released one via --released) and computes
R@1/5/10 T2I+I2T with NO training -- this script never touches train.py's training loop. Text is
always encoded via the model's real interface: TextEmbeddingCache.lookup() -> precomputed
Llama-3-8B-CC embedding -> model.encode_text(). Never raw-tokenized.

Accounting is multi-caption-safe throughout (correctly degenerates to the simple 1-1 case for
Urban1K/SG4V-1K/DOCCI, which happen to have exactly 1 caption/image):
  T2I: a caption "hits" if its OWN ground-truth image is anywhere in the top-k image results.
  I2T: an image "hits" if ANY of its ground-truth captions is anywhere in the top-k text results.
Each unique image is encoded exactly once (cached by image_id) even though Flickr/COCO have
multiple caption rows per image.

Usage:
  python eval_paper_benchmarks.py --benchmark all --ckpt <path> --out_json results/ours.json
  python eval_paper_benchmarks.py --benchmark flickr30k --released --out_json results/released.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip
from model.model_llm2clip import LLM2CLIPTextTeacher, LLM2CLIPReleasedTeacher, TextEmbeddingCache

# Defaults -- override via CLI for local/server path differences.
EVAL_DATA_ROOT = "/cm/shared/chautvh_second/Nhan_folder/work/eval_data"
DOCCI_JSON = "/cm/archive/luongtk/docci/captioner_docci.json"
DOCCI_IMAGE_ROOT = "/cm/archive/luongtk/docci"
SHAREGPT4V_FULL_JSON = "/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
SHAREGPT4V_IMAGE_ROOT = "/cm/archive/luongtk/sharegpt4v/data"
LLM2CLIP_RELEASED_CKPT = "/cm/shared/chautvh_second/Nhan_folder/ckpts/ViT-L-336"


# ---- Per-benchmark item loaders: all return list[{"image_id","image_path","caption"}] -------

def load_flickr30k_items(eval_data_root, max_items=None):
    import csv
    csv_path = os.path.join(eval_data_root, "flickr30k", "flickr_annotations_30k.csv")
    image_root = os.path.join(eval_data_root, "flickr30k", "images", "flickr30k-images")
    if not os.path.isdir(image_root):
        image_root = os.path.join(eval_data_root, "flickr30k", "flickr30k-images")
    items = []
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            img_id = row.get("img_id") or row["filename"]
            img_path = os.path.join(image_root, row["filename"])
            for cap in eval(row["raw"]):  # stored as a python-list literal string
                items.append({"image_id": str(img_id), "image_path": img_path, "caption": cap.strip()})
    if max_items:
        items = items[:max_items]
    return items


def load_flickr30k_items_local(luongtk_flickr_root, seed=42, n=1000, max_items=None):
    """ALTERNATE source: /cm/archive/luongtk/flickr/ (captions.txt + Images/), already present
    on this server -- no download needed. CAVEAT: captions.txt has NO split column, so this is
    a seeded 1000-image proxy subset, NOT the paper's exact Karpathy 1K test list. Must use the
    SAME seed/n as precompute_llm2vec_embeddings.py's load_flickr30k_local_eval_captions (both
    default seed=42, n=1000) or the cache will miss."""
    import csv
    import random
    csv_path = os.path.join(luongtk_flickr_root, "captions.txt")
    image_root = os.path.join(luongtk_flickr_root, "Images")
    images = set()
    with open(csv_path, encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        images.add(row["image"])
    selected = set(random.Random(seed).sample(sorted(images), min(n, len(images))))
    items = []
    for row in rows:
        if row["image"] not in selected:
            continue
        items.append({
            "image_id": row["image"],
            "image_path": os.path.join(image_root, row["image"]),
            "caption": row["caption"].strip(),
        })
    if max_items:
        items = items[:max_items]
    return items


def load_coco_items(eval_data_root, max_items=None):
    ann_path = os.path.join(eval_data_root, "coco", "annotations", "captions_val2017.json")
    image_root = os.path.join(eval_data_root, "coco", "val2017")
    data = json.load(open(ann_path, encoding="utf8"))
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    items = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        items.append({
            "image_id": str(img_id),
            "image_path": os.path.join(image_root, id_to_file[img_id]),
            "caption": ann["caption"].strip(),
        })
    if max_items:
        items = items[:max_items]
    return items


def load_urban1k_items(eval_data_root, max_items=None, urban1k_root=None):
    root = urban1k_root or os.path.join(eval_data_root, "urban1k", "Urban1k")
    cap_dir, img_dir = os.path.join(root, "caption"), os.path.join(root, "image")
    ids = sorted(fn[:-4] for fn in os.listdir(cap_dir) if fn.endswith(".txt"))
    if max_items:
        ids = ids[:max_items]
    items = []
    for uid in ids:
        with open(os.path.join(cap_dir, f"{uid}.txt"), encoding="utf8") as f:
            caption = f.read().strip().replace("\n", " ")
        items.append({"image_id": uid, "image_path": os.path.join(img_dir, f"{uid}.jpg"), "caption": caption})
    return items


def load_sg4v1k_items(sharegpt4v_full_json, sharegpt4v_image_root, max_items=None):
    """Paper's 'SG4V 1K subset' -- no separately-published LongCLIP list found. Uses this
    project's own share4v_val_dataset convention (first 1000 items, original file order) of
    the SAME source JSON the paper describes, as the closest available proxy. NOT guaranteed
    bit-identical to LongCLIP's specific list -- report this as a caveat, not as identical.
    Requires the FULL image tree (sam/images/..., coco/..., etc, i.e. the server's
    /cm/archive/luongtk/sharegpt4v/data/) -- only a subset of images may exist elsewhere."""
    data = json.load(open(sharegpt4v_full_json, encoding="utf8"))[:1000]
    if max_items:
        data = data[:max_items]
    items = []
    for i, d in enumerate(data):
        caption = d["conversations"][1]["value"].replace("\n", " ")
        items.append({
            "image_id": str(i),
            "image_path": os.path.join(sharegpt4v_image_root, d["image"]),
            "caption": caption,
        })
    return items


def load_sg4v1k_items_from_manifest(manifest_json, image_root, split="test", max_items=None):
    """ALTERNATE sg4v1k source for machines that only have a pre-built manifest.json (e.g. from
    make_sharegpt4v_subset.py / download_sharegpt4v_coco_subset.py) with flat-named images
    already downloaded, rather than the full 1.246M-item JSON + original sam/coco/... image
    tree. Caveat: this is a RANDOM subset (seeded shuffle) of the full corpus, not the paper's
    (or load_sg4v1k_items's) 'first 1000 in file order' convention, and its test split size
    depends on how the manifest was built (e.g. 799, not exactly 1000) -- a DIFFERENT proxy
    than load_sg4v1k_items, report which one was used in results."""
    data = json.load(open(manifest_json, encoding="utf8"))
    data = [d for d in data if d.get("split") == split]
    if max_items:
        data = data[:max_items]
    items = []
    for i, d in enumerate(data):
        items.append({
            "image_id": str(i),
            "image_path": os.path.join(image_root, d["image"]),
            "caption": d["caption"].replace("\n", " "),
        })
    return items


def load_docci_items(docci_json, docci_image_root, splits=("test", "qual_test"), max_items=None):
    """Paper says DOCCI eval = '1.5K high-resolution images'; this project's test+qual_test
    split is 5100 images (5000+100), not an exact size match -- pass --max_items 1500 for a
    closer-to-paper-size gallery if wanted, or leave unset to eval the full split."""
    data = json.load(open(docci_json, encoding="utf8"))
    data = [d for d in data if d.get("split") in splits]
    if max_items:
        data = data[:max_items]
    items = []
    for i, d in enumerate(data):
        caption = d["conversations"][1]["value"].replace("\n", " ")
        items.append({
            "image_id": d.get("image", str(i)),
            "image_path": os.path.join(docci_image_root, d["image"]),
            "caption": caption,
        })
    return items


def _load_sg4v1k(args):
    if args.sg4v_source == "manifest":
        return load_sg4v1k_items_from_manifest(args.sg4v_manifest_json, args.sg4v_manifest_image_root,
                                                max_items=args.max_items)
    return load_sg4v1k_items(args.sharegpt4v_full_json, args.sharegpt4v_image_root, args.max_items)


def _load_flickr30k(args):
    if args.flickr_source == "luongtk_local":
        return load_flickr30k_items_local(args.flickr_luongtk_root, max_items=args.max_items)
    return load_flickr30k_items(args.eval_data_root, args.max_items)


LOADERS = {
    "flickr30k": _load_flickr30k,
    "coco": lambda args: load_coco_items(args.eval_data_root, args.max_items),
    "urban1k": lambda args: load_urban1k_items(args.eval_data_root, args.max_items, args.urban1k_root),
    "sg4v1k": _load_sg4v1k,
    "docci": lambda args: load_docci_items(args.docci_json, args.docci_image_root, max_items=args.max_items),
}


# ---- Dataset / model loading -----------------------------------------------------------------

class ItemsDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        image = Image.open(it["image_path"]).convert("RGB")
        return self.preprocess(image), it["caption"], it["image_id"]


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


def load_model(args, device):
    """Two checkpoint families exist in this project (see plan doc Nhóm B/C):
      - ViT-B/16 + Llama-CC adapter (LLM2CLIPTextTeacher, --base_model llm2clip_text/llm2clip_frozen)
      - ViT-L/14@336 released + Llama-CC adapter (LLM2CLIPReleasedTeacher, --base_model llm2clip_released)
    --arch selects which; --ckpt=RELEASED (or --released) loads Microsoft's untouched checkpoint
    with no fine-tuning at all, as a zero-fine-tune reference point."""
    if args.released or args.ckpt == "RELEASED":
        model = LLM2CLIPReleasedTeacher(model_path=args.llm2clip_released_ckpt, device="cpu")
        preprocess_model = "ViT-L/14@336px"
    elif args.arch == "released":
        model = LLM2CLIPReleasedTeacher(model_path=args.llm2clip_released_ckpt, device="cpu")
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd)
        preprocess_model = "ViT-L/14@336px"
    else:  # arch == "text" (ViT-B/16)
        model = LLM2CLIPTextTeacher(clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                                    freeze_visual=False, device="cpu", adapter_type=args.adapter_type)
        sd = _remap_legacy_adapter_keys(torch.load(args.ckpt, map_location="cpu"))
        model.load_state_dict(sd)
        preprocess_model = "ViT-B/16"
    model = model.to(device).eval()
    _, preprocess = clip.load(preprocess_model, device="cpu")
    return model, preprocess


# ---- Evaluation (multi-caption-safe: also correct for the 1-caption/image benchmarks) --------

@torch.no_grad()
def evaluate(model, cache, dataloader, device):
    def cpu_fp32(x):
        return x.detach().to("cpu", dtype=torch.float32)

    img_feat_cache: Dict[str, torch.Tensor] = {}
    txt_feats_parts: List[torch.Tensor] = []
    cap2img: List[str] = []

    for imgs, caps, img_ids in tqdm(dataloader, desc="Extracting features"):
        new_ids, new_imgs = [], []
        for img, iid in zip(imgs, img_ids):
            if iid not in img_feat_cache:
                new_ids.append(iid)
                new_imgs.append(img)
        if new_imgs:
            batch_imgs = torch.stack(new_imgs).to(device, non_blocking=True)
            f_i = F.normalize(model.encode_image(batch_imgs), dim=-1)
            for iid, f in zip(new_ids, cpu_fp32(f_i)):
                img_feat_cache[iid] = f

        t_emb = cache.lookup(list(caps))
        f_t = F.normalize(model.encode_text(t_emb), dim=-1)
        txt_feats_parts.append(cpu_fp32(f_t))
        cap2img.extend(img_ids)

    img_ids_list = list(img_feat_cache.keys())
    img_feats = torch.stack([img_feat_cache[k] for k in img_ids_list]).float()  # [I, D]
    txt_feats = torch.cat(txt_feats_parts, dim=0).float()                       # [M, D]

    sims_t2i = txt_feats @ img_feats.T   # [M, I]
    sims_i2t = img_feats @ txt_feats.T   # [I, M]

    ks = (1, 5, 10)
    acc = {}

    # T2I: hit if the caption's own ground-truth image is in top-k image results.
    for k in ks:
        topk = sims_t2i.topk(k, dim=1).indices
        correct = sum(
            1 for i, row in enumerate(topk.tolist())
            if any(img_ids_list[j] == cap2img[i] for j in row)
        )
        acc[f"T2I_R{k}"] = correct / len(cap2img)

    # I2T: hit if any of the image's ground-truth captions is in top-k text results.
    imgid2capidx: Dict[str, List[int]] = {}
    for cap_idx, iid in enumerate(cap2img):
        imgid2capidx.setdefault(iid, []).append(cap_idx)
    for k in ks:
        topk = sims_i2t.topk(k, dim=1).indices
        correct = sum(
            1 for i, row in enumerate(topk.tolist())
            if any(idx in imgid2capidx.get(img_ids_list[i], []) for idx in row)
        )
        acc[f"I2T_R{k}"] = correct / len(img_ids_list)

    return acc, len(img_ids_list), len(cap2img)


def run_one(benchmark, args, device):
    items = LOADERS[benchmark](args)
    print(f"[data] {benchmark}: {len(items)} (image, caption) pairs")

    cache_dir = os.path.join(args.cache_dir, benchmark)
    cache = TextEmbeddingCache(cache_dir, device=device)

    model, preprocess = load_model(args, device)
    ds = ItemsDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    acc, n_images, n_caps = evaluate(model, cache, dl, device)
    result = {"benchmark": benchmark, "ckpt": args.ckpt if not args.released else "RELEASED",
              "arch": "released" if (args.released or args.arch == "released") else args.arch,
              "n_images": n_images, "n_captions": n_caps, **acc}
    print(f"[RESULT] {benchmark}: " + ", ".join(f"{k}={v:.4%}" for k, v in acc.items()))
    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["flickr30k", "coco", "urban1k", "sg4v1k", "docci", "all"],
                    required=True)
    ap.add_argument("--ckpt", default="RELEASED")
    ap.add_argument("--released", action="store_true",
                    help="Eval Microsoft's untouched released checkpoint (no fine-tuning at all).")
    ap.add_argument("--arch", choices=["text", "released"], default="released",
                    help="'text' = ViT-B/16 LLM2CLIPTextTeacher ckpt (Nhóm B); "
                         "'released' = ViT-L/14@336 LLM2CLIPReleasedTeacher ckpt (Nhóm C).")
    ap.add_argument("--adapter_type", choices=["linear", "mlp"], default="linear",
                    help="Only used when --arch text (must match how the ckpt was trained).")
    ap.add_argument("--cache_dir", required=True,
                    help="Parent dir of per-benchmark part-file caches (flickr30k/, coco/, ...).")
    ap.add_argument("--eval_data_root", default=EVAL_DATA_ROOT)
    ap.add_argument("--docci_json", default=DOCCI_JSON)
    ap.add_argument("--docci_image_root", default=DOCCI_IMAGE_ROOT)
    ap.add_argument("--sharegpt4v_full_json", default=SHAREGPT4V_FULL_JSON)
    ap.add_argument("--sharegpt4v_image_root", default=SHAREGPT4V_IMAGE_ROOT)
    ap.add_argument("--sg4v_source", choices=["full_json", "manifest"], default="full_json",
                    help="'full_json' = server (first 1000 of raw 1.246M JSON, needs full image "
                         "tree); 'manifest' = local machine (pre-built manifest.json's test split).")
    ap.add_argument("--sg4v_manifest_json", default=None)
    ap.add_argument("--sg4v_manifest_image_root", default=None)
    ap.add_argument("--urban1k_root", default=None,
                    help="Direct override to an existing caption/+image/ root "
                         "(e.g. /cm/archive/luongtk/Urban1k) -- skips the eval_data_root layout.")
    ap.add_argument("--flickr_source", choices=["hf", "luongtk_local"], default="hf",
                    help="'hf' = Karpathy-split CSV (paper-accurate). 'luongtk_local' = plain "
                         "image,caption CSV with no split info -- seeded 1000-image proxy.")
    ap.add_argument("--flickr_luongtk_root", default=None,
                    help="Root containing captions.txt + Images/ (for --flickr_source luongtk_local).")
    ap.add_argument("--llm2clip_released_ckpt", default=LLM2CLIP_RELEASED_CKPT)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmarks = list(LOADERS.keys()) if args.benchmark == "all" else [args.benchmark]

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    # Resume support: skip benchmarks already present in an existing --out_json (e.g. from a
    # prior run that crashed partway through --benchmark all) instead of redoing them.
    results = []
    done_benchmarks = set()
    if os.path.exists(args.out_json):
        results = json.load(open(args.out_json))
        done_benchmarks = {r["benchmark"] for r in results}
        if done_benchmarks:
            print(f"[resume] {args.out_json} already has: {sorted(done_benchmarks)}")

    for b in benchmarks:
        if b in done_benchmarks:
            print(f"[skip] {b} already in {args.out_json}")
            continue
        results.append(run_one(b, args, device))
        # Write after EVERY benchmark, not just at the end -- a crash on benchmark 4/5 must not
        # lose the first 3 (this bit us once already on the precompute side of this project).
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[checkpoint] saved {len(results)}/{len(benchmarks)} benchmark result(s) to {args.out_json}")

    print(f"[done] wrote {len(results)} benchmark result(s) to {args.out_json}")


if __name__ == "__main__":
    main()
