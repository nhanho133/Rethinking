#!/bin/bash
# Auto-generated deploy script — paste toàn bộ vào terminal server để ghi đè/tạo 7 file
# Chạy từ ~/Nhan_folder/train (hoặc đúng thư mục train/ trên server)
set -e

echo '--- writing eval_paper_benchmarks.py ---'
mkdir -p $(dirname "eval_paper_benchmarks.py")
cat > "eval_paper_benchmarks.py" << 'DEPLOY_EOF_MARKER'
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


def load_urban1k_items(eval_data_root, max_items=None):
    root = os.path.join(eval_data_root, "urban1k", "Urban1k")
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


LOADERS = {
    "flickr30k": lambda args: load_flickr30k_items(args.eval_data_root, args.max_items),
    "coco": lambda args: load_coco_items(args.eval_data_root, args.max_items),
    "urban1k": lambda args: load_urban1k_items(args.eval_data_root, args.max_items),
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
DEPLOY_EOF_MARKER

echo '--- writing precompute_llm2vec_embeddings.py ---'
mkdir -p $(dirname "precompute_llm2vec_embeddings.py")
cat > "precompute_llm2vec_embeddings.py" << 'DEPLOY_EOF_MARKER'
"""
Offline, one-time precompute of LLM2Vec (frozen Llama-3-8B-CC) text embeddings for every
caption/sub-caption string training could ever need, cached to disk keyed by sha1(text).
This lets train.py run the LLM2CLIP-text-teacher branch without ever loading the 8B model
during actual training.

Part-file design (fixes RAM OOM on full-scale runs): the full ShareGPT4V set produces ~20M
unique strings; holding all their float16[4096] embeddings in a single in-RAM dict is ~160GB
and OOM-kills the process partway. Instead we split the (deterministically sorted) string
list into fixed-size PARTS (default 1M strings), embed one part at a time, write it to its own
part_NNNNN.pt file, then free that part's dict from RAM before starting the next. Peak RAM is
therefore ~one part (~8GB) regardless of total size.

Crash-safety / resume: each part is written to a .tmp then atomically os.replace()'d, so a
part file only ever exists once fully written. On restart, any part whose file already exists
is skipped -> a crash loses at most the single in-progress part (re-embedded from scratch),
never a completed one. The sort key (len(s), s) is fully deterministic across processes/runs,
so part boundaries are identical every run and resume lands exactly where it left off.

Coverage strategy: star_bar_long_text_split / make_base_longer's only randomness at train time
is which pre-existing deterministic candidate gets picked; the candidate pools are fully
enumerable from (sentences, max_num_short_texts, fixed seed=42), so we enumerate every string
that could ever be produced rather than sampling simulated epochs.
"""
import argparse
import gc
import hashlib
import itertools
import json
import os
import sys

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from llm2vec import LLM2Vec

sys.path.append(os.path.dirname(__file__))
from sampling import star_bar_long_text_split

# H100 server paths
DOCCI_JSON = "/cm/archive/luongtk/docci/captioner_docci.json"
CC3M_MANIFEST = "/cm/shared/chautvh_second/Nhan_folder/work/cc3m/manifest.json"
SHAREGPT4V_MANIFEST = "/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v_subset_manifest.json"
LLM_PATH = "/cm/shared/chautvh_second/Nhan_folder/ckpts/Llama-3-8B-CC"
SPLITS_NEEDED = ("train", "test", "qual_test")  # matches dataset_mapping["docci"] in datasets_config.py
SEED = 42

# Paper-protocol zero-shot eval benchmarks (arXiv:2411.04997 Table 2) -- separate from the
# training-data sources above. Override with --eval_data_root if paths differ on your server.
EVAL_DATA_ROOT = "/cm/shared/chautvh_second/Nhan_folder/work/eval_data"
SHAREGPT4V_FULL_JSON = "/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"


def split_into_detail_captions(text_long):
    """Copied verbatim from train.py's CLIP_Clean_Train.split_into_detail_captions (pure
    function, no self-dependency) so this script doesn't need to instantiate the trainer."""
    sentences = [
        p.strip()
        for p in text_long.split('.')
        if p.strip()
        and len(p.strip()) >= 18
        and not (len(p.strip()) == 1 and p.strip().isalpha())
    ]
    return sentences


def enumerate_pos_pool(sentences, max_num_short_texts):
    """Every chunk string star_bar_long_text_split could ever hand back for this caption,
    covering both the main path (len(sentences) >= K) and train.py's fallback padding path."""
    n = len(sentences)
    if n == 0:
        return []
    if n >= max_num_short_texts:
        return list(star_bar_long_text_split(sentences, max_num_short_texts, SEED))
    pool = list(star_bar_long_text_split(sentences, n, SEED))
    for new_star in range(1, n + 1):
        pool.extend(star_bar_long_text_split(sentences, new_star, SEED))
    seen = set()
    out = []
    for c in pool:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def enumerate_longer_pool(pos_pool, full=False):
    """make_base_longer joins two chunks per slot. n>=K: unordered combinations reproduce it;
    n<K (short-caption fallback): full Cartesian product (both orders + self-pairs) to be
    miss-proof."""
    out = []
    if full:
        for a in range(len(pos_pool)):
            for b in range(len(pos_pool)):
                out.append((pos_pool[a].strip() + " " + pos_pool[b].strip()).strip())
    else:
        for a, b in itertools.combinations(range(len(pos_pool)), 2):
            out.append((pos_pool[a].strip() + " " + pos_pool[b].strip()).strip())
    return out


def load_docci_captions():
    with open(DOCCI_JSON, "r", encoding="utf8") as fp:
        data = json.load(fp)
    items = [d for d in data if d.get("split") in SPLITS_NEEDED]
    print(f"[collect] {len(items)} DOCCI items across splits {SPLITS_NEEDED}")
    out = []
    for item in items:
        caption = item["conversations"][1]["value"].replace("\n", " ")
        caption_short = caption.split(".")[0].strip() + "."
        out.append((caption, caption_short))
    return out


def load_sharegpt4v_captions():
    with open(SHAREGPT4V_MANIFEST, "r", encoding="utf8") as fp:
        data = json.load(fp)
    print(f"[collect] {len(data)} ShareGPT4V-COCO items")
    out = []
    for item in data:
        caption = item["caption"].replace("\n", " ")
        caption_short = item["caption_short"].replace("\n", " ")
        out.append((caption, caption_short))
    return out


def load_cc3m_captions():
    with open(CC3M_MANIFEST, "r", encoding="utf8") as fp:
        data = json.load(fp)
    print(f"[collect] {len(data)} DreamLIP-CC3M items")
    out = []
    for item in data:
        caption = item["caption"].replace("\n", " ")
        caption_short = item["caption_short"].replace("\n", " ")
        out.append((caption, caption_short))
    return out


def collect_all_strings(max_num_short_texts, dataset="docci"):
    loader = {"docci": load_docci_captions, "dreamlip_cc3m": load_cc3m_captions,
              "sharegpt4v_coco": load_sharegpt4v_captions}[dataset]
    caption_pairs = loader()

    all_strings = set()
    all_strings.add("")  # test_epoch_ver5 pads missing detail slots with "" and encodes it
    for caption, caption_short in caption_pairs:
        all_strings.add(caption)
        all_strings.add(caption_short)

        sentences = split_into_detail_captions(caption)
        all_strings.update(sentences)

        pos_pool = enumerate_pos_pool(sentences, max_num_short_texts)
        if not pos_pool:
            continue
        all_strings.update(pos_pool)
        is_fallback = len(sentences) < max_num_short_texts
        all_strings.update(enumerate_longer_pool(pos_pool, full=is_fallback))

    print(f"[collect] {len(all_strings)} unique strings need embeddings")
    return all_strings


# ---- Paper-protocol zero-shot EVAL benchmark loaders -----------------------------------
# These are deliberately NOT run through split_into_detail_captions / star_bar_long_text_split /
# make_base_longer -- that machinery is training-time sub-caption AUGMENTATION for the
# Rethinking loss. A fixed eval gallery (Flickr 1K, COCO 5K, Urban1K 1K, SG4V 1K, DOCCI test)
# just needs every raw caption string embedded once, nothing enumerated/combined.

def load_flickr30k_eval_captions(eval_data_root):
    """Karpathy-split test partition (1000 images), ALL captions per image (paper protocol),
    matching eval_zeroshot_retrieval.py::load_flickr30k's split filter."""
    import csv
    csv_path = os.path.join(eval_data_root, "flickr30k", "flickr_annotations_30k.csv")
    out = []
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            out.extend(c.strip() for c in eval(row["raw"]))
    print(f"[collect] Flickr30K test: {out.__len__()} captions (from images filtered split=='test')")
    return out


def load_coco_eval_captions(eval_data_root):
    """COCO val2017, ALL captions per image (paper protocol: 5K images x ~5 captions)."""
    ann_path = os.path.join(eval_data_root, "coco", "annotations", "captions_val2017.json")
    data = json.load(open(ann_path, encoding="utf8"))
    out = [ann["caption"].strip() for ann in data["annotations"]]
    print(f"[collect] COCO val2017: {len(out)} captions")
    return out


def load_urban1k_eval_captions(eval_data_root):
    """1000 images, 1 caption/image (long, dense -- like DOCCI/SG4V, not short alt-text)."""
    cap_dir = os.path.join(eval_data_root, "urban1k", "Urban1k", "caption")
    out = []
    for fname in sorted(os.listdir(cap_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(cap_dir, fname), encoding="utf8") as f:
            out.append(f.read().strip().replace("\n", " "))
    print(f"[collect] Urban1K: {len(out)} captions")
    return out


def load_sg4v1k_eval_captions(sg4v_source="full_json", manifest_json=None):
    """Paper's 'SG4V 1K subset' -- no separately-published LongCLIP list found.
    sg4v_source='full_json' (server): first 1000 items (original file order) of the same source
      JSON the paper describes -- needs the full image tree, server-only.
    sg4v_source='manifest' (local machine, no full image tree available): test split of a
      pre-built manifest.json (make_sharegpt4v_subset.py-style) whose images are already
      downloaded locally -- a DIFFERENT (random-subset) proxy, must match whatever
      eval_paper_benchmarks.py --sg4v_source was used at eval time or the cache will miss."""
    if sg4v_source == "manifest":
        data = json.load(open(manifest_json, encoding="utf8"))
        data = [d for d in data if d.get("split") == "test"]
        out = [d["caption"].replace("\n", " ") for d in data]
        print(f"[collect] SG4V-1K (manifest proxy, {manifest_json} test split): {len(out)} captions")
        return out
    data = json.load(open(SHAREGPT4V_FULL_JSON, encoding="utf8"))[:1000]
    out = [d["conversations"][1]["value"].replace("\n", " ") for d in data]
    print(f"[collect] SG4V-1K (proxy, first 1000 of {SHAREGPT4V_FULL_JSON}): {len(out)} captions")
    return out


def collect_eval_strings(dataset, eval_data_root, sg4v_source="full_json", sg4v_manifest_json=None):
    loader = {
        "flickr30k": lambda: load_flickr30k_eval_captions(eval_data_root),
        "coco": lambda: load_coco_eval_captions(eval_data_root),
        "urban1k": lambda: load_urban1k_eval_captions(eval_data_root),
        "sg4v1k": lambda: load_sg4v1k_eval_captions(sg4v_source, sg4v_manifest_json),
    }[dataset]
    captions = loader()
    all_strings = set(captions)
    all_strings.add("")
    print(f"[collect] {len(all_strings)} unique strings need embeddings ({dataset} eval)")
    return all_strings


def _patch_bnb_to():
    """transformers 4.44.2 vs accelerate 1.14.0: for a 4-bit model fully on one GPU,
    dispatch_model calls model.to(device), which PreTrainedModel.to hard-raises for bnb models.
    The model is already placed, so a device-only .to() is a safe no-op."""
    from transformers.modeling_utils import PreTrainedModel
    from transformers.utils.quantization_config import QuantizationMethod
    _orig_to = PreTrainedModel.to

    def _safe_to(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            has_dtype = ("dtype" in kwargs) or any(isinstance(a, torch.dtype) for a in args)
            if not has_dtype:
                return self
        return _orig_to(self, *args, **kwargs)

    PreTrainedModel.to = _safe_to


def load_l2v(quant_4bit=True):
    if quant_4bit:
        _patch_bnb_to()
    config = AutoConfig.from_pretrained(LLM_PATH, trust_remote_code=True)
    quant_kwargs = {}
    if quant_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_kwargs["torch_dtype"] = torch.bfloat16

    llm_model = AutoModel.from_pretrained(
        LLM_PATH, config=config, trust_remote_code=True, device_map={"": 0}, **quant_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
    return l2v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="docci",
                    choices=["docci", "dreamlip_cc3m", "sharegpt4v_coco",
                             "flickr30k", "coco", "urban1k", "sg4v1k"])
    ap.add_argument("--eval_mode", action="store_true",
                    help="For flickr30k/coco/urban1k/sg4v1k: just embed raw captions "
                         "(no sub-caption augmentation) -- required for these 4 datasets.")
    ap.add_argument("--eval_data_root", type=str, default=EVAL_DATA_ROOT,
                    help="Root dir containing flickr30k/, coco/, urban1k/ subfolders.")
    ap.add_argument("--sg4v_source", choices=["full_json", "manifest"], default="full_json",
                    help="For --dataset sg4v1k --eval_mode: 'full_json' (server) or "
                         "'manifest' (local, no full image tree).")
    ap.add_argument("--sg4v_manifest_json", type=str, default=None)
    ap.add_argument("--max_num_short_texts", type=int, default=4)
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to write part_NNNNN.pt files (one per part_size chunk).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--part_size", type=int, default=1_000_000,
                    help="Strings per part file. Peak RAM ~ part_size * 8KB. A crash re-does at "
                         "most one in-progress part.")
    ap.add_argument("--no_4bit", action="store_true", help="Load in bf16 instead of 4-bit.")
    ap.add_argument("--dry_run_n", type=int, default=None,
                    help="If set, only embed this many strings (smoke test).")
    ap.add_argument("--llm_path", type=str, default=None,
                    help="Override LLM_PATH (e.g. a local Llama-3-8B-CC checkpoint dir).")
    ap.add_argument("--docci_json", type=str, default=None,
                    help="Override DOCCI_JSON path (e.g. a local captioner_docci.json).")
    args = ap.parse_args()

    # Path overrides for running off-server (e.g. local machine). The loaders read these as
    # module globals, so rebind them before collect_all_strings / load_l2v are called.
    global LLM_PATH, DOCCI_JSON
    if args.llm_path:
        LLM_PATH = args.llm_path
    if args.docci_json:
        DOCCI_JSON = args.docci_json

    EVAL_DATASETS = ("flickr30k", "coco", "urban1k", "sg4v1k")
    if args.dataset in EVAL_DATASETS and not args.eval_mode:
        raise SystemExit(f"--dataset {args.dataset} requires --eval_mode "
                          f"(it has no sub-caption augmentation to enumerate).")

    # key=(len, s): sort by length (length-homogeneous batches -> minimal padding waste) with a
    # fully deterministic tie-break, so part boundaries are identical every run (safe resume).
    if args.eval_mode:
        strings = sorted(collect_eval_strings(args.dataset, args.eval_data_root,
                                              sg4v_source=args.sg4v_source,
                                              sg4v_manifest_json=args.sg4v_manifest_json),
                         key=lambda s: (len(s), s))
    else:
        strings = sorted(collect_all_strings(args.max_num_short_texts, dataset=args.dataset),
                         key=lambda s: (len(s), s))
    if args.dry_run_n:
        strings = strings[: args.dry_run_n]
        print(f"[dry-run] truncated to {len(strings)} strings")

    os.makedirs(args.out_dir, exist_ok=True)
    n = len(strings)
    n_parts = (n + args.part_size - 1) // args.part_size
    print(f"[plan] {n} strings -> {n_parts} parts of up to {args.part_size} each, into {args.out_dir}")

    l2v = None  # lazy: don't load the 8B model if every part is already done (pure resume)
    for pi in range(n_parts):
        part_path = os.path.join(args.out_dir, f"part_{pi:05d}.pt")
        if os.path.exists(part_path):
            print(f"[skip] part {pi+1}/{n_parts} already done: {part_path}")
            continue

        part = strings[pi * args.part_size : (pi + 1) * args.part_size]
        if l2v is None:
            print("[load] loading LLM2Vec (Llama-3-8B-CC)"
                  + (" [4-bit]" if not args.no_4bit else " [bf16]") + " ...")
            l2v = load_l2v(quant_4bit=not args.no_4bit)
            print("[load] model loaded, VRAM:",
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "cpu")

        cache = {}
        m = len(part)
        for i in range(0, m, args.batch_size):
            batch = part[i : i + args.batch_size]
            with torch.no_grad():
                embs = l2v.encode(batch, convert_to_tensor=True)
            embs = embs.to(torch.float16).cpu()
            for text, emb in zip(batch, embs):
                if not torch.isfinite(emb).all():
                    emb = torch.zeros_like(emb)  # empty-pad string can pool to NaN -> store zeros
                cache[hashlib.sha1(text.encode("utf8")).hexdigest()] = emb
            if (i // args.batch_size) % 20 == 0:
                print(f"[part {pi+1}/{n_parts}] {min(i+args.batch_size, m)}/{m}")

        # Atomic write: only appears at part_path once fully saved, so a crash mid-save never
        # leaves a truncated file that resume would wrongly skip.
        tmp_path = part_path + ".tmp"
        torch.save({"cache": cache, "dim": next(iter(cache.values())).shape[0]}, tmp_path)
        os.replace(tmp_path, part_path)
        print(f"[saved] part {pi+1}/{n_parts}: {part_path} ({len(cache)} embeddings)")

        del cache
        gc.collect()

    print(f"[done] all {n_parts} parts complete in {args.out_dir}")


if __name__ == "__main__":
    main()
DEPLOY_EOF_MARKER

echo '--- writing assemble_paper_comparison.py ---'
mkdir -p $(dirname "assemble_paper_comparison.py")
cat > "assemble_paper_comparison.py" << 'DEPLOY_EOF_MARKER'
"""
Assembles the final comparison table: paper-cited reference rows (from
paper_reference_numbers.json) + our own measured rows (from one or more
eval_paper_benchmarks.py --out_json result files), across the 5 benchmarks
(Flickr30K, COCO, SG4V-1K, Urban1K, DOCCI) x {I2T, T2I} x R@1.

Usage:
  python assemble_paper_comparison.py \
    --result vanilla:results/vanilla.json \
    --result full:results/full.json \
    --result released:results/released.json \
    --out eval_results/comparison.md
"""
import argparse
import json
import os

BENCHMARKS = ["flickr30k", "coco", "sg4v1k", "urban1k", "docci"]
BENCHMARK_LABELS = {"flickr30k": "Flickr", "coco": "COCO", "sg4v1k": "SG4V",
                     "urban1k": "Urban1K", "docci": "DOCCI"}
REF_ROWS = [
    ("clip_l14_336_baseline", "CLIP-L/14-336 baseline (cited, paper Table 2)"),
    ("llm2clip_15m_336", "LLM2CLIP-15M, ViT-L/14@336 (cited, paper Table 2 -- more data than ours, see caveat)"),
]


def load_measured_row(path):
    """path = eval_paper_benchmarks.py --out_json output: list of per-benchmark dicts."""
    data = json.load(open(path, encoding="utf8"))
    row = {}
    for entry in data:
        b = entry["benchmark"]
        row[b] = {"I2T_R1": entry.get("I2T_R1"), "T2I_R1": entry.get("T2I_R1")}
    return row


def fmt(v):
    return f"{v*100:.1f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", default=[],
                    help="label:path_to_out_json, repeatable (e.g. --result vanilla:results/vanilla.json)")
    ap.add_argument("--reference_json", default=os.path.join(os.path.dirname(__file__), "paper_reference_numbers.json"))
    ap.add_argument("--out", required=True, help="Output .md path (a .csv is written alongside)")
    args = ap.parse_args()

    ref = json.load(open(args.reference_json, encoding="utf8"))

    rows = []  # (label, {benchmark: {I2T_R1, T2I_R1}}, is_cited)
    for key, label in REF_ROWS:
        rows.append((label, ref[key], True))
    for item in args.result:
        label, path = item.split(":", 1)
        rows.append((label, load_measured_row(path), False))

    header = ["Config"] + [f"{BENCHMARK_LABELS[b]} {d}" for b in BENCHMARKS for d in ("I2T", "T2I")]
    lines_md = ["| " + " | ".join(header) + " |",
                "|" + "---|" * len(header)]
    lines_csv = [",".join(header)]

    for label, row, is_cited in rows:
        cells = [label + (" *(cited)*" if is_cited else "")]
        cells_csv = [label + (" (cited)" if is_cited else "")]
        for b in BENCHMARKS:
            d = row.get(b, {})
            cells.append(fmt(d.get("I2T_R1")))
            cells.append(fmt(d.get("T2I_R1")))
            cells_csv.append(str(d.get("I2T_R1", "")))
            cells_csv.append(str(d.get("T2I_R1", "")))
        lines_md.append("| " + " | ".join(cells) + " |")
        lines_csv.append(",".join(cells_csv))

    footnote = (
        "\n**Lưu ý**: 2 hàng *(cited)* lấy thẳng từ paper Table 2, dùng nhiều data hơn "
        "(15M cặp) và không có bản '336px + CC3M-only' được công bố để so khớp tuyệt đối "
        "-- chỉ mang tính tham khảo, KHÔNG phải baseline kiểm soát chặt. Phép so sánh có "
        "kiểm soát thật sự (cùng data, cùng checkpoint xuất phát, cùng lịch train) là giữa "
        "các hàng đo được (không có *(cited)*) với nhau.\n"
    )

    md = "\n".join(lines_md) + "\n" + footnote
    with open(args.out, "w", encoding="utf8") as f:
        f.write(md)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines_csv) + "\n")

    print(md)
    print(f"[done] wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
DEPLOY_EOF_MARKER

echo '--- writing eval_zeroshot_retrieval.py ---'
mkdir -p $(dirname "eval_zeroshot_retrieval.py")
cat > "eval_zeroshot_retrieval.py" << 'DEPLOY_EOF_MARKER'
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
DEPLOY_EOF_MARKER

echo '--- writing urban1k.py ---'
mkdir -p $(dirname "urban1k.py")
cat > "urban1k.py" << 'DEPLOY_EOF_MARKER'
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from dci import JsonDCIDataset
# import clip
# from train import CLIP_Clean_Train

# class Urban1kDataset(Dataset):
#     """
#     Dataset for Urban1k: pairs each image/ caption by filename (without extension).
#     """
#     def __init__(self, root_dir, max_items=None, device='cuda'):
#         self.image_dir   = os.path.join(root_dir, 'image')
#         self.caption_dir = os.path.join(root_dir, 'caption')
#         # all caption files (strip “.txt”)
#         self.ids = sorted([
#             fname[:-4] for fname in os.listdir(self.caption_dir)
#             if fname.endswith('.txt')
#         ])
#         if max_items is not None:
#             self.ids = self.ids[:max_items]

#         # load CLIP preprocess
#         self.device     = torch.device(device)
#         _, self.preprocess = clip.load('ViT-B/16', device=self.device)

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         idx = self.ids[idx]
#         # load caption
#         with open(os.path.join(self.caption_dir, f'{idx}.txt'), 'r', encoding='utf8') as f:
#             caption = f.read().strip().replace('\n', ' ')
#         # load & preprocess image
#         img_path = os.path.join(self.image_dir, f'{idx}.jpg')
#         image    = Image.open(img_path).convert('RGB')
#         image_t  = self.preprocess(image)
#         return image_t, caption, "None", img_path, "None"

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class Urban1kDataset(Dataset):
    """
    Dataset for Urban1k: pairs each image/caption by filename (without extension).
    Supports an internal 80/20 split via `split='train'` or `split='val'`.
    """
    def __init__(self,
                 root_dir,
                 split: str = 'train',
                 split_ratio: float = 0.8,
                 seed: int = 42,
                 max_items: int = None,
                 device: str = 'cuda',
                 model_name: str = 'ViT-B/16',
                 use_full_split: bool = False):
        """model_name: CLIP preprocess to use (e.g. 'ViT-L/14@336px' for llm2clip_released
        checkpoints -- was hardcoded to 'ViT-B/16' before, which mismatches that model's
        expected input resolution/normalization).
        use_full_split=True: paper's "Urban1K" benchmark is the FULL 1000 images as a single
        eval set (no 80/20 train/val carve-out) -- set this for zero-shot benchmark eval;
        the default 80/20 behavior is kept for backward compat with any existing callers."""
        assert split in ('train', 'val'), "split phải là 'train' hoặc 'val'"
        self.image_dir   = os.path.join(root_dir, 'image')
        self.caption_dir = os.path.join(root_dir, 'caption')

        # Lấy danh sách id (filename không extension) và shuffle
        ids = sorted([fname[:-4]
                      for fname in os.listdir(self.caption_dir)
                      if fname.endswith('.txt')])
        if max_items is not None:
            ids = ids[:max_items]

        if use_full_split:
            self.ids = ids
        else:
            random.seed(seed)
            random.shuffle(ids)
            # chia 80/20
            split_idx = int(len(ids) * split_ratio)
            if split == 'train':
                self.ids = ids[:split_idx]
            else:
                self.ids = ids[split_idx:]

        # load CLIP preprocess (model không dùng, chỉ lấy transform)
        self.device     = torch.device(device)
        _, self.preprocess = clip.load(model_name, device=self.device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        # load caption
        with open(os.path.join(self.caption_dir, f'{uid}.txt'),
                  'r', encoding='utf8') as f:
            caption = f.read().strip().replace('\n', ' ')

        # Tách câu đầu tiên
        sentences = [s.strip() for s in caption.split('.') if s.strip()]
        detail_caption = sentences[0] + '.' if sentences else caption

        # load & preprocess image
        img_path = os.path.join(self.image_dir, f'{uid}.jpg')
        image    = Image.open(img_path).convert('RGB')
        image_t  = self.preprocess(image)

        return image_t, caption, detail_caption, img_path, "None"

DEPLOY_EOF_MARKER

echo '--- writing datasets_config/docci.py ---'
mkdir -p $(dirname "datasets_config/docci.py")
cat > "datasets_config/docci.py" << 'DEPLOY_EOF_MARKER'
# import json
# import cv2
# from PIL import Image
# import clip

# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import random

# data4v_root = '/home/tachau/docci_data/'
# json_name = 'captioner_docci.json'
# image_root = '/home/tachau/docci_data/'

# class share4v_val_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.total_len = 1000
#         with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
#             self.json_data = json.load(fp)[:self.total_len]
#         _ , self.preprocess = clip.load("ViT-L/14")
#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption


# class share4v_train_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.total_len = 1000
#         with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
#             self.json_data = json.load(fp)[self.total_len:]
#         _ , self.preprocess = clip.load("ViT-L/14")

#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
        

#         # caption_short = caption.split(". ")[0]
#         caption_short = self.json_data[index]['conversations'][0]['value']
#         caption_short = caption_short.replace("\n", " ")
        
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption, caption_short, image_name

import json
import os
from PIL import Image
import clip
import torch.utils.data as data

# H100 server paths (image field in json = "docci_images/images/xxx.jpg" -> image_root+field OK)
# DOCCI_DATA_ROOT env override lets this run off-server (e.g. local machine) without touching
# the server default -- unset means the server path below is used unchanged.
data4v_root = os.environ.get('DOCCI_DATA_ROOT', '/cm/archive/luongtk/docci/')
json_path   = os.path.join(data4v_root, 'captioner_docci.json')
image_root  = data4v_root

class DocciDataset(data.Dataset):
    def __init__(self, split: str, max_items: int = None, model_name="ViT-B/16"):
        """
        split: one of 'train', 'qual_train', 'test', 'qual_test'
          - 'train'       => chỉ load split == 'train'
          - 'qual_train'  => chỉ load split == 'qual_train'
          - 'test'        => load cả split == 'test' và 'qual_test'
          - 'qual_test'   => chỉ load split == 'qual_test'
        max_items: nếu muốn giới hạn số mẫu
        """
        print(f"json path:{json_path}")
        valid_splits = ('train', 'qual_train', 'test', 'qual_test')
        if split not in valid_splits:
            raise ValueError(f"Unsupported split: {split}. Use one of {valid_splits}.")

        # Xác định danh sách splits sẽ lấy
        if split == 'train':
            target_splits = ['train']
        elif split == 'qual_train':
            target_splits = ['qual_train']
        elif split == 'test':
            target_splits = ['test', 'qual_test']
        else:  # split == 'qual_test'
            target_splits = ['qual_test']

        # 1) Load toàn bộ JSON
        with open(json_path, 'r', encoding='utf8') as fp:
            full_data = json.load(fp)

        # 2) Filter theo target_splits
        self.samples = [d for d in full_data if d.get('split') in target_splits]

        # 3) Nếu cần giới hạn số mẫu
        if max_items is not None:
            self.samples = self.samples[:max_items]

        # 4) Load CLIP preprocess once
        _ , self.preprocess = clip.load(model_name)
        # _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # caption đầy đủ
        caption = item['conversations'][1]['value'].replace("\n", " ")
        # caption ngắn
        caption_short = caption.split('.')[0].strip() + '.'

        img_path = os.path.join(image_root, item['image'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        # Trả về luôn item['split'] ban đầu để biết nó thuộc loại nào
        return image_tensor, caption, caption_short, img_path, item['split']

# import json
# import cv2
# from PIL import Image
# import clip

# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import random

# # Định nghĩa đường dẫn
# data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'
# json_name = 'captioner.json'
# image_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'

# class share4v_val_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.sample_size = 100
        
#         with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
#             self.json_data = json.load(fp)[:100000]
        
#         # Chọn ngẫu nhiên 100 mẫu
#         self.json_data = random.sample(self.json_data, self.sample_size)
        
#         _, self.preprocess = clip.load("ViT-L/14")
#         # _, self.preprocess = clip.load("ViT-B/16")
    
#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption


# class share4v_train_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.sample_size = 100 
        
#         with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
#             self.json_data = json.load(fp)[100000:]
        
#         # Chọn ngẫu nhiên 100 mẫu
#         self.json_data = random.sample(self.json_data, self.sample_size)
        
#         _, self.preprocess = clip.load("ViT-L/14")
#         # _, self.preprocess = clip.load("ViT-B/16")
    
#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
        
#         caption_short = self.json_data[index]['conversations'][0]['value']
#         caption_short = caption_short.replace("\n", " ")
        
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption, caption_short
DEPLOY_EOF_MARKER

echo '--- writing paper_reference_numbers.json ---'
mkdir -p $(dirname "paper_reference_numbers.json")
cat > "paper_reference_numbers.json" << 'DEPLOY_EOF_MARKER'
{
  "_source": "arXiv:2411.04997 Table 2, ViT-L/14@336px section (verified via WebFetch)",
  "clip_l14_336_baseline": {
    "flickr30k": {"I2T_R1": 0.877, "T2I_R1": 0.670},
    "coco":      {"I2T_R1": 0.580, "T2I_R1": 0.371},
    "sg4v1k":    {"I2T_R1": 0.862, "T2I_R1": 0.840},
    "urban1k":   {"I2T_R1": 0.728, "T2I_R1": 0.570},
    "docci":     {"I2T_R1": 0.674, "T2I_R1": 0.657}
  },
  "llm2clip_15m_336": {
    "_caveat": "Trained on DreamLIP-recaptioned CC3M+CC12M (~15M pairs) at 336px -- more data than our runs use. No published 336px+CC3M-only (3M) row exists in the paper (3M is only reported at 224px). Cited here as the closest available reference, not a controlled baseline.",
    "flickr30k": {"I2T_R1": 0.912, "T2I_R1": 0.821},
    "coco":      {"I2T_R1": 0.655, "T2I_R1": 0.536},
    "sg4v1k":    {"I2T_R1": 0.981, "T2I_R1": 0.984},
    "urban1k":   {"I2T_R1": 0.903, "T2I_R1": 0.932},
    "docci":     {"I2T_R1": 0.877, "T2I_R1": 0.890}
  }
}
DEPLOY_EOF_MARKER

echo 'DONE: all 7 files written'
