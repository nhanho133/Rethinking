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


def _flickr30k_local_1k_images(luongtk_flickr_root, seed=42, n=1000):
    """Deterministic 1000-image subset from a plain image,caption CSV with NO split column
    (e.g. /cm/archive/luongtk/flickr/captions.txt) -- NOT the paper's Karpathy 1K test split,
    just a reproducible proxy. Returns the sorted+seeded-sampled list of image filenames."""
    import csv
    import random
    csv_path = os.path.join(luongtk_flickr_root, "captions.txt")
    images = set()
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            images.add(row["image"])
    images = sorted(images)
    return random.Random(seed).sample(images, min(n, len(images)))


def load_flickr30k_local_eval_captions(luongtk_flickr_root, seed=42, n=1000):
    """ALTERNATE source: /cm/archive/luongtk/flickr/ (captions.txt + Images/), already present
    on this server, no download needed. CAVEAT: no Karpathy split info -- this is a seeded
    1000-image proxy subset (not the paper's exact 1K test list), same caveat class as the
    sg4v1k 'manifest' proxy source."""
    import csv
    selected = set(_flickr30k_local_1k_images(luongtk_flickr_root, seed, n))
    csv_path = os.path.join(luongtk_flickr_root, "captions.txt")
    out = []
    with open(csv_path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["image"] in selected:
                out.append(row["caption"].strip())
    print(f"[collect] Flickr30K (local luongtk proxy, seed={seed}): {len(selected)} images, {len(out)} captions")
    return out


def load_coco_eval_captions(eval_data_root):
    """COCO val2017, ALL captions per image (paper protocol: 5K images x ~5 captions)."""
    ann_path = os.path.join(eval_data_root, "coco", "annotations", "captions_val2017.json")
    data = json.load(open(ann_path, encoding="utf8"))
    out = [ann["caption"].strip() for ann in data["annotations"]]
    print(f"[collect] COCO val2017: {len(out)} captions")
    return out


def load_urban1k_eval_captions(eval_data_root, urban1k_root=None):
    """1000 images, 1 caption/image (long, dense -- like DOCCI/SG4V, not short alt-text).
    urban1k_root: direct override to a root already containing caption/+image/ (e.g. an
    existing /cm/archive/luongtk/Urban1k/ -- same format, skips the download step entirely)."""
    cap_dir = os.path.join(urban1k_root, "caption") if urban1k_root else \
        os.path.join(eval_data_root, "urban1k", "Urban1k", "caption")
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


def collect_eval_strings(dataset, eval_data_root, sg4v_source="full_json", sg4v_manifest_json=None,
                          urban1k_root=None, flickr_source="hf", flickr_luongtk_root=None):
    loader = {
        "flickr30k": lambda: (load_flickr30k_local_eval_captions(flickr_luongtk_root)
                              if flickr_source == "luongtk_local"
                              else load_flickr30k_eval_captions(eval_data_root)),
        "coco": lambda: load_coco_eval_captions(eval_data_root),
        "urban1k": lambda: load_urban1k_eval_captions(eval_data_root, urban1k_root),
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
    ap.add_argument("--urban1k_root", type=str, default=None,
                    help="Direct override to an existing caption/+image/ root "
                         "(e.g. /cm/archive/luongtk/Urban1k) -- skips downloading.")
    ap.add_argument("--flickr_source", choices=["hf", "luongtk_local"], default="hf",
                    help="'hf' = flickr_annotations_30k.csv with Karpathy split (paper-accurate). "
                         "'luongtk_local' = plain image,caption CSV with NO split info -- "
                         "seeded 1000-image proxy, not the exact paper test set.")
    ap.add_argument("--flickr_luongtk_root", type=str, default=None,
                    help="Root containing captions.txt + Images/ (for --flickr_source luongtk_local).")
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
                                              sg4v_manifest_json=args.sg4v_manifest_json,
                                              urban1k_root=args.urban1k_root,
                                              flickr_source=args.flickr_source,
                                              flickr_luongtk_root=args.flickr_luongtk_root),
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
