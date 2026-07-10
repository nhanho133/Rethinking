"""
Offline, one-time precompute of LLM2Vec (frozen Llama-3-8B-CC) text embeddings for every
caption/sub-caption string DOCCI training could ever need, cached to disk keyed by
sha1(text). This lets train.py run the LLM2CLIP-text-teacher branch without ever loading
the 8B model during actual training (which would not fit in 12GB VRAM alongside a
trainable visual tower).

Coverage strategy (see plan doc for derivation): star_bar_long_text_split and
make_base_longer's only randomness is which existing deterministic candidate gets picked
at train time -- the *candidate pools* themselves are fully enumerable from
(sentences, max_num_short_texts, fixed seed=42). So instead of sampling many simulated
epochs, we deterministically enumerate every string that could ever be produced.
"""
import argparse
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
    covering both the main path (len(sentences) >= K) and train.py's fallback padding path
    (len(sentences) < K, which draws extra chunks via a randint(1, len(sentences)) that we
    enumerate exhaustively here instead of leaving to chance)."""
    n = len(sentences)
    if n == 0:
        return []
    if n >= max_num_short_texts:
        return list(star_bar_long_text_split(sentences, max_num_short_texts, SEED))
    # fallback path: base = full 1-sentence-per-chunk split, plus every possible partial
    # partition train.py's while-loop could have drawn from (new_star in 1..n)
    pool = list(star_bar_long_text_split(sentences, n, SEED))
    for new_star in range(1, n + 1):
        pool.extend(star_bar_long_text_split(sentences, new_star, SEED))
    # dedup while preserving order
    seen = set()
    out = []
    for c in pool:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def enumerate_longer_pool(pos_pool, full=False):
    """make_base_longer joins two chunks (sorted by their index in the actual chunks list)
    per slot. Two cases:
      - n >= K (common): the chunks list is the deterministic, distinct, position-ordered
        star_bar(sentences, K), so make_base_longer only ever emits chunks[a]+" "+chunks[b]
        with a<b -> unordered combinations exactly reproduce it.
      - n <  K (rare, short captions): train.py's fallback builds the chunks list with global
        randomness whose order/multiplicity varies, so a joined pair can appear in either
        order and a chunk can pair with an equal-string duplicate ("A A"). To be miss-proof,
        enumerate the full Cartesian product (both orders + self-pairs) over the pool."""
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
        all_strings.add(caption)       # text_long (train L_long + eval full-text retrieval)
        all_strings.add(caption_short)  # fallback used when chunks ends up empty

        sentences = split_into_detail_captions(caption)
        # eval (test_epoch_ver5) encodes the raw individual detail sentences, not the
        # star-bar-joined chunks -> cache each one so retrieval eval has coverage too.
        all_strings.update(sentences)

        pos_pool = enumerate_pos_pool(sentences, max_num_short_texts)
        if not pos_pool:
            continue
        all_strings.update(pos_pool)
        # n<K triggers train.py's random fallback -> use the miss-proof full enumeration.
        is_fallback = len(sentences) < max_num_short_texts
        all_strings.update(enumerate_longer_pool(pos_pool, full=is_fallback))

    print(f"[collect] {len(all_strings)} unique strings need embeddings")
    return all_strings


def _patch_bnb_to():
    """transformers 4.44.2 pairs badly with accelerate 1.14.0: for a 4-bit model that fully
    fits on one GPU, dispatch_model calls model.to(device), which transformers' PreTrainedModel.to
    hard-raises for bitsandbytes models. But the model is *already* correctly placed, so a
    device-only .to() is a safe no-op. Patch it so device-only moves return self (dtype changes
    still raise, as intended). This also neutralizes LLM2Vec.encode()'s internal self.to(device)."""
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

    # Pin everything to GPU 0. device_map="auto" triggers accelerate's dispatch_model ->
    # model.to(device), which raises for 4-bit bitsandbytes models; device_map={"":0} avoids
    # that dispatch path and still places the whole model on cuda:0.
    llm_model = AutoModel.from_pretrained(
        LLM_PATH, config=config, trust_remote_code=True, device_map={"": 0}, **quant_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    # Workaround required by LLM2Vec, per the model card.
    llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
    return l2v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="docci",
                    choices=["docci", "dreamlip_cc3m", "sharegpt4v_coco"])
    ap.add_argument("--max_num_short_texts", type=int, default=4)
    ap.add_argument("--out", type=str, default="/cm/shared/chautvh_second/Nhan_folder/work/docci_cache.pt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_4bit", action="store_true", help="Load in bf16 instead of 4-bit (needs more VRAM).")
    ap.add_argument("--dry_run_n", type=int, default=None,
                     help="If set, only embed this many strings (smoke test before the full run).")
    args = ap.parse_args()

    # Sort by length so each fixed-size batch is length-homogeneous -> minimal padding waste.
    # Most strings are short (individual sentences / small chunks); only full captions and
    # longer-combos are long, so this makes the bulk of the run fast.
    strings = sorted(collect_all_strings(args.max_num_short_texts, dataset=args.dataset), key=len)
    if args.dry_run_n:
        strings = strings[: args.dry_run_n]
        print(f"[dry-run] truncated to {len(strings)} strings")

    print("[load] loading LLM2Vec (Llama-3-8B-CC)"
          + (" [4-bit]" if not args.no_4bit else " [bf16]") + " ...")
    l2v = load_l2v(quant_4bit=not args.no_4bit)
    print("[load] model loaded, VRAM:",
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated" if torch.cuda.is_available() else "cpu")

    cache = {}
    n = len(strings)
    for i in range(0, n, args.batch_size):
        batch = strings[i : i + args.batch_size]
        with torch.no_grad():
            embs = l2v.encode(batch, convert_to_tensor=True)
        embs = embs.to(torch.float16).cpu()
        for text, emb in zip(batch, embs):
            if not torch.isfinite(emb).all():
                # e.g. the empty-string pad slot can pool to NaN; store a deterministic zero
                # vector so downstream projection+normalize stays finite.
                emb = torch.zeros_like(emb)
            cache[hashlib.sha1(text.encode("utf8")).hexdigest()] = emb
        if (i // args.batch_size) % 20 == 0:
            print(f"[embed] {min(i+args.batch_size, n)}/{n}")

    torch.save({"cache": cache, "dim": next(iter(cache.values())).shape[0]}, args.out)
    print(f"[done] wrote {len(cache)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
