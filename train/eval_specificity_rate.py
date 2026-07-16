"""
Specificity Rate (SR), following SPECS (arXiv 2509.03897): for held-out (image, base_caption,
extended_caption) triples where extended_caption = base_caption + one more correct sentence
from the same long caption, SR = fraction of triples where
sim(image, extended) > sim(image, base). A trained loss with the SPECS-style hinge should push
SR above what a vanilla (ClipLoss-only) model achieves, since only `ours` explicitly optimizes
image-longer-caption similarity to exceed image-base-caption similarity.

Cheap and needs no new downloads/training: reuses the DOCCI/CC3M test-split captions and the
already-trained checkpoints + text embedding caches.
"""
import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(".")
sys.path.append("..")
from model.model_llm2clip import LLM2CLIPTextTeacher, TextEmbeddingCache
from precompute_llm2vec_embeddings import split_into_detail_captions
from sampling import star_bar_long_text_split
import clip

MAXK = 4
SEED = 42  # must match the seed baked into precompute's cache coverage


def build_triples(caption, rng, n_pairs=2):
    """base = one star-bar chunk (cached verbatim as a `pos` string), extended = that same
    chunk joined with a later chunk, in the EXACT format make_base_longer produces
    (`chunks[a].strip() + " " + chunks[b].strip()`, a<b) -- guaranteed to be a cache hit
    since precompute enumerated all such combinations."""
    sents = split_into_detail_captions(caption)
    if len(sents) < MAXK:
        return []
    chunks = list(star_bar_long_text_split(sents, MAXK, SEED))
    if len(chunks) < 2:
        return []
    pairs = [(a, b) for a in range(len(chunks)) for b in range(len(chunks)) if a < b]
    rng.shuffle(pairs)
    triples = []
    for a, b in pairs[:n_pairs]:
        base = chunks[a].strip()
        extended = (chunks[a].strip() + " " + chunks[b].strip()).strip()
        triples.append((base, extended))
    return triples


def _remap_legacy_adapter_keys(sd):
    """Checkpoints saved before the linear/mlp TextAdapter refactor used
    text_adapter.ln.* / text_adapter.proj.* directly; the current 'linear' kind wraps the
    same two layers in a Sequential named text_adaptor.0 / text_adaptor.1."""
    if "text_adapter.ln.weight" not in sd:
        return sd
    remap = {
        "text_adapter.ln.weight": "text_adapter.text_adaptor.0.weight",
        "text_adapter.ln.bias": "text_adapter.text_adaptor.0.bias",
        "text_adapter.proj.weight": "text_adapter.text_adaptor.1.weight",
        "text_adapter.proj.bias": "text_adapter.text_adaptor.1.bias",
    }
    return {remap.get(k, k): v for k, v in sd.items()}


def load_model(ckpt_path, device):
    model = LLM2CLIPTextTeacher(clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                                freeze_visual=False, device="cpu", adapter_type="linear")
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = _remap_legacy_adapter_keys(sd)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def specificity_rate(model, cache, images, captions, device, n_pairs=2, batch_size=64, seed=0):
    rng = random.Random(seed)
    _, preprocess = clip.load("ViT-B/16", device="cpu")

    # image features
    im_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = model.encode_image(tensors)
        f = f / f.norm(dim=-1, keepdim=True)
        im_feats.append(f)
    im_feats = torch.cat(im_feats, 0)  # [N, D]

    total, correct = 0, 0
    per_image_triples = [build_triples(c, rng, n_pairs=n_pairs) for c in captions]
    for idx, triples in enumerate(per_image_triples):
        if not triples:
            continue
        bases = [t[0] for t in triples]
        exts = [t[1] for t in triples]
        try:
            b_emb = cache.lookup(bases)
            e_emb = cache.lookup(exts)
        except KeyError:
            continue  # not every synthetic prefix is guaranteed cached; skip
        b_feat = model.encode_text(b_emb); b_feat = b_feat / b_feat.norm(dim=-1, keepdim=True)
        e_feat = model.encode_text(e_emb); e_feat = e_feat / e_feat.norm(dim=-1, keepdim=True)
        v = im_feats[idx:idx + 1]  # [1, D]
        sim_base = (v @ b_feat.t()).squeeze(0)
        sim_ext = (v @ e_feat.t()).squeeze(0)
        correct += (sim_ext > sim_base).sum().item()
        total += len(triples)
    return correct / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docci", "dreamlip_cc3m"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n_pairs", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "docci":
        data = json.load(open("/cm/archive/luongtk/docci/captioner_docci.json", encoding="utf8"))
        items = [d for d in data if d.get("split") in ("test", "qual_test")]
        images = ["/cm/archive/luongtk/docci/" + d["image"] for d in items]
        captions = [d["conversations"][1]["value"].replace("\n", " ") for d in items]
    else:
        data = json.load(open("/cm/shared/chautvh_second/Nhan_folder/work/cc3m/manifest.json", encoding="utf8"))
        items = [d for d in data if d["split"] == "test"]
        images = ["/cm/shared/chautvh_second/Nhan_folder/work/cc3m/images/" + d["image"] for d in items]
        captions = [d["caption"].replace("\n", " ") for d in items]

    print(f"[data] {len(images)} test images from {args.dataset}")

    model = load_model(args.ckpt, device)
    cache = TextEmbeddingCache(args.cache, device=device)

    sr, n = specificity_rate(model, cache, images, captions, device, n_pairs=args.n_pairs)
    print(f"[RESULT] ckpt={args.ckpt}")
    print(f"[RESULT] Specificity Rate = {sr:.4%} over {n} (base,extended) triples")


if __name__ == "__main__":
    main()
