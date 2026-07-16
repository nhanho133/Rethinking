"""
Zero-shot ImageNet(-V2) classification, following the standard CLIP protocol: build a
classifier weight per class by averaging the (adapter-projected) Llama embeddings of prompt
templates ("a photo of a {class}"), then classify each image by argmax cosine similarity.

Uses ImageNet-V2 matched-frequency (10k images, ungated) since ImageNet-1k val is gated.
Absolute numbers are lower than ImageNet-val (V2 is a distribution-shifted variant), but the
RELATIVE comparison between checkpoints (vanilla vs ours) is what matters here. Uses a 7-template
subset (not the full 80) to keep the one-time Llama prompt-embedding fast.

Important caveat baked into the interpretation: our checkpoints are fine-tuned 50 epochs on 15k
DOCCI long-caption images, which likely wrecks the base CLIP's general classification ability
for BOTH vanilla and ours -- so low numbers mostly measure fine-tuning damage, not the loss.
"""
import argparse
import hashlib
import os
import sys

import torch
from PIL import Image

sys.path.append(".")
sys.path.append("..")
sys.path.append("eval_all")
from model.model_llm2clip import LLM2CLIPTextTeacher, TextEmbeddingCache
from precompute_llm2vec_embeddings import load_l2v
from classes import imagenet_classes
import clip

TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a cropped photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a close-up photo of a {}.",
]
CACHE_PATH = "/cm/shared/chautvh_second/Nhan_folder/work/eval_data/imagenetv2/prompt_llm2vec_cache.pt"


def all_prompts():
    prompts = []
    for c in imagenet_classes:
        for t in TEMPLATES:
            prompts.append(t.format(c))
    return prompts


def ensure_prompt_cache():
    if os.path.exists(CACHE_PATH):
        print(f"[cache] found {CACHE_PATH}")
        return
    prompts = sorted(set(all_prompts()), key=len)
    print(f"[cache] embedding {len(prompts)} prompts via Llama-CC ...")
    l2v = load_l2v(quant_4bit=True)
    cache = {}
    bs = 64
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        embs = l2v.encode(batch, batch_size=len(batch), show_progress_bar=False, convert_to_tensor=True)
        embs = embs.to(torch.float16).cpu()
        for text, emb in zip(batch, embs):
            if not torch.isfinite(emb).all():
                emb = torch.zeros_like(emb)
            cache[hashlib.sha1(text.encode("utf8")).hexdigest()] = emb
        if i % (bs * 20) == 0:
            print(f"  [embed] {i}/{len(prompts)}", flush=True)
    torch.save({"cache": cache, "dim": 4096}, CACHE_PATH)
    print(f"[cache] wrote {len(cache)}")
    del l2v
    torch.cuda.empty_cache()


def _remap_legacy(sd):
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
def build_classifier(model, cache, device):
    """[1000, D] normalized classifier weights: per class, mean of adapter(prompt_emb) over templates."""
    weights = []
    for c in imagenet_classes:
        prompts = [t.format(c) for t in TEMPLATES]
        emb = cache.lookup(prompts)                 # [T, 4096]
        f = model.encode_text(emb)                  # [T, D]
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        f = f / f.norm()
        weights.append(f)
    return torch.stack(weights, 0)                  # [1000, D]


def find_image_dirs(root):
    """ImageNetV2 extracts to a folder with 1000 subdirs named 0..999 (class index)."""
    for dirpath, dirnames, _ in os.walk(root):
        if sum(d.isdigit() for d in dirnames) >= 900:
            return dirpath
    raise RuntimeError(f"could not locate the 1000 class subdirs under {root}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="/cm/shared/chautvh_second/Nhan_folder/work/eval_data/imagenetv2")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_prompt_cache()

    model = LLM2CLIPTextTeacher(clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                                freeze_visual=False, device="cpu", adapter_type="linear")
    model.load_state_dict(_remap_legacy(torch.load(args.ckpt, map_location="cpu")))
    model = model.to(device).eval()
    cache = TextEmbeddingCache(CACHE_PATH, device=device)
    _, preprocess = clip.load("ViT-B/16", device="cpu")

    classifier = build_classifier(model, cache, device)  # [1000, D]

    img_dir = find_image_dirs(args.root)
    print(f"[data] class dirs under {img_dir}")
    samples = []
    for cls in sorted(os.listdir(img_dir), key=lambda x: int(x) if x.isdigit() else 1e9):
        if not cls.isdigit():
            continue
        cdir = os.path.join(img_dir, cls)
        for fn in os.listdir(cdir):
            samples.append((os.path.join(cdir, fn), int(cls)))
    print(f"[data] {len(samples)} images")

    top1 = top5 = 0
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        labels = torch.tensor([lab for _, lab in batch], device=device)
        v = model.encode_image(imgs); v = v / v.norm(dim=-1, keepdim=True)
        logits = v @ classifier.t()                  # [B, 1000]
        top5_idx = logits.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == labels).sum().item()
        top5 += (top5_idx == labels.unsqueeze(1)).any(dim=1).sum().item()
        if i % (args.batch_size * 20) == 0:
            print(f"  [eval] {i}/{len(samples)}", flush=True)
    n = len(samples)
    print(f"[RESULT] ckpt={args.ckpt}")
    print(f"[RESULT] ImageNetV2 zero-shot: top1={top1/n:.4%} top5={top5/n:.4%} (n={n})")


if __name__ == "__main__":
    main()
