"""
Build a subset manifest for ShareGPT4V from the FULL original json already on the H100 server
(/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json, 1.246M items).
The full set is too big to precompute Llama embeddings for; take a shuffled subset of images
that actually exist under data/, and write a small manifest (image, caption, caption_short, split)
that sharegpt4v_coco.py + precompute_llm2vec_embeddings.py read.

Run once on the server:
  python make_sharegpt4v_subset.py --n 120000
"""
import argparse
import json
import os
import random

FULL_JSON = "/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
IMAGE_ROOT = "/cm/archive/luongtk/sharegpt4v/data/"
OUT = "/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v_subset_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_json", default=FULL_JSON)
    ap.add_argument("--image_root", default=IMAGE_ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=120000, help="số ảnh muốn lấy (đủ lớn để không học thuộc)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="lọc theo nguồn ảnh, vd: coco sam llava. None = tất cả")
    ap.add_argument("--test_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[load] reading full json {args.full_json} (~1.5GB, chờ chút)...", flush=True)
    data = json.load(open(args.full_json, encoding="utf8"))
    print(f"[load] {len(data)} items", flush=True)

    if args.sources:
        data = [d for d in data if d["image"].split("/")[0] in set(args.sources)]
        print(f"[filter] {len(data)} items after source filter {args.sources}", flush=True)

    random.Random(args.seed).shuffle(data)

    results = []
    checked = 0
    for d in data:
        checked += 1
        img_rel = d["image"]
        if not os.path.isfile(os.path.join(args.image_root, img_rel)):
            continue  # skip missing images (e.g. a source dir not present)
        caption = d["conversations"][1]["value"].replace("\n", " ")
        caption_short = caption.split(".")[0].strip() + "."
        results.append({"image": img_rel, "caption": caption, "caption_short": caption_short})
        if len(results) >= args.n:
            break
        if checked % 20000 == 0:
            print(f"  [scan] checked={checked} kept={len(results)}", flush=True)

    random.Random(args.seed + 1).shuffle(results)
    n_test = int(len(results) * args.test_frac)
    for i, r in enumerate(results):
        r["split"] = "test" if i < n_test else "train"

    json.dump(results, open(args.out, "w"))
    print(f"[done] wrote {len(results)} items -> {args.out} "
          f"(train={len(results)-n_test}, test={n_test})", flush=True)
    from collections import Counter
    print("[sources]", Counter(r["image"].split("/")[0] for r in results))


if __name__ == "__main__":
    main()
