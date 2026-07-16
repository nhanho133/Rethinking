"""
Download a subset of ShareGPT4V's COCO-train2017-sourced captions. Images come from COCO's
official, stable host (images.cocodataset.org) -- unlike DreamLIP-CC3M's web-URL images, this
should have close to 100% success rate. Captions are the same ShareCaptioner model DreamLIP
used for its "SV" columns (see earlier finding), applied to COCO's more repetitive/curated
photo domain instead of noisy web images -- intended to avoid the ceiling effect seen on the
CC3M subset's overly diverse images.
"""
import argparse
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-dataset-fetch/1.0)"}
COCO_URL_TMPL = "http://images.cocodataset.org/train2017/{fname}"


def try_download(idx, fname, caption, out_dir, timeout=10):
    try:
        url = COCO_URL_TMPL.format(fname=fname)
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out_fname = f"{idx:08d}.jpg"
        img.save(os.path.join(out_dir, out_fname), "JPEG", quality=90)
        caption_short = caption.split(".")[0].strip() + "."
        return {"image": out_fname, "caption": caption, "caption_short": caption_short}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json")
    ap.add_argument("--out_dir", default="/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v/images")
    ap.add_argument("--manifest", default="/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v/manifest.json")
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = json.load(open(args.json, encoding="utf8"))
    coco_items = [d for d in data if d["image"].startswith("coco/")]
    print(f"[pool] {len(coco_items)} COCO-sourced candidate rows")
    random.Random(0).shuffle(coco_items)

    results = []
    attempted = 0
    idx = 0
    t0 = time.time()
    pos = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while len(results) < args.target and pos < len(coco_items):
            batch = coco_items[pos: pos + args.batch_size]
            pos += args.batch_size
            futs = []
            for item in batch:
                fname = os.path.basename(item["image"])
                caption = item["conversations"][1]["value"].replace("\n", " ")
                futs.append(ex.submit(try_download, idx, fname, caption, args.out_dir))
                idx += 1
            for fut in futs:
                try:
                    r = fut.result(timeout=20)
                except Exception:
                    r = None
                attempted += 1
                if r:
                    results.append(r)
            elapsed = time.time() - t0
            print(f"[progress] attempted={attempted} ok={len(results)} "
                  f"({len(results)/max(attempted,1):.1%} hit rate) elapsed={elapsed:.0f}s", flush=True)

    print(f"[done] {len(results)} images downloaded from {attempted} attempts "
          f"({len(results)/max(attempted,1):.1%} hit rate) in {time.time()-t0:.0f}s", flush=True)

    random.Random(42).shuffle(results)
    n_test = int(len(results) * args.test_frac)
    for i, r in enumerate(results):
        r["split"] = "test" if i < n_test else "train"

    with open(args.manifest, "w", encoding="utf8") as f:
        json.dump(results, f)
    print(f"[manifest] wrote {len(results)} items -> {args.manifest} "
          f"(train={len(results)-n_test}, test={n_test})", flush=True)


if __name__ == "__main__":
    main()
