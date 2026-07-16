"""
Download a subset of DreamLIP's re-captioned CC3M (the actual dataset LLM2CLIP's official
stage-2 training consumes -- see llm2clip/data/download_dataset.sh) by fetching images from
their original (often dead, ~5-8 years old) web URLs. Unlike img2dataset (not installed, adds
webdataset/deps complexity), this is a minimal threaded downloader tailored to exactly what we
need: process candidate rows in fixed-size batches (simple, no dynamic resubmission -> no
mutate-while-iterating hazards), stop once `--target` successful downloads are reached, verify
each image decodes, save a manifest JSON compatible with the DocciDataset-style interface
train.py expects (image, caption, caption_short, img_path, split).
"""
import argparse
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-dataset-fetch/1.0)"}


def try_download(idx, url, caption, caption_short, out_dir, timeout=5):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=False)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 64:
            return None
        fname = f"{idx:08d}.jpg"
        img.save(os.path.join(out_dir, fname), "JPEG", quality=90)
        return {"image": fname, "caption": caption, "caption_short": caption_short}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/cm/shared/chautvh_second/Nhan_folder/work/cc3m/raw/cc3m_chunk.parquet")
    ap.add_argument("--out_dir", default="/cm/shared/chautvh_second/Nhan_folder/work/cc3m/images")
    ap.add_argument("--manifest", default="/cm/shared/chautvh_second/Nhan_folder/work/cc3m/manifest.json")
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--caption_col", default="longSV_captions")
    ap.add_argument("--short_col", default="shortSV_captions")
    ap.add_argument("--min_caption_len", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    df = df[df[args.caption_col].str.len() >= args.min_caption_len]
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    print(f"[pool] {len(df)} candidate rows (len-filtered) available", flush=True)

    results = []
    attempted = 0
    t0 = time.time()
    idx = 0
    pos = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while len(results) < args.target and pos < len(df):
            batch = df.iloc[pos: pos + args.batch_size]
            pos += args.batch_size
            futs = []
            for row in batch.itertuples():
                url = getattr(row, "Image_Path", None) or row._1
                cap = getattr(row, args.caption_col)
                cap_short = getattr(row, args.short_col)
                futs.append(ex.submit(try_download, idx, url, cap, cap_short, args.out_dir))
                idx += 1
            for fut in futs:
                try:
                    r = fut.result(timeout=15)
                except Exception:
                    r = None
                attempted += 1
                if r:
                    results.append(r)
            elapsed = time.time() - t0
            print(f"[progress] attempted={attempted} ok={len(results)} "
                  f"({len(results)/max(attempted,1):.1%} hit rate) elapsed={elapsed:.0f}s",
                  flush=True)

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
