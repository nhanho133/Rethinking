#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_all.py — Orchestrate ALL evals (Flickr30k, COCO, Urban1k, ShareGPT4V, ImageNet-O, ImageNet-S, ImageNet-V2)
Run examples:
  - Single checkpoint:
      python eval_all.py --ckpt /path/to/B16-longclip-*.pt --device cuda:0
  - A folder of checkpoints:
      python eval_all.py --ckpt_dir /path/to/ckpt_dir --pattern "*.pt"
  - Override data locations (optional):
      python eval_all.py --flickr_csv /data/flickr30k/flickr_annotations_30k.csv --flickr_root /data/flickr30k
                         --coco_root /data/coco2017 --urban_root /data/Urban1k
                         --imageneto_dir /data/imagenet-o --imagenets_dir /data/ImageNetS919/validation
                         --imagenetv2_dir /data/imagenetv2-top-images-format-val
Outputs:
  - Prints a compact summary to stdout
  - Optionally writes JSON/CSV/XLSX tables via --out_json/--out_csv/--out_xlsx
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# ────────────────────────────────────────────────────────────────────────────────
# Local imports (expect this file to be placed alongside the eval modules)
# ────────────────────────────────────────────────────────────────────────────────
from coco import run_coco_eval
from flickr import run_flickr30k_eval_allcaps
from urban1kEval import run_urban1k_eval
# import pdb
# pdb.set_trace()
from sharegpt4vEval import run_share4v_eval
from imageneto import run_zeroshot as run_imagenet_o
from imagetnetSub import evaluate_zeroshot as run_imagenet_s
from imagenetV2 import run_zeroshot_imagenetv2 as run_imagenet_v2
from docci import run_docci_eval
from dci import run_dci_eval
from art import run_artpedia_eval
from openEvenv1 import run_openevent_eval
# If LongCLIP is in a repo two levels up (as in user's snippet)
sys.path.append("../..")
from model import longclip


# ────────────────────────────────────────────────────────────────────────────────
# Loader per user's snippet
# ────────────────────────────────────────────────────────────────────────────────
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

def load_longclip(ckpt_path: str, device: str = DEVICE_DEFAULT):
    ckpt_path = str(Path(ckpt_path).expanduser())
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, preprocess = longclip.load(ckpt_path, device=device)
    model = model.to(device).eval()
    try:
        model.preprocess = preprocess  # convenience
    except Exception:
        pass
    print(f"✅ Loaded LongCLIP from: {ckpt_path}\n🖥️ Device: {device}")
    return model, preprocess


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def list_ckpts(ckpt_dir: str, pattern: str) -> List[Path]:
    p = Path(ckpt_dir)
    files = sorted(p.glob(pattern))
    return [f for f in files if f.is_file()]

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x

def flatten_metrics(prefix: str, metrics: Any) -> Dict[str, Any]:
    """Flatten a nested metrics structure into flat {f"{prefix}_{key}": value}"""
    flat: Dict[str, Any] = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            kk = str(k).replace(" ", "").replace("→", "2").replace("→", "2")
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    name = f"{prefix}_{kk}_R{k2}" if isinstance(k2, (int, float)) else f"{prefix}_{kk}_{k2}"
                    flat[name] = _to_float(v2)
            else:
                flat[f"{prefix}_{kk}"] = _to_float(v)
    else:
        flat[prefix] = _to_float(metrics)
    return flat

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def print_block(title: str, kv: Dict[str, Any]):
    print(f"\n==== {title} ====")
    for k in sorted(kv.keys()):
        v = kv[k]
        if isinstance(v, float):
            # Pretty print % for recalls/acc where value ∈ [0,1]
            if 0.0 <= v <= 1.0 and ("R" in k or "acc" in k or "top1" in k or "Top1" in k):
                print(f"{k:30s} : {v*100:.2f}%")
            else:
                print(f"{k:30s} : {v:.6f}")
        else:
            print(f"{k:30s} : {v}")

def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, e

# ────────────────────────────────────────────────────────────────────────────────
# Runner for all tasks for a given model
# ────────────────────────────────────────────────────────────────────────────────
def run_all_tasks(model, preprocess, args) -> Dict[str, Any]:
    results: Dict[str, Any] = {"_meta": {"start": now_str()}}

    # Flickr30k
    if "flickr" not in args.disable:
        metrics, err = safe_call(
            run_flickr30k_eval_allcaps,
            model,
            preprocess,
            csv_path=args.flickr_csv,
            root_dir=args.flickr_root,
            split=args.flickr_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_items=args.max_items
        )
        if err is None:
            results.update(flatten_metrics("flickr", metrics))
        else:
            results["flickr_error"] = str(err)

    # COCO
    if "coco" not in args.disable:
        metrics, err = safe_call(
            run_coco_eval,
            model,
            preprocess,
            data_root=args.coco_root,
            split=args.coco_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_items=args.max_items,
        )
        if err is None:
            results.update(flatten_metrics("coco", metrics))
        else:
            results["coco_error"] = str(err)

    # Urban1k
    if "urban1k" not in args.disable:
        metrics, err = safe_call(
            run_urban1k_eval,
            model,
            preprocess,
            data_root=args.urban_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_items=args.max_items,
            max_details=args.max_details,
        )
        if err is None:
            results.update(flatten_metrics("urban1k", metrics))
        else:
            results["urban1k_error"] = str(err)

    # Docci
    if "docci" not in args.disable:
        metrics, err = safe_call(
            run_docci_eval,
            model,
            preprocess,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_items=args.max_items,
            max_details=args.max_details,
        )
        if err is None:
            results.update(flatten_metrics("docci", metrics))
        else:
            results["docci_error"] = str(err)

    # # openV1
    # if "openV1" not in args.disable:
    #     root_dir = "/home/ubuntu/shared/OpenEvenv1/train/Train Set"
    #     csv_name = "gt_train.csv"
    #     images_dir = "train_images_compressed90"
    #     metrics, err = safe_call(
    #         run_openevent_eval,
    #         model, preprocess,
    #         root_dir=root_dir,
    #         csv_name=csv_name,
    #         images_dir=images_dir,
    #         split="test",
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         device=args.device,
    #         max_items=args.max_items,
    #         max_details=args.max_details,
    #     )
    #     if err is None:
    #         results.update(flatten_metrics("openV1", metrics))
    #     else:
    #         results["openV1_error"] = str(err)


    # if "art" not in args.disable:
    #     metrics, err = safe_call(
    #         run_artpedia_eval,
    #         model,
    #         preprocess,
    #         split='test',
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         device=args.device,
    #         max_items=args.max_items,
    #         max_details=args.max_details,
    #     )
    #     if err is None:
    #         results.update(flatten_metrics("art", metrics))
    #     else:
    #         results["art_error"] = str(err)

    # Dci
    if "dci" not in args.disable:
        metrics, err = safe_call(
            run_dci_eval,
            model,
            preprocess,
            json_path="/home/ubuntu/hieu.tq/Git/GOAL/datasets/DCI_test.json",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_items=args.max_items,
            max_details=args.max_details,
        )
        if err is None:
            results.update(flatten_metrics("dci", metrics))
        else:
            results["dci_error"] = str(err)

    # ShareGPT4V (dataset loader supplies its own preprocess)
    if "share4v" not in args.disable:
        metrics, err = safe_call(
            run_share4v_eval,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_details=args.max_details,
        )
        if err is None:
            results.update(flatten_metrics("share4v", metrics))
        else:
            results["share4v_error"] = str(err)

    # ImageNet-O
    if "imagenet_o" not in args.disable:
        acc, err = safe_call(
            run_imagenet_o,
            model,
            preprocess,
            data_dir=args.imageneto_dir,
            num_workers=args.num_workers_cls,
            batch_size=args.batch_size_cls,
            device=args.device,
        )
        if err is None:
            results["imagenet-o_top1"] = _to_float(acc)
        else:
            results["imagenet-o_error"] = str(err)

    # ImageNet-S
    if "imagenet_s" not in args.disable:
        acc, err = safe_call(
            run_imagenet_s,
            model,
            preprocess,
            data_dir=args.imagenets_dir,
            num_workers=args.num_workers_cls,
            batch_size=args.batch_size_cls,
            device=args.device,
        )
        if err is None:
            results["imagenet-s_top1"] = _to_float(acc)
        else:
            results["imagenet-s_error"] = str(err)

    # ImageNet-V2
    if "imagenet_v2" not in args.disable:
        acc, err = safe_call(
            run_imagenet_v2,
            model,
            preprocess,
            data_dir=args.imagenetv2_dir,
            num_workers=args.num_workers_cls,
            batch_size=args.batch_size_cls,
            device=args.device,
        )
        if err is None:
            results["imagenet-v2_top1"] = _to_float(acc)
        else:
            results["imagenet-v2_error"] = str(err)

    results["_meta"]["end"] = now_str()
    return results


# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────
def write_outputs(all_rows: List[Dict[str, Any]], args):
    # JSON
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"📝 Saved JSON to: {args.out_json}")

    # CSV
    if args.out_csv:
        # Lazy import pandas only if needed
        import pandas as pd
        df = pd.DataFrame(all_rows)
        df.to_csv(args.out_csv, index=False)
        print(f"🧾 Saved CSV to: {args.out_csv}")

    # XLSX
    if args.out_xlsx:
        try:
            import pandas as pd
            with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
                df = pd.DataFrame(all_rows)
                df.to_excel(writer, index=False, sheet_name="All")
            print(f"📊 Saved XLSX to: {args.out_xlsx}")
        except Exception as e:
            print(f"[warn] Could not write XLSX ({e}). Install openpyxl or check path.")


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run ALL evals for LongCLIP checkpoints.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="Path to a single .pt checkpoint")
    src.add_argument("--ckpt_dir", type=str, help="Directory containing multiple .pt checkpoints")
    p.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern for --ckpt_dir")

    p.add_argument("--device", type=str, default=DEVICE_DEFAULT)

    # Common loader knobs
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--max_details", type=int, default=4)

    # Classification loader knobs (ImageNet*)
    p.add_argument("--batch_size_cls", type=int, default=256)
    p.add_argument("--num_workers_cls", type=int, default=8)

    # Data roots (override as needed; otherwise each module's default is used)
    p.add_argument("--flickr_csv", type=str, default="/home/ubuntu/shared/hieu.tq/flickr30k/flickr_annotations_30k.csv")
    p.add_argument("--flickr_root", type=str, default="/home/ubuntu/shared/hieu.tq/flickr30k")
    p.add_argument("--flickr_split", type=str, default="test")

    p.add_argument("--coco_root", type=str, default="/home/ubuntu/shared/ShareGPT4V/data/coco/images/coco2017")
    p.add_argument("--coco_split", type=str, default="val")

    p.add_argument("--urban_root", type=str, default="/home/ubuntu/shared/hieu.tq/Urban1k/Urban1k")

    p.add_argument("--imageneto_dir", type=str, default="/home/ubuntu/shared/hieu.tq/data/imagenet-o/imagenet-o")
    p.add_argument("--imagenets_dir", type=str, default="/home/ubuntu/shared/hieu.tq/imagenet-s/data/ImageNetS919/validation")
    p.add_argument("--imagenetv2_dir", type=str, default="/home/ubuntu/shared/hieu.tq/data/ImageNetV2/imagenetv2-top-images-format-val")

    # Disable some tasks
    p.add_argument("--disable", nargs="*", default=[], help="Names to skip: flickr coco docci dci urban1k share4v imagenet_o imagenet_s imagenet_v2 art")

    # Outputs
    p.add_argument("--out_json", type=str, default="eval_all_results.json")
    p.add_argument("--out_csv", type=str, default="eval_all_results.csv")
    p.add_argument("--out_xlsx", type=str, default=None)

    return p.parse_args()

def main():
    args = parse_args()

    ckpt_list: List[Path] = []
    if args.ckpt:
        ckpt_list = [Path(args.ckpt)]
    else:
        ckpt_list = list_ckpts(args.ckpt_dir, args.pattern)
        if not ckpt_list:
            raise FileNotFoundError(f"No checkpoints found in {args.ckpt_dir} with pattern {args.pattern!r}")

    all_rows: List[Dict[str, Any]] = []

    for ckpt_path in ckpt_list:
        print("\n" + "="*90)
        print(f"Running ALL evals for: {ckpt_path.name}")
        print("="*90)

        model, preprocess = load_longclip(str(ckpt_path), device=args.device)

        t0 = time.time()
        row = {"ckpt_name": ckpt_path.name, "ckpt_path": str(ckpt_path)}

        metrics = run_all_tasks(model, preprocess, args)
        elapsed = time.time() - t0

        row.update(metrics)
        row["_meta_elapsed_sec"] = round(elapsed, 2)
        all_rows.append(row)

        # Pretty print small summary
        printable = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not k.startswith("_")}
        print_block("Summary (numbers only)", printable)

    write_outputs(all_rows, args)
    print("\nDone.")

if __name__ == "__main__":
    main()
