#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenEventV1 Track 2 — Submission Builder (LongCLIP-style)
- Ưu tiên tokenizer: model_sail.longclip → model.longclip → model.finelip
- Ưu tiên encoder:   model.encode_image / model.encode_text
  (fallback an toàn sang encode_image_full/encode_text_full nếu cần)
- Viết CSV: query_id, image_id_1..image_id_10; nén ZIP sẵn sàng upload Codabench.

Ví dụ:
python submit_openeventv1_track2.py \
  --ckpt_path /path/to/ckpt.pt \
  --phase public \
  --query_csv /path/to/query_public.csv \
  --gallery_dir "/path/to/Gallery" \
  --device cuda:0 \
  --batch_size 64 --num_workers 8 \
  --output_prefix submission_longclip
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import normalize
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train")  # thư mục chứa sharegpt4v.py
sys.path.append("/home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/datasets")
sys.path.append("../..")

# -------------------------
# Project paths (tùy bạn)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append("..")

# ========== Utils ==========
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images_recursive(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def build_gallery_from_dir(images_dir: Path) -> Dict[str, Path]:
    """
    Map image_id -> image_path từ tên file (stem).
    Ví dụ: 123abc.jpeg -> image_id = "123abc"
    """
    mapping: Dict[str, Path] = {}
    files = find_images_recursive(images_dir)
    for p in files:
        image_id = p.stem
        if image_id in mapping and mapping[image_id] != p:
            # Giữ file đầu tiên nếu trùng id; có thể log nếu cần
            pass
        else:
            mapping[image_id] = p
    return mapping


def build_gallery_from_csv(csv_path: Path) -> Dict[str, Path]:
    """
    CSV 2 cột: image_id,file_path  (header bắt buộc)
    """
    mapping: Dict[str, Path] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "image_id" in reader.fieldnames and "file_path" in reader.fieldnames, \
            "gallery_csv phải có cột 'image_id' và 'file_path'"
        for r in reader:
            iid = str(r["image_id"]).strip()
            p = Path(r["file_path"]).expanduser()
            if not iid or not p.exists():
                continue
            if iid not in mapping:
                mapping[iid] = p
    return mapping


def read_queries(query_csv: Path) -> List[Tuple[str, str]]:
    """
    Trả về list (query_id, query_text), giữ nguyên thứ tự file.
    Hỗ trợ header 'query_index' (chuẩn Track 2) hoặc 'query_id'.
    """
    out = []
    with open(query_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        qid_key = "query_index" if "query_index" in reader.fieldnames else "query_id"
        assert qid_key in reader.fieldnames and "query_text" in reader.fieldnames, \
            "query.csv phải có cột 'query_index' (hoặc 'query_id') và 'query_text'"
        for r in reader:
            qid = r[qid_key].strip()
            qtext = (r["query_text"] or "").strip()
            if qid:
                out.append((qid, qtext))
    return out


# ========== Tokenizer resolver ==========
def resolve_tokenize():
    """
    Ưu tiên:
      1) model_sail.longclip.tokenize
      2) model.longclip.tokenize
      3) model.finelip.tokenize
    """
    try:
        from model_sail import longclip as _lc
        return _lc.tokenize, "longclip(model_sail)"
    except Exception:
        pass
    try:
        from model import longclip as _lc
        return _lc.tokenize, "longclip(model)"
    except Exception:
        pass
    try:
        from model import finelip as _fl
        return _fl.tokenize, "finelip(model)"
    except Exception:
        pass
    raise ImportError(
        "Không tìm thấy tokenizer. Hãy đảm bảo có `model_sail.longclip` "
        "hoặc `model.longclip` hoặc `model.finelip`."
    )


# ========== Model loader resolver ==========
def load_model_and_preprocess(ckpt_path: str, prefer: str = "auto"):
    """
    Trả về (model, preprocess, backend_str)
    Thử theo thứ tự:
      - longclip (model_sail → model)
      - finelip
    """
    backend_tried = []
    if prefer in ("auto", "longclip"):
        try:
            from model_sail import longclip
            m, pp = longclip.load(ckpt_path)
            return m, pp, "longclip(model_sail)"
        except Exception as e:
            backend_tried.append(f"model_sail.longclip: {e}")
        try:
            from model import longclip
            m, pp = longclip.load(ckpt_path)
            return m, pp, "longclip(model)"
        except Exception as e:
            backend_tried.append(f"model.longclip: {e}")

    if prefer in ("auto", "finelip"):
        try:
            from model import finelip
            # Giữ API giống môi trường bạn dùng
            m, pp = finelip.load(ckpt_path, device="cpu", run_finelip=True)
            return m, pp, "finelip(model)"
        except Exception as e:
            backend_tried.append(f"model.finelip: {e}")

    raise RuntimeError("Không load được model từ ckpt_path. Tried: \n- " + "\n- ".join(backend_tried))


# ========== Dataset cho gallery ==========
class ImageDataset(Dataset):
    def __init__(self, id_to_path: Dict[str, Path], preprocess):
        self.ids = list(id_to_path.keys())
        self.paths = [id_to_path[_id] for _id in self.ids]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img), self.ids[idx]


# ========== Encoders an toàn (wrapper) ==========
def _encode_image_safe(model, images: torch.Tensor) -> torch.Tensor:
    """
    Cố gắng gọi model.encode_image → [B,D]
    Nếu không có, fallback model.encode_image_full → lấy token 0 (CLS).
    """
    out = None
    if hasattr(model, "encode_image"):
        out = model.encode_image(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
    if out is None and hasattr(model, "encode_image_full"):
        full = model.encode_image_full(images)  # [B, P+1, D]
        out = full[:, 0, :]
    if out is None:
        raise AttributeError("Model không có encode_image hoặc encode_image_full")
    return out


def _encode_text_safe(model, tokens: torch.Tensor) -> torch.Tensor:
    """
    Cố gắng gọi model.encode_text → [B,D]
    Nếu không có, fallback encode_text_full @ text_projection rồi chọn theo chỉ số EOT (argmax token id).
    """
    out = None
    if hasattr(model, "encode_text"):
        out = model.encode_text(tokens)
        if isinstance(out, (tuple, list)):
            out = out[0]
    if out is None and hasattr(model, "encode_text_full"):
        if not hasattr(model, "text_projection"):
            raise AttributeError("Model có encode_text_full nhưng thiếu text_projection")
        t_full = model.encode_text_full(tokens) @ model.text_projection  # [B, T, D] hoặc [B, D]
        if t_full.ndim == 3:
            pick = tokens.argmax(dim=-1)  # [B]
            out = t_full[torch.arange(t_full.shape[0], device=t_full.device), pick]
        else:
            out = t_full
    if out is None:
        raise AttributeError("Model không có encode_text hoặc encode_text_full phù hợp")
    return out


# ========== Core pipeline ==========
@torch.no_grad()
def encode_gallery(model, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    all_feats: List[torch.Tensor] = []
    all_ids: List[str] = []
    for images, ids in tqdm(dataloader, desc="Encoding gallery"):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            emb = _encode_image_safe(model, images)  # [B, D]
        emb = normalize(emb.float(), dim=-1)
        all_feats.append(emb.cpu())
        all_ids.extend(list(ids))
    feats = torch.cat(all_feats, dim=0).numpy()
    return feats, all_ids


@torch.no_grad()
def encode_queries(model, texts: List[str], tokenize_fn, device: torch.device, chunk: int = 256) -> np.ndarray:
    outs: List[torch.Tensor] = []
    for s in range(0, len(texts), chunk):
        chunk_txt = texts[s:s+chunk]
        tokens = tokenize_fn(chunk_txt, truncate=True).to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            emb = _encode_text_safe(model, tokens)  # [B, D]
        emb = normalize(emb.float(), dim=-1)
        outs.append(emb.cpu())
    return torch.cat(outs, dim=0).numpy()


def topk_for_each_query(q_feats: np.ndarray, g_feats: np.ndarray, k: int = 10, batch: int = 512) -> np.ndarray:
    """
    q_feats: [M, D], g_feats: [N, D] (đều đã L2-normalized)
    Trả về chỉ số top-k trong gallery theo từng query: shape [M, k]
    """
    M, _ = q_feats.shape
    out_idx = np.zeros((M, k), dtype=np.int64)

    for s in tqdm(range(0, M, batch), desc="Scoring queries"):
        qe = q_feats[s:s+batch]                  # [b, D]
        sims = qe @ g_feats.T                    # [b, N]
        part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        # sort cục bộ theo điểm giảm dần
        part_sorted = part[np.arange(part.shape[0])[:, None],
                           np.argsort(-sims[np.arange(sims.shape[0])[:, None], part], axis=1)]
        out_idx[s:s+batch] = part_sorted
    return out_idx


def write_submission_csv(
    out_csv: Path,
    query_ids: List[str],
    topk_indices: np.ndarray,
    gallery_ids: List[str],
    fill_token: str = "#"
):
    """
    header: query_id, image_id_1..image_id_10
    """
    assert topk_indices.shape[1] == 10, "Top-k phải = 10 để khớp format submission."
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["query_id"] + [f"image_id_{i}" for i in range(1, 11)]
        writer.writerow(header)
        for qi, row in zip(query_ids, topk_indices):
            preds = [gallery_ids[j] if (0 <= j < len(gallery_ids)) else fill_token for j in row.tolist()]
            preds = [p if (p and isinstance(p, str)) else fill_token for p in preds]
            writer.writerow([qi] + preds)


def zip_single_file(csv_path: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser("OpenEventV1 Track 2 submission builder (LongCLIP-style)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Đường dẫn checkpoint (LongCLIP/FineLIP)")
    parser.add_argument("--phase", type=str, required=True, choices=["public", "private"], help="Chọn phase (ghi tên file)")
    parser.add_argument("--query_csv", type=str, required=True, help="Đường dẫn tới query.csv (Public/Private)")
    # Gallery nguồn ảnh: 1) quét thư mục, 2) CSV map
    parser.add_argument("--gallery_dir", type=str, default=None, help="Thư mục chứa ảnh gallery (quét đệ quy)")
    parser.add_argument("--gallery_csv", type=str, default=None, help="CSV có cột image_id,file_path (ưu tiên hơn gallery_dir)")
    # Cache
    parser.add_argument("--cache_dir", type=str, default="./cache_openeventv1", help="Thư mục lưu cache npy")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size encode ảnh")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_prefix", type=str, default="submission_longclip", help="Tiền tố tên file xuất (csv/zip)")
    parser.add_argument("--prefer_backend", type=str, default="auto", choices=["auto", "longclip", "finelip"],
                        help="Ưu tiên loader nào (auto = thử longclip rồi finelip)")
    parser.add_argument("--max_gallery", type=int, default=None, help="Giới hạn số ảnh gallery (debug)")
    parser.add_argument("--query_chunk", type=int, default=256, help="Chunk encode query (tránh tràn VRAM)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Load model & preprocess
    model, preprocess, backend = load_model_and_preprocess(args.ckpt_path, prefer=args.prefer_backend)
    print(f"[Info] Loaded model via {backend}")
    # FP16 trên GPU để tăng tốc
    if device.type == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    # 2) Resolve tokenizer
    tokenize_fn, tok_src = resolve_tokenize()
    print(f"[Info] Using tokenizer from {tok_src}")

    # 3) Load queries
    query_pairs = read_queries(Path(args.query_csv))  # [(qid, qtext), ...]
    query_ids = [q for q, _ in query_pairs]
    query_texts = [t for _, t in query_pairs]
    print(f"[Info] Loaded {len(query_ids)} queries from: {args.query_csv}")

    # 4) Build gallery map
    if args.gallery_csv:
        id2path = build_gallery_from_csv(Path(args.gallery_csv))
    elif args.gallery_dir:
        id2path = build_gallery_from_dir(Path(args.gallery_dir))
    else:
        raise ValueError("Bạn phải cung cấp --gallery_csv hoặc --gallery_dir")
    if not id2path:
        raise RuntimeError("Không tìm thấy ảnh gallery hợp lệ.")
    if args.max_gallery is not None:
        # Giới hạn (ổn định thứ tự bằng sort id)
        keep_ids = sorted(list(id2path.keys()))[:args.max_gallery]
        id2path = {k: id2path[k] for k in keep_ids}
    print(f"[Info] Gallery size: {len(id2path)} images")

    # 5) Encode gallery (có cache, gắn ckpt để tránh nhầm)
    ckpt_tag = Path(args.ckpt_path).stem
    gal_cache_feats = Path(args.cache_dir) / f"gallery_feats_{len(id2path)}_{ckpt_tag}.npy"
    gal_cache_ids = Path(args.cache_dir) / f"gallery_ids_{len(id2path)}_{ckpt_tag}.txt"

    if gal_cache_feats.exists() and gal_cache_ids.exists():
        print(f"[Cache] Loading gallery features from: {gal_cache_feats}")
        g_feats = np.load(gal_cache_feats)
        with open(gal_cache_ids, "r", encoding="utf-8") as f:
            gallery_ids = [line.strip() for line in f if line.strip()]
        assert g_feats.shape[0] == len(gallery_ids), "Cache mismatch gallery size."
    else:
        gal_ds = ImageDataset(id2path, preprocess)
        gal_loader = DataLoader(
            gal_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )
        g_feats, gallery_ids = encode_gallery(model, gal_loader, device)
        np.save(gal_cache_feats, g_feats)
        with open(gal_cache_ids, "w", encoding="utf-8") as f:
            for iid in gallery_ids:
                f.write(iid + "\n")

    # 6) Encode queries
    q_feats = encode_queries(model, query_texts, tokenize_fn, device, chunk=args.query_chunk)  # [M, D]

    # 7) Ranking & lấy Top-10
    topk_idx = topk_for_each_query(q_feats, g_feats, k=10, batch=512)  # [M, 10]

    # 8) Ghi submission.csv + zip
    out_csv = Path(f"{args.output_prefix}_{args.phase}.csv")
    write_submission_csv(out_csv, query_ids, topk_idx, gallery_ids, fill_token="#")
    out_zip = Path(f"{args.output_prefix}_{args.phase}.zip")
    zip_single_file(out_csv, out_zip)

    print(f"[Done] Wrote CSV: {out_csv.resolve()}")
    print(f"[Done] Wrote ZIP: {out_zip.resolve()}")
    print("Giờ bạn có thể upload file ZIP lên Codabench (chọn đúng phase).")


if __name__ == "__main__":
    main()
