# Rethinking long-context VLM as multi-view contrastive learning — LLM2CLIP stage-2

Loss Rethinking (L_long + DreamLIP multi-positive + SPECS hinge) ghép vào text-teacher
Llama-3-8B-CC (frozen) của LLM2CLIP, train stage-2 trên 8×H100.

Code tối thiểu để chạy 3 version:
1. **baseline** LLM2CLIP (ClipLoss) trên ShareGPT4V
2. **ours** (loss Rethinking) trên ShareGPT4V
3. **ours** trên CC3M

## Setup
Xem `SETUP_H100.md` (env conda, tải checkpoint + fix config Llama-CC) và `H100_STAGE2_PLAN.md`.
Path server đã set sẵn trong code:
- data: `/cm/archive/luongtk/{docci,sharegpt4v}`
- ckpt: `/cm/shared/chautvh_second/Nhan_folder/ckpts/{Llama-3-8B-CC,ViT-L-336}`
- work (cache/manifest/output): `/cm/shared/chautvh_second/Nhan_folder/work`

## Chạy
```bash
cd train
mkdir -p /cm/shared/chautvh_second/Nhan_folder/work

# ===== TASK 1 + 2: ShareGPT4V (data đã có sẵn server) =====
python make_sharegpt4v_subset.py --n 120000                        # tạo subset manifest 1 lần
python precompute_llm2vec_embeddings.py --dataset sharegpt4v_coco \
  --batch_size 256 --no_4bit \
  --out /cm/shared/chautvh_second/Nhan_folder/work/sgpt4v_cache.pt   # precompute embedding (bf16)

CACHE=/cm/shared/chautvh_second/Nhan_folder/work/sgpt4v_cache.pt

# TASK 1 — baseline
torchrun --nproc_per_node=8 train.py --train_data sharegpt4v_coco --base_model llm2clip_released \
  --text_cache_path $CACHE --loss_mode vanilla --adapter_type mlp \
  --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 --exp_name sgpt4v_baseline

# TASK 2 — ours
torchrun --nproc_per_node=8 train.py --train_data sharegpt4v_coco --base_model llm2clip_released \
  --text_cache_path $CACHE --loss_mode full --adapter_type mlp \
  --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 --exp_name sgpt4v_ours

# ===== TASK 3: CC3M (chưa có trên server -> tải trước) =====
python download_cc3m_subset.py --target 120000 --workers 128
python precompute_llm2vec_embeddings.py --dataset dreamlip_cc3m --batch_size 256 --no_4bit \
  --out /cm/shared/chautvh_second/Nhan_folder/work/cc3m_cache.pt
torchrun --nproc_per_node=8 train.py --train_data dreamlip_cc3m --base_model llm2clip_released \
  --text_cache_path /cm/shared/chautvh_second/Nhan_folder/work/cc3m_cache.pt \
  --loss_mode full --adapter_type mlp --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 \
  --exp_name cc3m_ours
```

Retrieval R@1 in ra mỗi epoch (test_epoch_ver5 trong train.py) — không cần script eval riêng.
Smoke test 1 GPU trước khi chạy full: bỏ `torchrun --nproc_per_node=8`, thêm `--epochs 1 --batch_size 8`.

## Ablation (tùy chọn)
`--without_hinge_loss` (bỏ hinge) · `--w_pos 0 --w_longer 0` (chỉ hinge) · `--weight_hinge_loss 0.5`.
