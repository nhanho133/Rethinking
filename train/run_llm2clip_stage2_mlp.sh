#!/usr/bin/env bash
# Stage 2 (v2) — same as Stage 2 but with LLM2CLIP's faithful TextProj adapter (4-layer
# residual MLP, ~270M params) instead of the single-Linear adapter. adapter_lr lowered to
# 1e-4 (the deep MLP is far bigger than the old linear; 1e-3 risks instability/overfit).
set -e
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
EXP=llm2clip_text_mlp_docci_50ep
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
mkdir -p "${EXP}"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$PY" train.py \
  --train_data docci --base_model llm2clip_text --text_cache_path "${CACHE}" \
  --max_num_short_texts 4 --epochs 50 --batch_size 8 \
  --lr 1e-6 --adapter_lr 1e-4 --exp_name "${EXP}" \
  > "${EXP}/${EXP}.log" 2>&1 &
echo "Stage2-MLP PID: $!"
