#!/usr/bin/env bash
# THE experiment for the thesis: on the faithful released LLM2CLIP platform
# (frozen ViT-L-336 + their pretrained TextProj adapter), fine-tune on DOCCI with
#   (1) vanilla = LLM2CLIP's ClipLoss (L_long only)      -> baseline
#   (2) full    = + DreamLIP multi-positive + SPECS hinge -> ours
# Identical everything else. Runs sequentially on the single GPU.
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
COMMON="--train_data docci --base_model llm2clip_released --text_cache_path ${CACHE} \
  --max_num_short_texts 4 --epochs 20 --batch_size 8 --lr 1e-6 --adapter_lr 1e-5"

echo "[$(date +%T)] === BASELINE (vanilla ClipLoss) ==="
EXP=llm2clip_released_vanilla; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode vanilla --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === OURS (full: +DreamLIP+SPECS) ==="
EXP=llm2clip_released_ours; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode full --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === BOTH DONE ==="
