#!/usr/bin/env bash
# THE cross-dataset confirmation: vanilla ClipLoss vs ours (DreamLIP+SPECS) on DreamLIP-CC3M
# -- the actual long-caption data LLM2CLIP's official stage-2 training consumes. Same platform
# (ViT-B/16 + frozen Llama-3-8B-CC + linear adapter) as the DOCCI runs. Confirms whether the
# DOCCI finding (ours clearly beats vanilla) generalizes beyond one small dataset.
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
CACHE=/home/tachau/dreamlip_cc3m/llm2vec_cc_cache.pt
COMMON="--train_data dreamlip_cc3m --base_model llm2clip_text --text_cache_path ${CACHE} \
  --adapter_type linear --max_num_short_texts 4 --epochs 30 --batch_size 8 \
  --lr 1e-6 --adapter_lr 1e-3"

echo "[$(date +%T)] === CC3M BASELINE (vanilla ClipLoss) ==="
EXP=cc3m_vanilla_b16; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode vanilla --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === CC3M OURS (full: +DreamLIP+SPECS) ==="
EXP=cc3m_ours_b16; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode full --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === CC3M BOTH DONE ==="
