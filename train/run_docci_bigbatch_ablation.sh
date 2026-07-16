#!/usr/bin/env bash
# Same vanilla-vs-ours ablation as before, but with batch_size=96 instead of 8 -- tests
# whether more in-batch negatives (real fix, not grad accum) removes the "vanilla loss
# saturates near-zero too fast" pathology seen at batch=8.
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
COMMON="--train_data docci --base_model llm2clip_text --text_cache_path ${CACHE} \
  --adapter_type linear --max_num_short_texts 4 --epochs 30 --batch_size 96 \
  --lr 1e-6 --adapter_lr 1e-3"

echo "[$(date +%T)] === DOCCI-BIGBATCH VANILLA ==="
EXP=docci_bigbatch_vanilla; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode vanilla --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === DOCCI-BIGBATCH OURS ==="
EXP=docci_bigbatch_ours; mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train.py $COMMON --loss_mode full --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === DOCCI-BIGBATCH BOTH DONE ==="
