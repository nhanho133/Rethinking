#!/usr/bin/env bash
# Stage 2 — full fine-tune of the LLM2CLIP text-teacher variant (mirrors LLM2CLIP's own
# stage-2 recipe: LLM text teacher frozen, train the visual tower + text adapter against it).
# ViT-B/16 unfrozen at lr=1e-6 (matches the LongCLIP DOCCI run), text adapter at lr=1e-3.
# Same DOCCI split, 50 epochs, batch 8 -> a ViT-size-controlled comparison vs LongCLIP
# (T2I R@1 81.2%, I2T R@1 78.7%).
set -e
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
EXP=llm2clip_text_docci_50ep
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
mkdir -p "${EXP}"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$PY" train.py \
  --train_data docci \
  --base_model llm2clip_text \
  --text_cache_path "${CACHE}" \
  --max_num_short_texts 4 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-6 \
  --adapter_lr 1e-3 \
  --exp_name "${EXP}" \
  > "${EXP}/${EXP}.log" 2>&1 &
echo "Stage2 PID: $!"
