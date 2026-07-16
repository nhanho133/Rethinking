#!/usr/bin/env bash
# Stage 1 — frozen-probe go/no-go check for the LLM2CLIP text-teacher variant.
# Visual (ViT-B/16) frozen + Llama-3-8B-CC text cache frozen; only the 4096->512 text
# adapter trains. Higher lr than full fine-tuning since the adapter is randomly initialized.
# Fast (adapter-only) -> a quick signal on whether the frozen Llama-CC text space is usable
# as an alignment target before committing to Stage 2's full fine-tune.
set -e
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
EXP=llm2clip_frozen_probe
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
mkdir -p "${EXP}"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$PY" train.py \
  --train_data docci \
  --base_model llm2clip_frozen \
  --text_cache_path "${CACHE}" \
  --max_num_short_texts 4 \
  --epochs 12 \
  --batch_size 32 \
  --lr 1e-3 \
  --exp_name "${EXP}" \
  > "${EXP}/${EXP}.log" 2>&1 &
echo "Stage1 PID: $!"
