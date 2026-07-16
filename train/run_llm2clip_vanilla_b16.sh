#!/usr/bin/env bash
# Vanilla-LLM2CLIP-loss baseline on the ViT-B/16 from-scratch platform, to pair with the
# existing "ours" run (llm2clip_text_docci_50ep = 80.1/78.5). Identical config, only loss differs.
cd /home/tachau/Rethinking_project/train
PY=/home/tachau/rethink_venv/bin/python
EXP=llm2clip_text_vanilla_b16_50ep
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
mkdir -p "$EXP"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$PY" train.py \
  --train_data docci --base_model llm2clip_text --text_cache_path "$CACHE" \
  --adapter_type linear --loss_mode vanilla \
  --max_num_short_texts 4 --epochs 50 --batch_size 8 --lr 1e-6 --adapter_lr 1e-3 \
  --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1 &
echo "vanilla-B16 PID: $!"
