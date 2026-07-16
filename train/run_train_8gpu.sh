#!/usr/bin/env bash
# Task 4 (PDF: Test case Rethinking project, 9/7/2026): multi-GPU launch template for the
# 8xH100 server. Mirrors run_docci_bigbatch_ablation.sh but uses torchrun so train.py's
# existing setup_distributed()/DistributedSampler/DistributedDataParallel path is exercised.
#
# NOTE (read before scaling up): batch_size below is PER GPU (train.py --batch_size help
# text says so), so total effective batch = batch_size * NPROC. Contrastive in-batch
# negatives are computed per-GPU only -- text/image features are NOT all_gather'd across
# ranks before the loss, so going from 1 to 8 GPUs raises *throughput*, not the number of
# negatives per loss computation. If the goal on the H100 box is more negatives (not just
# more images/sec), add an all_gather step for the embeddings in model_llm2clip.py /
# model_longclip.py forward() before computing logits -- untested here since this dev box
# only has 1 GPU.
set -euo pipefail
cd /home/tachau/Rethinking_project/train

PY=/home/tachau/rethink_venv/bin/python
NPROC=${NPROC:-8}
CACHE=/home/tachau/docci_data/llm2vec_cc_cache.pt
COMMON="--train_data docci --base_model llm2clip_text --text_cache_path ${CACHE} \
  --adapter_type linear --max_num_short_texts 4 --epochs 30 --batch_size 96 \
  --lr 1e-6 --adapter_lr 1e-3"

echo "[$(date +%T)] === DOCCI-8GPU VANILLA ==="
EXP=docci_8gpu_vanilla; mkdir -p "$EXP"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  train.py $COMMON --loss_mode vanilla --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === DOCCI-8GPU OURS ==="
EXP=docci_8gpu_ours; mkdir -p "$EXP"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  train.py $COMMON --loss_mode full --exp_name "$EXP" > "${EXP}/${EXP}.log" 2>&1

echo "[$(date +%T)] === DOCCI-8GPU BOTH DONE ==="
