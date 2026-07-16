#!/usr/bin/env bash
# Wait until GPU 0 has >= THRESH MiB free (stably), then launch training.
# Does NOT touch any other GPU job.
cd /home/truongchau/Chau/Rethinking_project/train
PY=/home/truongchau/miniconda3/envs/anyedit/bin/python
EXP=sharegpt4v_coco_llava_676k_B16_recon
WLOG=gpu_wait.log
THRESH=18000    # MiB free required for batch-16 (~15GB peak + margin)
STABLE=2        # consecutive 60s checks to confirm it's not a transient dip

count=0
echo "[$(date +%H:%M:%S)] watcher start; need >=${THRESH}MiB free x${STABLE}" >> "$WLOG"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  ts=$(date +%H:%M:%S)
  if [ -n "$free" ] && [ "$free" -ge "$THRESH" ]; then
    count=$((count+1))
    echo "[$ts] free=${free}MiB OK ($count/$STABLE)" >> "$WLOG"
    [ "$count" -ge "$STABLE" ] && { echo "[$ts] GPU FREE -> launching" >> "$WLOG"; break; }
  else
    [ "$count" -ne 0 ] && echo "[$ts] free=${free:-?}MiB dipped, reset" >> "$WLOG"
    count=0
  fi
  sleep 60
done

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
  "$PY" train.py --train_data sharedgpt4v --max_num_short_texts 6 \
  --batch_size 16 --base_model "ViT-B/16" --exp_name "$EXP" \
  > "${EXP}.log" 2>&1 &
echo "[$(date +%H:%M:%S)] TRAINING LAUNCHED pid $!" >> "$WLOG"
