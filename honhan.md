cd /cm/shared/chautvh_second/Nhan_folder/train
conda activate rethink
CACHE_DIR=/cm/shared/chautvh_second/Nhan_folder/work/sgpt4v_full_parts

CUDA_VISIBLE_DEVICES=4,5 nohup torchrun --nproc_per_node=2 --master_port=29501 train.py \
  --train_data sharegpt4v_coco \
  --base_model llm2clip_released \
  --text_cache_path $CACHE_DIR \
  --loss_mode full \
  --adapter_type mlp \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-5 \
  --adapter_lr 1e-5 \
  --exp_name sgpt4v_full_ours_2gpu \
  > ours_train.log 2>&1 &

tail -f ours_train.log
