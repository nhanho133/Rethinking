CUDA_VISIBLE_DEVICES=1,3 torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  train.py \
  --base_model ViT-B/16 \
  --train_data docci \
  --batch_size 4 \
  --epochs 10 \
  --lr 1e-6 \
  --weight_decay 1e-2 \
  --warmup_length 200 \
  --exp_name runs/exp_docci_ddp_bs16x4 

# python train.py \
#   --base_model ViT-B/16 \
#   --train_data docci \
#   --batch_size 8 \
#   --epochs 10 \
#   --lr 1e-6 \
#   --weight_decay 1e-2 \
#   --warmup_length 200 

# python train.py \
#   --base_model ViT-B/16 \
#   --train_data sharedgpt4v \
#   --batch_size 2 \
#   --epochs 10 \
#   --lr 1e-6 \
#   --weight_decay 1e-2 \
#   --warmup_length 200 \
#   --exp_name runs/exp_docci_ddp_bs16x4 