#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3


# exp_name="sharedgpt4v_propose_train_L14_100k"
base_model="ViT-B/16"
# base_model="ViT-L/14"
max_num_short_texts=6
exp_name="sharedgpt4v_10k_old_sampling_train_B16_numtxt_6"

# exp_name="artpedia_old_sampling_train_B16_numtxt_4"

mkdir -p "${exp_name}" # create folder exp_name if it doesn't exist
save_log="${exp_name}/${exp_name}.log" # save file log into the folder exp_name

# Missing 4,5,6 of artpedia

# init_ckpt="/home/ubuntu/shared/chau.thv/sd_weigth/LongCLIP-B/longclip-B.pt"
# init_ckpt="/home/ubuntu/hieu.tq/Git/FineLIP/train/experiments/experiment/ckpt/experiment_16_epoch_6_finelip-B-share_full.pt"
# /home/ubuntu/hieu.tq/Git/FineLIP/train/experiments/experiment/ckpt/experiment_16_epoch_6_finelip-B-share_full.pt
# openv1_new_sampling_train_B16_numtxt_4_init_fine4v
#  train_data: "sharedgpt4v", "docci", "dci", "openv1", "artpedia"
train_data="sharedgpt4v"
# batch_size=8
batch_size=16
# max_num_short_texts=8
nohup python train.py --train_data ${train_data} --max_num_short_texts ${max_num_short_texts} --exp_name ${exp_name} --batch_size ${batch_size} --base_model ${base_model}  > ${save_log} 2>&1 &

# python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 8 --base_model "ViT-B/16" 
# python train.py --train_data "docci" --max_num_short_texts 3 --batch_size 8 --base_model "ViT-B/16" 
# nohup python train.py --train_data "docci" --max_num_short_texts 4 --batch_size 16 --base_model "ViT-B/16" --exp_name propose_icml_docci_full_good > propose_icml_docci_full_good.log 2>&1 &
nohup python train.py --train_data "docci" --max_num_short_texts 4 --batch_size 16 --base_model "ViT-B/16" --exp_name propose_icml_docci_full_good_no_hinge_longer > propose_icml_docci_full_good_no_hinge_longer.log 2>&1 &
nohup python train.py --train_data "docci" --max_num_short_texts 3 --batch_size 8 --base_model "ViT-B/16" --exp_name propose_icml_docci1k_with_neg_loss > propose_icml_docci1k_with_neg_loss.log 2>&1 &
# nohup python train.py --train_data ${train_data} --init_ckpt ${init_ckpt} --max_num_short_texts ${max_num_short_texts} --exp_name ${exp_name} --batch_size ${batch_size} --base_model ${base_model}  > ${save_log} 2>&1 &
echo "Process started with PID: $!"


# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 8 --base_model "ViT-B/16" --exp_name propose_icml_share4v_full_good_with_neg_loss > propose_icml_share4v_full_good_with_neg_loss.log 2>&1 &

# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 8 --base_model "ViT-B/16" --exp_name propose_icml_share4v_full_good > propose_icml_share4v_full_good.log 2>&1 &

# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 16 --base_model "ViT-B/16" --exp_name propose_icml_share4v_full_good_batch16 > propose_icml_share4v_full_good_batch16.log 2>&1 &


# PID: 215004
# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 8 --base_model "ViT-B/32" --exp_name propose_icml_share4v_full_good_ViTB32 > propose_icml_share4v_full_good_ViTB32.log 2>&1 &

# PID: 245152
# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 16 --base_model "ViT-B/32" --exp_name propose_icml_share4v_full_good_ViTB32 > propose_icml_share4v_full_good_ViTB32.log 2>&1 &


# 369640
# nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 16 --base_model "ViT-B/16" --exp_name propose_icml_share4v_full_good_batch16 > propose_icml_share4v_full_good_batch16.log 2>&1

# 391683  propose_icml_share4v_full_good_ViTB32_batch16
nohup python train.py --train_data "sharedgpt4v" --max_num_short_texts 5 --batch_size 16 --base_model "ViT-B/32" --exp_name propose_icml_share4v_full_good_ViTB32_batch16 > propose_icml_share4v_full_good_ViTB32_batch16.log 2>&1 &

# 394263 propose_icml_docci_full_good_ViTB32
# nohup python train.py --train_data "docci" --max_num_short_texts 4 --batch_size 16 --base_model "ViT-B/32" --exp_name propose_icml_docci_full_good_ViTB32 > propose_icml_docci_full_good_ViTB32.log 2>&1 &
