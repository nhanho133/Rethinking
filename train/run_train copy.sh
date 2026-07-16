#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4


exp_name="openv1_propose_train"
mkdir -p "${exp_name}" # create folder exp_name if it doesn't exist
save_log="${exp_name}/${exp_name}.log" # save file log into the folder exp_name

#  train_data: "sharedgpt4v", "docci", "dci", "openv1", "artpedia"
train_data="openv1"
batch_size=16
# max_num_short_texts=8
max_num_short_texts=4
nohup python train.py --train_data ${train_data} --max_num_short_texts ${max_num_short_texts} --exp_name ${exp_name} --batch_size ${batch_size}  > ${save_log} 2>&1 &
echo "Process started with PID: $!"
