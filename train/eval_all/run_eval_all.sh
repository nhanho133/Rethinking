export CUDA_VISIBLE_DEVICES=5
# nohup python eval_all.py \
# --ckpt /home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/efficient_full_openv1_softmax_learnable_mps/ckpt/B16-longclip-09-11--10_26_13_-4.pt \
# --out_csv eval_efficient_SOFTMAX_openv1_epoch05.csv \
# --out_json eval_efficient_SOFTMAX_openv1_epoch05.json > ./logs/epoch5/eval_efficient_SOFTMAX_openv1_epoch05.log 2>&1 &

nohup python eval_all.py \
python eval_all.py --ckpt /home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/shared1k_propose/ckpt/B16-longclip-10-11--15_03_09_-2.pt\


# nohup python eval_all.py \
# --ckpt /home/ubuntu/hieu.tq/Git/KDPL_test/KDPL/src/LongCLIPMul_docci/train/efficient_full_sharedgpt4v_sparsemax_learnable_mps/ckpt/B16-longclip-09-13--18_51_27_-1.pt \
# --out_csv eval_efficient_sparsemax_sharedgpt_epoch02.csv \
# --out_json eval_efficient_sparsemax_sharedgpt_epoch02.json > ./logs_sharedgpt/epoch02/eval_efficient_sparsemax_sharedgpt_epoch02.log 2>&1 &
