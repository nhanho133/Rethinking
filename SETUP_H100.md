# SETUP nhanh trên server 8×H100 (không git clone — chỉ tải zip về)

Gói này CHỈ chứa code (~2MB). 2 thứ NẶNG phải tải riêng trên server: (1) Python packages,
(2) model checkpoints ~18GB, (3) dataset. Server cần internet outbound tới pip + HuggingFace.
Nếu server HOÀN TOÀN offline (không tải được gì) -> xem mục "Trường hợp offline hoàn toàn" cuối file.

## 0. Giải nén
```bash
unzip rethinking_h100.zip -d rethinking_h100 && cd rethinking_h100
export PROJ=$PWD
```

## 1. Python env
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
# 1a. torch KHỚP CUDA của server (ví dụ cu124). KIỂM TRA `nvidia-smi` trước.
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu124
# 1b. phần còn lại
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git      # package `import clip`
# 1c. verify
python -c "import torch,transformers,llm2vec,clip; print(torch.cuda.device_count(),'GPUs', torch.__version__)"
```

## 2. Tải model checkpoint (~18GB, từ HuggingFace)
```bash
export CKPT=$PROJ/ckpts && mkdir -p $CKPT
python -c "from huggingface_hub import snapshot_download as d; \
 d('microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned', local_dir='$CKPT/Llama-3-8B-CC'); \
 d('microsoft/LLM2CLIP-Openai-L-14-336', local_dir='$CKPT/ViT-L-336')"
```
### ⚠️ 2b. FIX BẮT BUỘC cho checkpoint Llama-CC (nếu không sẽ crash lúc load)
transformers 4.44.2 không tương thích remote-code mặc định. Chạy 2 lệnh sau MỘT LẦN:
```bash
cd $CKPT/Llama-3-8B-CC
# (a) trỏ auto_map về file modeling local (bỏ prefix McGill-NLP--)
python -c "import json; c=json.load(open('config.json')); \
 c['auto_map']={'AutoModel':'modeling_llama_encoder.LlamaEncoderModel'}; \
 json.dump(c, open('config.json','w'), indent=2)"
# (b) thêm attn_mask_utils.py (bản McGill gốc, chỉ dùng AttentionMaskConverter có sẵn trong 4.44.2)
python -c "from huggingface_hub import hf_hub_download as d; import shutil; \
 p=d('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','attn_mask_utils.py'); \
 shutil.copy(p,'attn_mask_utils.py')"
cd $PROJ
```

## 3. Sửa PATH trong code (5 chỗ — dùng path thật của server)
| File | Biến cần sửa |
|---|---|
| `model/model_llm2clip.py` | `LLM2CLIPReleasedTeacher` default `model_path` -> `$CKPT/ViT-L-336` |
| `train/precompute_llm2vec_embeddings.py` | `LLM_PATH` -> `$CKPT/Llama-3-8B-CC`; `*_MANIFEST` -> path data |
| `train/datasets_config/sharegpt4v_coco.py` (hoặc dreamlip_cc3m.py) | `MANIFEST_PATH`, `IMAGE_ROOT` |
| `train/download_sharegpt4v_coco_subset.py` | `--json`, `--out_dir`, `--manifest` (hoặc truyền qua CLI) |
| `train/eval_*.py` | path cache + eval-data hardcoded |
Mẹo: `grep -rn "/home/tachau" .` để tìm hết path cũ cần thay.

## 4. Dataset TRAIN (chọn ShareGPT4V-COCO để bắt đầu — ổn định nhất)
```bash
export DATA=$PROJ/data && mkdir -p $DATA/sharegpt4v
python -c "from huggingface_hub import hf_hub_download as d; \
 d('Lin-Chen/ShareGPT4V','share-captioner_coco_lcs_sam_1246k_1107.json',repo_type='dataset',local_dir='$DATA/sharegpt4v')"
python train/download_sharegpt4v_coco_subset.py --target 118000 --workers 128 \
  --json $DATA/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json \
  --out_dir $DATA/sharegpt4v/images --manifest $DATA/sharegpt4v/manifest.json
```

## 5. Precompute text embedding (bf16 trên H100 — sửa load_l2v thành quant_4bit=False)
```bash
python train/precompute_llm2vec_embeddings.py --dataset sharegpt4v_coco \
  --max_num_short_texts 4 --batch_size 256 --no_4bit \
  --out $DATA/sharegpt4v/llm2vec_cc_cache.pt
```

## 6. CODE CÒN PHẢI SỬA trước khi chạy 8 GPU (xem H100_STAGE2_PLAN.md mục 5)
- ⭐ **all-gather cross-GPU** trong `model/model_llm2clip.py` forward (BẮT BUỘC cho multi-GPU đúng).
- Unfreeze ViT-L trong `LLM2CLIPReleasedTeacher` (thêm `freeze_visual=False`).
- (đã có sẵn: DDP wrap, checkpoint rank-0, mọi cờ ablation, loss_w áp cả 2 model.)

## 7. Chạy (ví dụ ours, 8 GPU)
```bash
torchrun --nproc_per_node=8 train/train.py \
  --train_data sharegpt4v_coco --base_model llm2clip_released \
  --llm2clip_model_path $CKPT/ViT-L-336 \
  --text_cache_path $DATA/sharegpt4v/llm2vec_cc_cache.pt \
  --adapter_type mlp --loss_mode full --max_num_short_texts 4 \
  --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 \
  --w_pos 1.0 --w_longer 1.0 --w_hinge 1.0 --exp_name h100_ours
```
Ablation: đổi `--loss_mode vanilla` (baseline), `--without_hinge_loss`, `--w_pos 0 --w_longer 0`.

## 8. Smoke test nhanh (bắt lỗi trước khi chạy full)
```bash
python train/test_case_1_split.py                 # test case 1 (không cần GPU/data)
# 1 GPU, 1 epoch, subset nhỏ:
CUDA_VISIBLE_DEVICES=0 python train/train.py --train_data sharegpt4v_coco \
  --base_model llm2clip_released --llm2clip_model_path $CKPT/ViT-L-336 \
  --text_cache_path $DATA/sharegpt4v/llm2vec_cc_cache.pt --loss_mode full \
  --epochs 1 --batch_size 8 --exp_name _smoke
```

---
## Trường hợp server OFFLINE HOÀN TOÀN (không pip/HF được)
Phải bundle thêm trên máy CÓ mạng rồi transfer:
1. `pip download -r requirements.txt -d wheels/` + torch wheel + CLIP wheel -> đóng gói `wheels/`,
   trên server `pip install --no-index --find-links wheels/ -r requirements.txt`.
2. Tải sẵn 2 checkpoint (18GB) + dataset về, nén, transfer qua cơ chế "download file" của server.
3. Đây là gói lớn (~20-40GB) — cần hỏi lại cách transfer file lớn vào server.
