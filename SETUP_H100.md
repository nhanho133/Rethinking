# SETUP trên server 8×H100

## 0. Proxy (mỗi terminal mới đều phải export)
```bash
export http_proxy="http://chautvh:Unitok%40Apr2026@10.16.32.11:8080"
export https_proxy="$http_proxy"; export HTTP_PROXY="$http_proxy"; export HTTPS_PROXY="$http_proxy"
```

## 1. Conda env
```bash
conda create -n rethink python=3.10 -y && conda activate rethink
nvidia-smi | grep -i "CUDA Version"          # xem version để chọn cuXXX bên dưới
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # đổi cu124 cho khớp
pip install transformers==4.44.2 llm2vec accelerate sentencepiece safetensors \
    huggingface-hub ftfy regex tqdm pandas numpy pillow matplotlib requests scikit-learn scipy
pip install git+https://github.com/openai/CLIP.git
python -c "import torch,transformers,llm2vec,clip; print('gpus', torch.cuda.device_count())"   # phải =8
```
> Nếu tải checkpoint bị lỗi `brotli ... Segmentation fault`: `pip uninstall -y brotli brotlicffi` rồi tải lại.

## 2. Tải checkpoint + FIX BẮT BUỘC
```bash
export CKPT=/cm/shared/chautvh_second/Nhan_folder/ckpts && mkdir -p $CKPT
hf download microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned --local-dir $CKPT/Llama-3-8B-CC
hf download microsoft/LLM2CLIP-Openai-L-14-336 --local-dir $CKPT/ViT-L-336

# FIX config Llama-CC (không có sẽ crash lúc load model):
cd $CKPT/Llama-3-8B-CC
python -c "import json;c=json.load(open('config.json'));c['auto_map']={'AutoModel':'modeling_llama_encoder.LlamaEncoderModel'};json.dump(c,open('config.json','w'),indent=2);print('patched')"
python -c "from huggingface_hub import hf_hub_download as d;import shutil;shutil.copy(d('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','attn_mask_utils.py'),'attn_mask_utils.py');print('added attn_mask_utils')"
cd -
```

## 3. Chạy 3 task
Xem `README.md`. Data DOCCI/ShareGPT4V đã có sẵn trên server (path đã set trong code).
