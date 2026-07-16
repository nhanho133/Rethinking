# HANDOFF CONTEXT — Rethinking (LongCLIP loss) ghép vào LLM2CLIP text-teacher

> File này tổng hợp toàn bộ context từ lúc bắt đầu (định nghĩa 3 task path) đến hiện tại, để
> dán sang platform/AI khác tiếp tục làm việc mà không mất ngữ cảnh.

## 1. Mục tiêu gốc của project

Có sẵn 1 model LongCLIP ViT-B/16 fine-tune trên DOCCI với loss tự đề xuất
("Rethinking": Adaptive Multi-Positive kiểu DreamLIP + specificity hinge kiểu SPECS),
kết quả baseline đã train xong: **T2I R@1 81.2% / I2T R@1 78.7%** trên test split DOCCI
(`train/docci_3060_50ep/`).

**Mục tiêu**: ghép loss "Rethinking" đó vào **LLM2CLIP** (thay vì tự train text encoder, dùng
thẳng text teacher đóng băng của LLM2CLIP — `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`,
Llama-3-8B đã qua fine-tune Caption-to-Caption contrastive của Microsoft) để so sánh:
**loss gốc ClipLoss của LLM2CLIP** vs **loss Rethinking**, trên cùng 1 nền model.

## 2. Lịch sử: đã làm gì, kết quả gì (chạy trên GPU đơn RTX 3060 12GB, trước khi có H100)

### Vòng 1 — Batch nhỏ (batch=8), ViT-B/16 tự train + Llama-CC frozen + adapter linear (4096→512)

| # | Cấu hình | Dataset | T2I R@1 | I2T R@1 |
|---|---|---|---|---|
| 1 | LongCLIP gốc (text tự train, không LLM2CLIP) | DOCCI | 81.2% | 78.7% |
| 2 | LLM2CLIP-text, loss **vanilla** (ClipLoss gốc) | DOCCI | 74.2% | 75.3% |
| 3 | LLM2CLIP-text, loss **ours** (DreamLIP+SPECS) | DOCCI | **80.1%** | **78.5%** |
| 4 | LLM2CLIP-text, loss **vanilla** | DreamLIP-CC3M (15k subset) | 94.2% | 97.1% |
| 5 | LLM2CLIP-text, loss **ours** | DreamLIP-CC3M (15k subset) | **97.4%** | **98.1%** |

Kết luận vòng 1: ours thắng vanilla nhất quán ở batch nhỏ, nhưng **chưa vượt qua LongCLIP gốc**
trên DOCCI (80.1/78.5 < 81.2/78.7) — kết luận đúng phải là "ours > vanilla trên nền LLM2CLIP",
không phải "LLM2CLIP+ours > LongCLIP".

### Vòng 2 — Kiểm chứng ở batch lớn (96) + Specificity Rate (metric SPECS) + Zero-shot

**Phát hiện lõi: lợi thế của "ours" ở vòng 1 chủ yếu là hiện tượng batch nhỏ, không phải thắng
tuyệt đối.**

- **In-domain DOCCI R@1**: batch=8 ours thắng (80.1/78.5 > 74.2/75.3); batch=96 **vanilla thắng**
  (77.6/79.5 > 73.4/74.8) — hạ trọng số sub-loss (`loss_w`) chỉ cải thiện chút ít, không đảo ngược.
  Nguyên nhân: ở batch lớn `L_long` bão hòa ~0 rất nhanh còn `L_pos` vẫn cao (~50-100x), gradient
  bị `L_pos` hút, kéo model rời mục tiêu retrieval câu đầy đủ.
- **Specificity Rate (SPECS metric)**: DOCCI b8: vanilla 87.1 vs ours **98.8** (+11.7). CC3M b8:
  vanilla 69.9 vs ours **99.8** (+29.8). DOCCI b96: 86.2 vs 87.2 — gần như hòa.
- **Zero-shot COCO val R@1** (train DOCCI, không fine-tune): b8 ours thắng T2I (21.4>14.7) nhưng
  **thua I2T** (16.2<20.0). b96 vanilla thắng cả 2 chiều (18.3/22.2 > 17.3/16.1).
- **Zero-shot Flickr30K R@1**: cùng mẫu hình — b8 ours thắng T2I thua I2T; b96 vanilla thắng cả 2.
- **Zero-shot ImageNetV2 classification**: b8 ours 28.6/58.0 > vanilla 22.1/48.3; b96 **vanilla**
  30.5/60.1 > ours 28.4/57.3. (Tham chiếu: CLIP ViT-B/16 gốc chưa fine-tune ~62% top1 — mọi
  checkpoint đều tụt mạnh vì fine-tune 50ep/15k ảnh phá khả năng phân loại tổng quát, như dự đoán.)

**Kết luận vòng 2 (khớp cảnh báo trong paper SPECS: "strong specificity degrades retrieval/
classification"):**
1. Batch nhỏ: ours có lợi thế nhưng là **đánh đổi** — thắng specificity/T2I, thua I2T/
   generalization.
2. Batch lớn: lợi thế biến mất gần hết, **vanilla thắng đều** ở cả 4 loại đánh giá độc lập
   (in-domain, specificity rate, zero-shot retrieval, zero-shot classification).
3. Hệ quả: **không thể bê nguyên config batch nhỏ lên chạy quy mô lớn.** Cần thiết kế lại cách
   cân bằng 4 loss (adaptive/uncertainty weighting thay vì hằng số `loss_w=(1,1,1,1)` cố định,
   hoặc warm-up `L_long` trước rồi mới thêm sub-loss) trước khi commit tài nguyên lớn.

### Giới hạn/hạn chế đã biết (từ vòng 1+2)
- Chưa train được visual ViT-L/14-336 GỐC của LLM2CLIP trên 3060 (OOM do Adam optimizer states
  ~9GB cho 304M+274M params) — mọi so sánh vòng 1+2 dùng ViT-B/16 để cô lập biến text-teacher.
- CC3M subset chỉ 15k/2.87M ảnh gốc (băng thông/thời gian tải giới hạn).
- Adapter dùng bản linear đơn giản; đã thử MLP 4-layer residual chuẩn LLM2CLIP nhưng overfit
  nặng trên data nhỏ (15k ảnh), kết quả tệ hơn linear.
- Mỗi cấu hình chỉ 1 seed — chưa đo độ ổn định thống kê.
- Zero-shot dùng 1 caption/ảnh (không phải 5-caption protocol chuẩn COCO).

## 3. Code hiện có (đã viết xong, tái sử dụng được)

Nằm ở `model/` và `train/` (local: `/home/tachau/Rethinking_project/`, server:
`~/Nhan_folder/{model,train,...}` — đã "kéo ra ngoài" khỏi thư mục `Rethinking-main/` cũ).

- **`model/model_llm2clip.py`**:
  - `TextEmbeddingCache`: load cache `sha1(text) -> float16[llm_dim]` precompute sẵn, `lookup()`
    fail loudly nếu miss (không bao giờ âm thầm live-encode 8B lúc train).
  - `LLM2CLIPTextTeacher`: ViT-B/16 (tự train hoặc freeze) + Llama-CC frozen (qua cache) +
    `TextAdapter` (linear hoặc mlp, 4096→512). `forward()` trả 7-tuple
    `(loss, L_long, L_pos, L_longer, L_hinge, logit_scale, tau)` — y hệt signature
    `model_longclip.py` gốc. Loss = `w_long*L_long + w_pos*L_pos + w_longer*L_longer (+
    w_hinge*L_hinge nếu use_hinge)`, mặc định `loss_w=(1,1,1,1)`, override được qua
    `--w_pos/--w_longer/--w_hinge`.
  - `LLM2CLIPReleasedTeacher` (subclass): bản **"chuẩn LLM2CLIP" thật sự** — load thẳng
    checkpoint released `microsoft/LLM2CLIP-Openai-L-14-336` qua
    `AutoModel.from_pretrained(..., trust_remote_code=True)`, dùng **ViT-L/14-336 gốc** của họ
    (không phải ViT-B/16) + `get_text_features` = TextProj gốc của họ (L2-norm → 4x residual
    MLP → LN → Linear → 1280-d). `freeze_visual` mặc định **False** trên H100 (train full visual
    tower — đúng recipe "stage 2" thật của LLM2CLIP; trên 3060 từng phải freeze vì OOM).
  - Cả 2 class dùng chung `_encode_grid` / `_adaptive_mp_loss` (cross-GPU negatives qua
    `all_gather_grad`, no-op nếu 1 GPU) / hinge loss logic.

- **`train/precompute_llm2vec_embeddings.py`**: script offline, load Llama-3-8B-CC (4-bit hoặc
  bf16 qua `--no_4bit`), **enumerate hết mọi chuỗi text cần thiết** (không phải sample ngẫu
  nhiên mỗi epoch) rồi cache 1 lần ra `.pt`. Hỗ trợ `--dataset {docci, dreamlip_cc3m,
  sharegpt4v_coco}`. **Chỉ chạy được trên 1 GPU** (`device_map={"":0}` hardcode — set
  `CUDA_VISIBLE_DEVICES` để chọn GPU vật lý nào).

- **`train/make_sharegpt4v_subset.py`**: lấy subset N ảnh từ file JSON gốc ShareGPT4V
  (`/cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json`, **1.246 triệu
  items** tổng cộng), random shuffle seed=42, ghi manifest
  `/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v_subset_manifest.json`.

- **`train/sampling.py`**: `star_bar_long_text_split(sentences, K, seed)` — **QUAN TRỌNG**: với
  seed cố định (=42 trong precompute), hàm này chỉ trả về **ĐÚNG MỘT cách chia** n câu thành K
  đoạn liên tiếp (`partition_indices_random` dùng `np.random.default_rng(seed)` — deterministic),
  **không phải liệt kê mọi cách chia có thể (`C(n-1,K-1)`)**. Đây là lý do số chuỗi cần cache
  KHÔNG nổ tổ hợp — xem mục 5 bên dưới.

- **`train/train.py`**: `--base_model {llm2clip_frozen, llm2clip_text, llm2clip_released, ...}`,
  `--loss_mode {full, vanilla}`, `--adapter_type {linear, mlp}`, `--adapter_lr`,
  `--w_pos/--w_longer/--w_hinge`, `--without_hinge_loss`, `--weight_hinge_loss`,
  `--grad_accum_steps`, `--freeze_visual`, `--llm2clip_model_path` (default trỏ tới ckpt
  ViT-L-336 released). `test_epoch_ver5` tính R@1 mỗi epoch, không cần script eval riêng.

- **Eval bổ sung (vòng 2)**: `train/eval_specificity_rate.py`, `train/eval_zeroshot_retrieval.py`,
  `train/eval_zeroshot_imagenet.py`.

- **`train/plot_losses.py`**: vẽ loss từ `exp_name/loss_log.csv` (train.py tự ghi mỗi 50 step).

## 4. Trạng thái hiện tại: chuyển sang chạy trên server 8×H100 (session này)

### Đã hoàn thành trên server:
1. **Conda env `rethink`, Python 3.10** (KHÔNG dùng `base` — base là Python 3.13, gây lỗi build
   `tokenizers` vì PyO3/maturin chưa support 3.13).
2. **Proxy** (mọi terminal mới đều phải export lại):
   ```bash
   export http_proxy="http://<PROXY_USER>:<PROXY_PASS>@<PROXY_HOST>:<PROXY_PORT>"
   export https_proxy="$http_proxy"; export HTTP_PROXY="$http_proxy"; export HTTPS_PROXY="$http_proxy"
   ```
3. **Cài package**: `torch`/`torchvision` (cu124), `transformers==4.44.2`, `llm2vec`, `accelerate`,
   `sentencepiece`, `safetensors`, `huggingface-hub`, `ftfy`, `regex`, `tqdm`, `pandas`, `numpy`,
   `pillow`, `matplotlib`, `requests`, `scikit-learn`, `scipy`.
   - `pip install git+https://github.com/openai/CLIP.git` **KHÔNG dùng được** — proxy công ty
     chặn CONNECT tới `github.com` (403), dù `huggingface.co`/`pypi.org` vẫn qua được bình thường.
     → dùng thay: `pip install clip-anytorch` (mirror PyPI, cùng API `import clip`).
   - `clip-anytorch` cần `pkg_resources` nhưng `setuptools` mới (82.0+) đã bỏ nó ra
     → phải `pip install "setuptools<81"` để downgrade.
4. **Checkpoint đã tải** về `/cm/shared/chautvh_second/Nhan_folder/ckpts/`:
   `Llama-3-8B-CC` (= `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`) và `ViT-L-336`
   (= `microsoft/LLM2CLIP-Openai-L-14-336`).
5. **Fix config Llama-CC bắt buộc** (nếu chưa patch sẽ crash lúc load model, KHÔNG cần làm lại
   nếu đã patch — check trước khi chạy lại):
   ```bash
   cd /cm/shared/chautvh_second/Nhan_folder/ckpts/Llama-3-8B-CC
   python -c "import json;c=json.load(open('config.json'));c['auto_map']={'AutoModel':'modeling_llama_encoder.LlamaEncoderModel'};json.dump(c,open('config.json','w'),indent=2);print('patched:', c['auto_map'])"
   python -c "from huggingface_hub import hf_hub_download as d;import shutil;shutil.copy(d('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','attn_mask_utils.py'),'attn_mask_utils.py');print('added')"
   ```
6. **Verify import thành công** (`imports OK [...]`, `torch 2.13`, `gpus 8`) — env sẵn sàng.

### Quyết định hiện tại (đang triển khai):
- Server có **8 GPU nhưng chỉ GPU 2 và 3 đang free** — giới hạn dùng 2 GPU này.
- User muốn **reproduce đúng kiểu LLM2CLIP thật** → dùng `--base_model llm2clip_released`
  (ViT-L/14-336 gốc, KHÔNG phải ViT-B/16 như vòng 1+2 trên 3060), `freeze_visual=False`
  (train full visual tower, đúng recipe stage-2 thật của LLM2CLIP).
- User muốn **dùng FULL ShareGPT4V** (toàn bộ ~1.246 triệu ảnh gốc, không chỉ subset 120k như
  kế hoạch ban đầu) — "chơi hết".

### Toán học đã giải thích cho user (tại sao cache không nổ tổ hợp):
Với `n` câu trong caption, `K=max_num_short_texts` (mặc định 4 trong precompute script):
- Nếu `n >= K`: `star_bar_long_text_split` với seed cố định trả về **đúng 1 partition = K
  chuỗi** (không phải `C(n-1,K-1)` cách chia). `longer_pool` = ghép cặp 2-trong-K =
  `C(K,2)` chuỗi. Tổng mỗi ảnh dài: ~`1(full)+1(short)+n(câu lẻ)+K+C(K,2)` ≈ 15-18 chuỗi.
- Nếu `n < K` (rất phổ biến ở ShareGPT4V-COCO vì caption ngắn hơn DOCCI nhiều): rơi vào fallback,
  nhiều chuỗi trùng nhau (full caption = caption_short = câu duy nhất = pos_pool...) → sau khi
  gom vào `set()` toàn cục chỉ còn 1-2 chuỗi thật khác nhau.
- Đo thực tế ở subset 120k ảnh: **281,740 chuỗi độc nhất** → tỉ lệ **~2.35 chuỗi/ảnh** (thấp vì
  đa số caption ShareGPT4V-COCO ngắn, rơi fallback).
- Ước lượng cho FULL 1.246M ảnh: `1,246,000 × 2.35 ≈ 2.93 triệu chuỗi`. Với throughput đã đo
  (~2,560 chuỗi/phút sau khi sort theo độ dài để giảm padding) → **~19 tiếng** precompute trên
  1 GPU. Cache float16 4096-dim: `2.93M × 4096 × 2 bytes ≈ 24 GB` trên đĩa.

### Lệnh kế tiếp cần chạy (chưa chạy, đang chờ user xác nhận/thực thi):

```bash
cd ~/Nhan_folder/train

# 1. Subset manifest = LẤY HẾT toàn bộ 1.246M ảnh
python make_sharegpt4v_subset.py --n 2000000

# 2. Precompute embedding FULL, CHỈ 1 GPU (script không multi-GPU), chọn GPU 2 (free), chạy nền
export CUDA_VISIBLE_DEVICES=2
nohup python precompute_llm2vec_embeddings.py --dataset sharegpt4v_coco \
  --batch_size 256 --no_4bit \
  --out /cm/shared/chautvh_second/Nhan_folder/work/sgpt4v_full_cache.pt \
  > precompute_full.log 2>&1 &
tail -f precompute_full.log   # ~19 tiếng, theo dõi tiến độ

# 3. SAU KHI bước 2 xong (log in "[done] wrote ... embeddings"), train trên GPU 2+3:
export CUDA_VISIBLE_DEVICES=2,3
CACHE=/cm/shared/chautvh_second/Nhan_folder/work/sgpt4v_full_cache.pt

# baseline — reproduce đúng ClipLoss gốc LLM2CLIP
torchrun --nproc_per_node=2 train.py --train_data sharegpt4v_coco --base_model llm2clip_released \
  --text_cache_path $CACHE --loss_mode vanilla \
  --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 --exp_name sgpt4v_full_baseline

# ours — loss Rethinking
torchrun --nproc_per_node=2 train.py --train_data sharegpt4v_coco --base_model llm2clip_released \
  --text_cache_path $CACHE --loss_mode full \
  --epochs 20 --batch_size 64 --lr 1e-5 --adapter_lr 1e-5 --exp_name sgpt4v_full_ours
```

**Điểm cần lưu ý / rủi ro chưa xử lý khi resume ở platform khác:**
- Kết luận vòng 2 cảnh báo: config `loss_w=(1,1,1,1)` mặc định có thể khiến "ours" **thua**
  vanilla ở batch lớn (đã thấy rõ ở batch=96 trên 3060). Batch dùng ở lệnh trên là 64/GPU × 2
  GPU = 128 effective (hoặc hơn nếu dùng cross-GPU negatives all_gather) — **lớn hơn cả batch=96
  từng thấy vanilla thắng** → cần cân nhắc tune `--w_pos/--w_longer/--w_hinge` hoặc warm-up
  trước khi kết luận, đừng vội lấy kết quả full-scale đầu tiên làm kết luận cuối.
  - **Đã bàn nhưng CHƯA quyết định:** có nên tune loss weight trước khi chạy full 19-tiếng
    precompute + train hay không.
- `precompute_llm2vec_embeddings.py` **hardcode GPU 0** trong process của nó — phải set
  `CUDA_VISIBLE_DEVICES` từ bên ngoài để chọn GPU vật lý, không có cờ `--gpu` trong script.
- Chưa chạy bước 1 (`make_sharegpt4v_subset.py --n 2000000`) trong session này — cần chạy và xác
  nhận số ảnh thực tế ghi ra manifest trước khi launch precompute 19-tiếng.
- File `download_sharegpt4v_coco_subset.py` (khác với `make_sharegpt4v_subset.py`) từng tải sẵn
  16k ảnh COCO+ShareGPT4V ở vòng 2 nhưng **chưa dùng để train** — không liên quan tới quyết định
  full-scale hiện tại (dùng thẳng path gốc `/cm/archive/luongtk/sharegpt4v/` trên server, không
  cần tải lại).

## 5. Đường dẫn quan trọng (server)

- Data gốc: `/cm/archive/luongtk/{docci,sharegpt4v}` (ShareGPT4V JSON gốc 1.246M items +
  ảnh tại `/cm/archive/luongtk/sharegpt4v/data/`).
- Checkpoint: `/cm/shared/chautvh_second/Nhan_folder/ckpts/{Llama-3-8B-CC,ViT-L-336}`.
- Work/cache/manifest/output: `/cm/shared/chautvh_second/Nhan_folder/work/`.
- Code: `~/Nhan_folder/{model,train,datasets_config}` (đã "kéo ra ngoài" thư mục
  `Rethinking-main/` cũ — path trong code là tuyệt đối, không phụ thuộc vị trí code).
- Checkpoint kết quả vòng 1+2 (trên máy 3060, tham chiếu so sánh):
  `train/docci_3060_50ep/`, `train/llm2clip_text_docci_50ep/ckpt/`,
  `train/llm2clip_text_vanilla_b16_50ep/ckpt/`, `train/cc3m_ours_b16/ckpt/`,
  `train/cc3m_vanilla_b16/ckpt/`, `train/docci_bigbatch_{vanilla,ours,ours_tuned}/`.
- Cache embedding cũ (3060, vòng 1+2, KHÔNG phải full-scale):
  `/home/tachau/docci_data/llm2vec_cc_cache.pt` (DOCCI, 281k chuỗi),
  `/home/tachau/dreamlip_cc3m/llm2vec_cc_cache.pt` (CC3M, 302k chuỗi).

## 6. Việc chưa làm (tổng hợp toàn bộ)

- Chưa chạy full ShareGPT4V (1.246M ảnh, GPU 2+3, `llm2clip_released`) — đây là việc đang dở,
  ngay bước đầu tiên (`make_sharegpt4v_subset.py --n 2000000`).
- Chưa quyết định có cần tune `loss_w` trước khi commit 19 tiếng precompute không.
- Chưa train visual ViT-L/14-336 gốc thành công ở quy mô lớn (mới chỉ zero-shot batch=8 trên
  3060: T2I 51.6% / I2T 73.2% trên DOCCI, chưa fine-tune).
- Chưa test cross-dataset/zero-shot generalization cho bản full-scale.
- Chưa làm semantic segmentation (mIoU/SAN).
- Chưa chạy nhiều seed để đo độ ổn định thống kê.
