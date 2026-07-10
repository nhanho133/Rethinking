# Plan train Stage 2 hoàn chỉnh: Rethinking loss + LLM2CLIP trên server 8×H100

> Mục tiêu: chạy đúng "stage 2 của LLM2CLIP" (LLM text teacher đóng băng, train visual + adapter)
> NHƯNG thay ClipLoss gốc bằng loss Rethinking (L_long + Adaptive Multi-Positive + SPECS hinge),
> ở **quy mô thật** (data lớn + batch lớn + multi-GPU) mà máy 3060 không làm được.

---

## 0. Bối cảnh — vì sao plan này (đọc trước khi làm)

Các thí nghiệm nhỏ trên 3060 (DOCCI/CC3M/ShareGPT4V 15k ảnh, batch 8-96) cho kết luận nhất quán
qua 4 metric (in-domain retrieval, Specificity Rate, zero-shot COCO/Flickr, zero-shot ImageNetV2):

- **batch nhỏ (8):** loss Rethinking (ours) thắng ClipLoss (vanilla).
- **batch lớn (96):** lợi thế biến mất, vanilla thắng.

**Nguyên nhân đã xác định:** ở batch lớn, `L_long` (contrastive câu đầy đủ) bão hòa ~0 rất nhanh,
còn `L_pos` vẫn cao → với trọng số `(1,1,1,1)`, gradient bị `L_pos` hút, kéo model rời mục tiêu
retrieval. NHƯNG thí nghiệm "batch lớn" của tôi bị **lẫn biến**: batch tăng nhưng data vẫn bé
(15k) nên model học thuộc → chưa tách được "do batch" hay "do data nhỏ".

→ **Câu hỏi plan này trả lời:** ở regime THẬT của LLM2CLIP (data hàng trăm nghìn-triệu ảnh đa
dạng + batch 512 + 8 GPU), khi `L_long` KHÔNG bão hòa được (không học thuộc nổi), loss Rethinking
có thắng ClipLoss không? Đây mới là phép thử quyết định.

---

## 1. Yêu cầu môi trường

- 8× H100 80GB (tổng ~640GB VRAM) → thoải mái train full ViT-L/14-336 + batch lớn + giữ Llama-8B
  bf16 trong lúc precompute (không cần 4-bit như trên 3060).
- ~500GB-1TB đĩa trống (checkpoint 8B ~16GB + ViT-L ~2.3GB + images + embedding cache).
- Python venv giống máy hiện tại (`torch 2.12.1+cu13x`, `transformers 4.44.2`, `llm2vec 0.2.3`,
  `accelerate`, `bitsandbytes`, `matplotlib`, `clip`).

**Lưu ý path:** mọi path dưới đây theo máy 3060 hiện tại (`/home/tachau/...`). Trên server H100,
đặt biến `ROOT=/path/on/h100` và thay tương ứng. Nên set env cho gọn:
```bash
export LLM_CKPT=$ROOT/ckpts/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned
export VITL_CKPT=$ROOT/ckpts/LLM2CLIP-Openai-L-14-336
export DATA=$ROOT/data
```

---

## 2. Phase 0 — Setup môi trường (0.5 ngày)

```bash
# 2.1 clone repo (đã có toàn bộ code từ máy 3060) + venv
cd $ROOT && git clone <repo> Rethinking_project   # hoặc rsync từ máy 3060
python -m venv rethink_venv && source rethink_venv/bin/activate
pip install torch torchvision transformers==4.44.2 llm2vec accelerate bitsandbytes \
            sentencepiece matplotlib ftfy regex tqdm pandas pillow requests
pip install git+https://github.com/openai/CLIP.git   # package `clip`

# 2.2 tải checkpoint LLM2CLIP
python -c "from huggingface_hub import snapshot_download as d; \
  d('microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned', local_dir='$LLM_CKPT'); \
  d('microsoft/LLM2CLIP-Openai-L-14-336', local_dir='$VITL_CKPT')"

# 2.3 QUAN TRỌNG: fix tương thích transformers 4.44.2 cho checkpoint Llama-CC
#     (đã làm sẵn trên máy 3060 — copy y nguyên 2 thay đổi này vào $LLM_CKPT):
#   (a) sửa config.json: auto_map -> "modeling_llama_encoder.LlamaEncoderModel" (bỏ prefix McGill-NLP--)
#   (b) thêm file attn_mask_utils.py (bản McGill gốc) vào thư mục checkpoint
#   Chi tiết ở .claude/plans/https-github-com-microsoft-llm2clip-git-zesty-pascal.md mục "Stage 0".

# 2.4 smoke test import
python -c "import torch,transformers,llm2vec,clip; print(torch.cuda.device_count(),'GPUs')"
```

---

## 3. Phase 1 — Dataset (1-2 ngày, phần lâu nhất là tải ảnh)

### Chọn dataset (khuyến nghị chạy theo thứ tự)

| Ưu tiên | Dataset | Ảnh | Ổn định tải | Ghi chú |
|---|---|---|---|---|
| **A (chính)** | ShareGPT4V-COCO | ~118k (COCO train2017) | ✅ Cao (host COCO chính thức) | Caption ShareCaptioner (dài ~145 từ). Pipeline đã có sẵn. |
| **B (trung thành nhất)** | DreamLIP-CC3M | ~2.87M (URL web) | ⚠️ ~66% link sống | Đúng data LLM2CLIP dùng. Cần `img2dataset`. |
| C (scale lớn) | DreamLIP-CC12M | ~10M | ⚠️ link rot | Chỉ làm nếu A/B đã cho tín hiệu tốt. |

**Khuyến nghị:** bắt đầu **A ở scale ~100k-118k** (đủ lớn để model KHÔNG học thuộc — đây chính là
điều tách biến mà thí nghiệm 15k không làm được), rồi mới lên B/C nếu cần.

### Code tải (đã có sẵn, chỉ tăng `--target`)
```bash
# ShareGPT4V-COCO: tải caption JSON gốc + ảnh COCO
python -c "from huggingface_hub import hf_hub_download as d; \
  d('Lin-Chen/ShareGPT4V','share-captioner_coco_lcs_sam_1246k_1107.json',repo_type='dataset',local_dir='$DATA/sharegpt4v')"
# sửa path trong train/download_sharegpt4v_coco_subset.py (JSON, out_dir, manifest) -> $DATA/sharegpt4v
python train/download_sharegpt4v_coco_subset.py --target 118000 --workers 128

# HOẶC DreamLIP-CC3M (trung thành LLM2CLIP): tải CSV rồi img2dataset
bash llm2clip/data/download_dataset.sh cc3m     # tải CSV từ qidouxiong619/dreamlip_long_captions
# (hoặc dùng train/download_cc3m_subset.py --target 500000 đã viết, đơn giản hơn img2dataset)
```

### Sửa path dataset config
Trong `train/datasets_config/sharegpt4v_coco.py` (và `dreamlip_cc3m.py`): sửa `MANIFEST_PATH` và
`IMAGE_ROOT` trỏ về `$DATA/...` trên H100.

---

## 4. Phase 2 — Precompute text embedding Llama-CC (0.5-1 ngày)

Trên H100, load Llama-8B **bf16** (không cần 4-bit). Sửa `train/precompute_llm2vec_embeddings.py`:
- `load_l2v(quant_4bit=False)` khi chạy trên H100 (bf16 nhanh hơn nhiều 4-bit).
- Đường dẫn `LLM_PATH` -> `$LLM_CKPT`.

```bash
python train/precompute_llm2vec_embeddings.py --dataset sharegpt4v_coco \
  --max_num_short_texts 4 --batch_size 256 --no_4bit \
  --out $DATA/sharegpt4v/llm2vec_cc_cache.pt
```

**⚠️ Cảnh báo scale (phải xử lý nếu data > ~300k ảnh):** số chuỗi cần embed ≈ 19× số ảnh
(caption + mọi tổ hợp sub-caption). 15k ảnh → 280k chuỗi → cache 2.3GB (nằm vừa RAM). Nhưng
**1M ảnh → ~19M chuỗi → cache ~150GB, KHÔNG nạp nổi vào RAM dạng dict.** Với data lớn cần 1 trong 2:
1. **Shard cache** ra nhiều file (theo hash prefix), `TextEmbeddingCache.lookup` load lazy theo shard.
2. HOẶC giảm liệt kê: chỉ cache đúng các chuỗi 1 seed sinh ra thay vì mọi tổ hợp (chấp nhận ràng
   buộc seed cố định, không shuffle `pos_longer` theo batch position). → đơn giản hơn nhiều.
   → **Khuyến nghị (2)** cho scale lớn: sửa `make_base_longer` dùng seed cố định theo image-id
   (không theo vị trí batch `j`), rồi chỉ cache 1 tổ hợp/slot. Giảm 19× → ~4-5× số ảnh.

---

## 5. Phase 3 — SỬA CODE (phần quan trọng nhất, ~2-3 ngày)

### 5.1 ⭐ CRITICAL: Cross-GPU all-gather cho contrastive loss
**Đây là fix bắt buộc để multi-GPU đúng.** Hiện tại `model/model_llm2clip.py::forward` tính
contrastive **chỉ trong batch local mỗi GPU** → 8 GPU × batch B = mỗi GPU vẫn chỉ so B negative,
KHÔNG phải 8B. Với phát hiện "số negative quyết định kết quả", đây là điểm phải sửa.

Cần gather `v` (image feat) và `tl`/`Tf` (text feat) qua tất cả GPU trước khi tính logits, dùng
`torch.distributed.nn.all_gather` (có backward). Template chuẩn: `llm2clip/eva_clip/loss.py::gather_features`
(dùng `gather_with_grad` + `local_loss`).

Sửa trong `model/model_llm2clip.py`:
```python
import torch.distributed as dist
import torch.distributed.nn

def _gather(x):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return torch.cat(torch.distributed.nn.all_gather(x), dim=0)
    return x

# L_long: query local vs key global
def forward(...):
    v = encode_image(image); v = v / v.norm(...)          # [B, D] local
    tl = encode_text(text_long); tl = tl / tl.norm(...)   # [B, D] local
    v_all, tl_all = _gather(v), _gather(tl)               # [B*W, D] global
    rank = dist.get_rank() if dist.is_initialized() else 0
    labels = torch.arange(B, device=device) + rank * B
    logits_i = logit_scale * (v @ tl_all.t())            # local img vs global txt
    logits_t = logit_scale * (tl @ v_all.t())            # local txt vs global img
    L_long = 0.5*(F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels))
```
Và tương tự trong `_adaptive_mp_loss`: gather `v` và `Tf[:, j, :]` để mỗi slot j so global negatives.
(Đây là phần tốn công nhất — test kỹ shape + labels offset theo rank.)

### 5.2 Unfreeze ViT-L trong LLM2CLIPReleasedTeacher
Hiện `LLM2CLIPReleasedTeacher.__init__` đóng băng `vision_model` (vì 3060 OOM). Trên H100 mở khóa:
thêm tham số `freeze_visual=False`, chỉ freeze khi được yêu cầu. Optimizer param-group (đã có sẵn
logic trong train.py) sẽ tự nhận visual làm nhóm lr riêng.
- Cần thêm `--llm2clip_model_path $VITL_CKPT` và dùng `--base_model llm2clip_released`.
- **Data preprocess phải là 336px** cho ViT-L: sửa nhánh preprocess trong train.py
  (`llm2clip_released` -> `clip.load("ViT-L/14@336px").preprocess`) — đã có sẵn.

### 5.3 Xử lý mất cân bằng loss ở batch lớn (từ phát hiện chính)
Thêm **warm-up L_long**: N step đầu chỉ train `L_long` (đặt `w_pos=w_longer=w_hinge=0`), sau đó ramp
lên. Cách rẻ nhất: dùng cờ đã có `--w_pos/--w_longer/--w_hinge` + thêm 1 arg `--sub_loss_warmup_steps`.
Trong `train_epoch`, nếu `step < warmup`: ép `_text_pos=_text_pos_longer=None, use_hinge=False`.
→ Đảm bảo `L_long` được ưu tiên trước khi sub-loss can thiệp (tránh đúng vấn đề batch=96 đã thấy).

### 5.4 DDP checkpoint chỉ ở rank 0
Kiểm tra `train()`: `torch.save(self.model.state_dict())` phải bọc `if self.is_main:` (tránh 8 GPU
ghi đè). DDP-wrapped model → lưu `self.model.module.state_dict()`. (Kiểm tra + sửa nếu cần.)

### 5.5 Ablation matrix qua cờ (đã có sẵn hết)
`--loss_mode vanilla|full`, `--without_hinge_loss`, `--weight_hinge_loss X`, `--w_pos/--w_longer/--w_hinge`,
`--adapter_type linear|mlp`. Trên data lớn nên thử lại `--adapter_type mlp` (bản 270M chuẩn LLM2CLIP,
bị overfit ở 15k nhưng ở scale lớn nhiều khả năng thắng linear).

---

## 6. Phase 4 — Chạy train (torchrun 8 GPU)

```bash
# Ví dụ: ours (full loss) trên ShareGPT4V, ViT-L unfrozen, batch 64/GPU = 512 global
torchrun --nproc_per_node=8 train/train.py \
  --train_data sharegpt4v_coco --base_model llm2clip_released \
  --llm2clip_model_path $VITL_CKPT \
  --text_cache_path $DATA/sharegpt4v/llm2vec_cc_cache.pt \
  --adapter_type mlp --loss_mode full \
  --max_num_short_texts 4 --epochs 20 --batch_size 64 \
  --lr 1e-5 --adapter_lr 1e-5 --warmup_length 500 \
  --sub_loss_warmup_steps 1000 \
  --w_pos 1.0 --w_longer 1.0 --w_hinge 1.0 \
  --exp_name h100_ours_sgpt4v
```
Hyperparam gợi ý (theo run.sh gốc LLM2CLIP): lr 1e-5 cho cả visual+adapter, wd 0.05, 20 epoch,
grad-clip 5.0, fp16/bf16. Batch 64/GPU với ViT-L-336 trên H100 80GB thoải mái (có thể lên 128/GPU).

### Bảng ablation cần chạy (mỗi dòng 1 run, giữ mọi thứ khác giống nhau)
| # | loss_mode | Mục đích |
|---|---|---|
| 1 | `vanilla` | Baseline = LLM2CLIP gốc (ClipLoss) |
| 2 | `full` | Ours = + DreamLIP MP + SPECS hinge |
| 3 | `full --without_hinge_loss` | Ablation: chỉ MP, không hinge |
| 4 | `full --w_pos 0 --w_longer 0` | Ablation: chỉ hinge, không MP |
→ So (2) vs (1) trả lời câu hỏi chính; (3),(4) tách đóng góp từng phần.

---

## 7. Phase 5 — Evaluation (dùng lại code đã viết)

```bash
# Zero-shot cross-dataset (protocol chuẩn của DreamLIP/LLM2CLIP)
python train/eval_zeroshot_retrieval.py --dataset coco --ckpt <ckpt>       # + --released cho ref
python train/eval_zeroshot_retrieval.py --dataset flickr30k --ckpt <ckpt>
python train/eval_zeroshot_imagenet.py --ckpt <ckpt>                       # ImageNetV2 top1/5
python train/eval_specificity_rate.py --dataset sharegpt4v_coco --ckpt <ckpt> --cache ...
# Vẽ loss
python train/plot_losses.py <exp>/<exp>.log --out <exp>/loss.png
python train/plot_losses.py --compare vanilla.log:vanilla ours.log:ours --out cmp.png
```
Lưu ý: eval scripts đang hardcode ViT-B/16 + cache DOCCI/CC3M paths → sửa cho ViT-L-336 + cache
ShareGPT4V (thêm `--released`-style branch, hoặc tham số hóa preprocess & cache path).
Ref: chạy `--released` (LLM2CLIP-L-336 gốc, chưa train) để có mốc "LLM2CLIP thật mạnh cỡ nào".

---

## 8. Checklist thực thi (theo thứ tự)

- [ ] Phase 0: venv + tải 2 checkpoint + fix config.json/attn_mask_utils cho Llama-CC + smoke import
- [ ] Phase 1: tải ShareGPT4V-COCO ~118k (hoặc DreamLIP-CC3M), sửa path trong datasets_config
- [ ] Phase 2: precompute embedding bf16; **xử lý sharding/giảm-liệt-kê nếu >300k ảnh**
- [ ] Phase 3.1: ⭐ thêm all-gather cross-GPU vào forward (L_long + _adaptive_mp_loss) — test shape kỹ
- [ ] Phase 3.2: unfreeze ViT-L (freeze_visual=False) + preprocess 336px
- [ ] Phase 3.3: thêm `--sub_loss_warmup_steps` (warm-up L_long)
- [ ] Phase 3.4: checkpoint chỉ rank 0, lưu `.module.state_dict()`
- [ ] Phase 4: smoke 1 GPU 1 epoch subset → rồi torchrun 8 GPU đủ epoch; chạy 4 dòng ablation
- [ ] Phase 5: eval zero-shot COCO/Flickr/ImageNet + Specificity + plot; so (2) vs (1)

## 9. Rủi ro / điều cần canh
- **all-gather sai labels-offset theo rank** → loss vô nghĩa. Test bằng 2-GPU trước, kiểm tra
  loss ~ log(global_batch) lúc epoch 0.
- **Precompute cache quá lớn** (điểm nghẽn đĩa/RAM ở scale triệu ảnh) — xử lý ở Phase 2.
- **`--adapter_type mlp` (270M)** cần lr/wd hợp lý; ở scale lớn mới nên dùng (overfit ở data nhỏ).
- Vẫn nên chạy **≥2 seed** cho 2 dòng chính (vanilla, ours) vì kết quả 3060 cho thấy độ nhạy cao.

## 10. Tham chiếu file đã có (mang từ máy 3060 lên)
- `model/model_llm2clip.py` — wrapper (LLM2CLIPTextTeacher / LLM2CLIPReleasedTeacher / TextEmbeddingCache / TextAdapter linear|mlp)
- `train/train.py` — nhánh llm2clip_* + DDP wrap + mọi cờ ablation + param-group lr
- `train/precompute_llm2vec_embeddings.py` — `--dataset docci|dreamlip_cc3m|sharegpt4v_coco`, `--no_4bit`
- `train/datasets_config/{docci,dreamlip_cc3m,sharegpt4v_coco}.py` + `datasets_config.py`
- `train/download_{cc3m,sharegpt4v_coco}_subset.py`
- `train/eval_{zeroshot_retrieval,zeroshot_imagenet,specificity_rate}.py`, `plot_losses.py`, `test_case_1_split.py`
- Báo cáo kết quả 3060: `.claude/plans/https-github-com-microsoft-llm2clip-git-zesty-pascal.md`
