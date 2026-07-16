# Swap only the text encoder to LLM2CLIP's frozen Llama-3-8B-CC, keep everything else from Rethinking

## Context

The project has been fine-tuning a custom LongCLIP ViT-B/16 model on DOCCI using a
proposed "Adaptive Multi-Positive + specificity hinge" loss (`model/model_longclip.py`,
`train/train.py`). A completed 50-epoch run reached T2I R@1 81.2% / I2T R@1 78.7% on the
DOCCI test split (`train/docci_3060_50ep/`).

**Revised scope (superseding an earlier draft of this plan):** the user wants to keep
*everything* about the Rethinking project as-is (loss design, DOCCI pipeline, ViT-B/16
visual backbone) and change exactly one thing: replace the text side with **LLM2CLIP's
released frozen text teacher, `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`** (the
"CC" = Caption-to-Caption contrastive-finetuned Llama-3-8B, i.e. LLM2CLIP's own "stage 2"
text tower), instead of switching to LLM2CLIP's own visual tower (ViT-L/14-336). This is a
deliberate improvement over the earlier draft: it keeps the visual backbone identical
(ViT-B/16) across the old and new runs, so any change in retrieval numbers can be
attributed to the frozen-LLM-text-teacher idea itself, not to also swapping in a bigger ViT.
The user also wants training to mirror **LLM2CLIP's own "stage 2" alignment recipe**
(freeze the LLM text tower completely; train the visual side against it), and wants to see
a **cheap frozen-both-sides check first** before committing to a full fine-tune.

## Key facts confirmed by reading the code

- `train/train.py::train_epoch` (L227-296): `pos_list` chunks (`star_bar_long_text_split`,
  fixed `num_seed=42`) are deterministic per image across epochs. `pos_longer_list`
  (`make_base_longer`, seeded by batch position `j`) varies epoch-to-epoch but is bounded —
  at most `C(5,2)=10` distinct pair combinations per image for `max_num_short_texts≤5` — so
  **all combinations are enumerable and cacheable** ahead of time.
- `model/model_longclip.py::forward` (L518-562):
  `forward(image, text_long, tokenized_caps=None, learnable_mps=False, text_pos=None,
  text_neg=None, text_pos_longer=None, use_hinge=True, use_sparsemax=False) ->
  (loss, L_long, L_pos, L_longer, L_hinge, logit_scale, tau)`. `_encode_grid` (L462),
  `_adaptive_mp_loss` (L494), and the hinge block (L546-552) operate purely on
  **normalized embeddings + cosine similarity** — no dependency on LongCLIP's tokenizer, so
  this logic is reusable unchanged regardless of what produces the embeddings.
  `learnable_mps` is a dead parameter (never read in the body) — irrelevant here.
- `train/train.py::test_epoch_ver5` (L491+) calls `model.encode_image(...)` /
  `model.encode_text(longclip.tokenize(...))` directly — the other integration seam.
- `train/datasets_config/docci.py` uses `clip.load(model_name)` only for `.preprocess` (the
  image transform) — since the visual backbone stays ViT-B/16, **this file needs no
  change at all**.
- Environment: `/home/tachau/rethink_venv/bin/pip list` has only `torch 2.12.1+cu130`,
  `torchvision`, `clip`. `transformers`, `llm2vec`, `accelerate`, `bitsandbytes`,
  `sentencepiece` are all missing — must install.
- GPU: single RTX 3060, ~11.6GB free of 12GB. Disk: ~382GB free (fine for the ~16GB
  Llama-3-8B-CC bf16 checkpoint; no LLM2CLIP visual checkpoint needed at all now, since we
  keep our own ViT-B/16 — this removes the ~2.3GB CLIP-L/14-336 download from the earlier
  draft entirely).
- LLM2CLIP's frozen text tower produces **4096-dim** embeddings via LLM2Vec (mean-pooling,
  max_length=512), fundamentally different dimensionality from CLIP ViT-B/16's **512-dim**
  joint space — a learned linear projection is required to bring them into the same space
  (this is exactly what LLM2CLIP's own `text_projection` does for its own visual tower; we
  need our own, sized for 4096→512, since we're keeping ViT-B/16, not adopting LLM2CLIP's
  own visual checkpoint/projection).

## Approach — two stages, cheap check before full commitment

Because the Llama-3-8B-CC teacher is **frozen** (never receives gradients) and the DOCCI
caption/sub-caption strings needed are enumerable in advance, precompute every needed text
embedding **once**, cache to disk, and never load the 8B model during either training stage.

### Stage 0 — Environment + one-time text embedding precompute (shared by both stages below)
- `pip install transformers accelerate bitsandbytes sentencepiece llm2vec` into
  `/home/tachau/rethink_venv`. **Bail-out**: if this conflicts with the existing
  `torch 2.12.1+cu130` pin, stop and reassess before touching the working LongCLIP env.
- Download `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` from Hugging Face
  (~16GB). Confirm actual license in the downloaded repo (GitHub README says MIT, HF model
  cards say Apache-2.0 — check directly, not blocking for internal experimentation).
- New file: `train/precompute_llm2vec_embeddings.py` — load Llama-3-8B-CC via
  `LLM2Vec.from_pretrained(...)`, **4-bit quantized** (`bitsandbytes`, `load_in_4bit=True`,
  ~5-6GB) to fit the 3060. Reuse `train.py`'s existing chunking functions directly
  (`split_into_detail_captions`, `star_bar_long_text_split`, `make_base_longer`) to
  enumerate every `text_long` / `pos` chunk / `pos_longer` combination needed across the
  full DOCCI train+test set, embed each, and cache as content-addressed
  `sha1(text) -> float16[4096]` (safetensors/.npz) plus a manifest.
  **Bail-out**: if 4-bit Llama-3-8B OOMs at batch size 1 on this GPU, stop and report back
  (do not silently fall back to CPU or drop captions).

### Stage 1 — Cheap check: freeze both sides, train only a linear projection
Goal: fast sanity check of whether Llama-3-8B-CC's frozen text space is even usable as an
alignment target for DOCCI, before spending GPU time on full fine-tuning.
- New file: `model/model_llm2clip_frozen.py` — wraps:
  - **Frozen** OpenAI CLIP ViT-B/16 (`clip.load("ViT-B/16")`, `requires_grad_(False)`) for
    `encode_image` — reuses the exact same pretrained weights `longclip.load_from_clip`
    normally starts fine-tuning from, just never updated here.
  - **Frozen** cached Llama-3-8B-CC embeddings (via a `TextEmbeddingCache.lookup(text)`,
    same cache format as Stage 0) for the text side.
  - One **trainable** `nn.Linear(4096, 512)` projecting cached text embeddings into CLIP's
    joint space (this is the only learnable parameter in this stage).
  - Same `forward()` signature/7-tuple as `model_longclip.py`, reusing `_encode_grid` /
    `_adaptive_mp_loss` / hinge logic verbatim — only `encode_text` differs (cache lookup +
    linear projection instead of tokenize + transformer).
- Wire into `train.py` behind a new `--base_model llm2clip_frozen_probe` branch; since only
  a 4096×512 linear layer trains, this can run with a **large batch size** (no backprop
  through ViT or LLM) and finish in well under an hour for many epochs — a fast go/no-go
  signal via `test_epoch_ver5`'s retrieval numbers before Stage 2.
- **Decision point**: if Stage 1 retrieval is reasonable (meaningfully above random chance,
  even if below the fully fine-tuned LongCLIP number), proceed to Stage 2. If it's
  degenerate (near-chance retrieval), stop and re-examine the projection design/cache
  correctness before investing in Stage 2.

### Stage 2 — Full fine-tune: unfreeze ViT-B/16, matches LLM2CLIP's own "stage 2" recipe
- New file: `model/model_llm2clip.py` — same as Stage 1's wrapper but `encode_image`'s
  CLIP ViT-B/16 is **trainable** (this mirrors LLM2CLIP's own methodology: freeze the LLM
  text teacher, train the visual tower + a text projection against it), still returning the
  same 7-tuple, still reusing the Adaptive-MP+hinge loss unchanged.
- Modify `train/train.py`:
  - `CLIP_Clean_Train.__init__` (L56-140): new branch for `args.base_model ==
    "llm2clip_text"` loading this wrapper + `TextEmbeddingCache` instead of
    `longclip.load_from_clip`. `--init_ckpt` can reuse Stage 1's trained projection layer as
    a warm start for the projection (optional).
  - `train_epoch` (L227-296) / `test_epoch_ver5` (L491+): swap
    `longclip.tokenize(...).to(device)` for `self.text_cache.lookup(...)` on this branch;
    all chunk-generation logic above it is untouched.
  - `train()` (~L629): add a `save_ckpt_list` arm for this branch (same periodic-save fix
    already applied for long DOCCI runs).
  - No changes needed to `datasets_config/docci.py` — visual preprocessing stays ViT-B/16.
- Batch size: since the visual tower is the same size/resolution as the existing LongCLIP
  run, batch 8 (the value already proven to avoid OOM) is the starting point — re-verify
  empirically since the frozen-text-cache lookup path has a different memory profile than
  live tokenization.
- Run the same 50-epoch, same DOCCI train/test split, same Adaptive-MP+hinge recipe as the
  completed LongCLIP run — now a **clean, ViT-size-controlled comparison**.

## Implementation notes (actual, as executed)

Environment quirks resolved while wiring up Stage 0 (all in
`train/precompute_llm2vec_embeddings.py` + the local checkpoint dir):
- transformers 4.44.2 + accelerate 1.14.0 vs the checkpoint's remote code required three
  compat fixes: (a) repointed the checkpoint's `config.json` `auto_map` from the McGill-NLP
  remote ref to its own bundled `modeling_llama_encoder.py`; (b) added the missing
  `attn_mask_utils.py` (faithful copy of McGill's, depends only on `AttentionMaskConverter`
  which exists in 4.44.2); (c) monkeypatched `PreTrainedModel.to` so a device-only `.to()`
  on a 4-bit bitsandbytes model is a safe no-op (the model is already correctly placed) —
  fixes both accelerate's `dispatch_model` and `LLM2Vec.encode`'s internal `self.to`.
- 4-bit Llama-3-8B-CC loads in ~4.65 GB VRAM on the 3060 — comfortable.
- Cache coverage: the eval loop needs the raw individual detail sentences (+ the empty pad
  string), and the `pos_longer` fallback for short (<K-sentence) captions can emit either
  concatenation order and self-pairs — both are now enumerated. Coverage was validated
  offline (0 misses over 51,200 generated chunk strings across 400 randomized batches)
  before committing to the full embed. Total unique strings: 281,740.
- Throughput: sorting strings by length (length-homogeneous batches -> minimal padding)
  took the embed rate from ~640/min to ~2560/min.

## Verification

- Stage 0: spot-check the cache — pick 5 random DOCCI images, regenerate their expected
  chunk strings via `train.py`'s own functions, confirm every hash exists in the cache with
  a finite, non-degenerate embedding.
- Stage 1: unit-check `forward()` on a synthetic batch (finite loss, correct 7-tuple shape)
  before a real run; then a short real run, confirming `test_epoch_ver5` retrieval is well
  above chance (chance ≈ 1/638 ≈ 0.16% R@1 on the DOCCI test set) as the go/no-go gate.
- Stage 2: 1-epoch smoke run on a small DOCCI subset (`max_items`) catches shape/cache-miss
  errors before the full 50-epoch commitment; monitor `nvidia-smi` stays under 12GB.
- Full run: monitor `Epoch N ▶ Avg Loss` decreasing without NaN and per-epoch `R@1` from
  `test_epoch_ver5`, same pattern used for the completed LongCLIP runs.
- Final comparison: LongCLIP-ViT-B/16 (existing: T2I R@1 81.2%, I2T R@1 78.7%, epoch 49,
  `train/docci_3060_50ep/`) vs Stage 1 (frozen-both probe) vs Stage 2 (full fine-tune,
  frozen-Llama-CC text teacher) — same visual backbone throughout, so the delta is
  attributable to the text-teacher swap.

---

## KẾT QUẢ THỰC NGHIỆM (cập nhật 2026-07-08)

### Tóm tắt những gì đã làm

1. **Tích hợp LLM2CLIP thật vào pipeline Rethinking** — không phải mô phỏng:
   - Text teacher: checkpoint gốc `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`
     (đã qua "stage 1" Caption-to-Caption Contrastive của họ), chạy 4-bit trên GPU 3060
     12GB (~4.65GB VRAM), đóng băng hoàn toàn.
   - Sửa 3 lỗi tương thích phiên bản `transformers`/`accelerate` để checkpoint gốc của
     Microsoft chạy được trên môi trường hiện có.
   - Visual: giữ nguyên **ViT-B/16** (không đổi sang ViT-L/14-336 của LLM2CLIP) để cô lập
     đúng một biến — text teacher — khi so sánh với LongCLIP.
   - Cơ chế **precompute + cache embedding**: vì Llama-8B đóng băng, tính trước toàn bộ
     embedding cần thiết (caption đầy đủ + mọi tổ hợp sub-caption có thể sinh ra), lưu cache,
     không cần load lại 8B model khi train chính thức.
   - Thêm cờ `--loss_mode vanilla|full` vào `train.py`: `vanilla` = đúng `ClipLoss` gốc của
     LLM2CLIP (chỉ `L_long`, contrastive 1-positive/ảnh); `full` = loss Rethinking đầy đủ
     (`L_long` + Adaptive Multi-Positive kiểu DreamLIP + specificity hinge kiểu SPECS).

2. **Phát hiện quan trọng về dữ liệu**: đọc code thật của LLM2CLIP (`llm2clip/data/download_dataset.sh`)
   thì bộ CC3M/CC12M/YFCC15M mà họ dùng để train stage 2 **không phải alt-text ngắn gốc**, mà
   là bản đã được **DreamLIP recaption thành caption dài** (`qidouxiong619/dreamlip_long_captions`
   trên HuggingFace). Đã tải một subset 15,117 ảnh từ chính bộ này (~66% tỷ lệ link còn sống)
   để làm bài test thứ hai, sát với data gốc LLM2CLIP nhất có thể trong điều kiện phần cứng hiện có.

3. **Giới hạn phần cứng đã xác nhận**: train visual ViT-L/14-336 (kể cả có grad-checkpointing)
   trên GPU 3060 12GB bị OOM do Adam optimizer states của 304M+274M params (~9GB) — không
   phải do activation. Vì vậy mọi run "vanilla vs ours" đều dùng ViT-B/16 (train được) thay vì
   ViT-L/14-336 gốc của LLM2CLIP.

### Bảng kết quả (Recall@1, DOCCI/CC3M test split)

| # | Cấu hình | Dataset | T2I R@1 | I2T R@1 |
|---|---|---|---|---|
| 1 | LongCLIP (text encoder tự train, không LLM2CLIP) | DOCCI | 81.2% | 78.7% |
| 2 | LLM2CLIP-text, loss **vanilla** (ClipLoss gốc) | DOCCI | 74.2% | 75.3% |
| 3 | LLM2CLIP-text, loss **ours** (DreamLIP+SPECS) | DOCCI | **80.1%** | **78.5%** |
| 4 | LLM2CLIP-text, loss **vanilla** (ClipLoss gốc) | DreamLIP-CC3M | 94.2% | 97.1% |
| 5 | LLM2CLIP-text, loss **ours** (DreamLIP+SPECS) | DreamLIP-CC3M | **97.4%** | **98.1%** |
| — | LLM2CLIP-ViT-L-336 released, zero-shot (batch 8) | DOCCI | 51.6% | 73.2% |

Tất cả các run 2-5 dùng chung platform: ViT-B/16 + Llama-3-8B-CC frozen + adapter linear
(4096→512), 50 epoch (DOCCI) / 30 epoch (CC3M), cùng optimizer/lr, cùng train-test split —
**chỉ khác đúng biến loss_mode**.

### Kết luận

1. **Loss Rethinking (DreamLIP Multi-Positive + SPECS hinge) thắng ClipLoss gốc của LLM2CLIP
   một cách nhất quán** — đúng ở cả 2 dataset độc lập (DOCCI và DreamLIP-CC3M — chính là data
   LLM2CLIP thật dùng), cả 4/4 chỉ số (T2I R@1, I2T R@1). Đây là bằng chứng thực nghiệm cho
   luận điểm: *ghép loss DreamLIP+SPECS vào stage-2 của LLM2CLIP giúp nó mạnh hơn chính nó*.
   - DOCCI: +5.9 điểm T2I, +3.2 điểm I2T.
   - CC3M: +3.2 điểm T2I, +1.0 điểm I2T (chênh nhỏ hơn do hiệu ứng trần — xem bên dưới).

2. **DOCCI là bài test phân biệt tốt hơn CC3M-subset** để so 2 loss: tập test CC3M subset
   (755 ảnh) quá đa dạng chủ đề nên cả 2 loss đều đạt >94% (hiệu ứng "chạm trần" — retrieval
   dễ tới mức khó phân biệt được đóng góp thật của loss). DOCCI (ảnh cùng chủ đề/phong cách,
   khó phân biệt hơn) cho khoảng cách rõ ràng hơn nhiều — nên dùng làm bằng chứng chính khi
   trình bày kết quả.

3. **(2) vs (3) so với (1)**: cả vanilla lẫn ours trên nền LLM2CLIP-text đều **chưa vượt qua**
   LongCLIP gốc (81.2/78.7) trên DOCCI — ours chỉ gần bằng (80.1/78.5). Nên khi báo cáo, không
   nên khẳng định "LLM2CLIP + loss của tôi > LongCLIP"; điều đã chứng minh được là hẹp hơn:
   *"loss của tôi > loss gốc của LLM2CLIP, trên cùng nền LLM2CLIP"*.

4. **Vanilla vẫn cải thiện chậm dù loss/batch gần như 0** (do batch_size=8 quá nhỏ khiến
   contrastive loss bão hòa sớm) — retrieval của vanilla vẫn tăng dần suốt 50 epoch nhờ
   gradient dư, không "chết" hẳn. Đây là giới hạn của thiết lập batch nhỏ trên GPU đơn, không
   phải do bản chất ClipLoss — cần lưu ý khi diễn giải, không nên nói ClipLoss "không học được gì".

### Sản phẩm / đường dẫn tham chiếu
- Code: `model/model_llm2clip.py` (wrapper LLM2CLIP-text), `train/precompute_llm2vec_embeddings.py`
  (precompute cache, hỗ trợ cả DOCCI và CC3M), `train/download_cc3m_subset.py` (tải subset CC3M),
  `train/datasets_config/dreamlip_cc3m.py`, cờ `--loss_mode`/`--adapter_type`/`--adapter_lr` trong `train.py`.
- Checkpoint cuối: `train/llm2clip_text_docci_50ep/ckpt/`, `train/llm2clip_text_vanilla_b16_50ep/ckpt/`,
  `train/cc3m_ours_b16/ckpt/`, `train/cc3m_vanilla_b16/ckpt/`.
- Cache embedding: `/home/tachau/docci_data/llm2vec_cc_cache.pt` (DOCCI, 281k chuỗi),
  `/home/tachau/dreamlip_cc3m/llm2vec_cc_cache.pt` (CC3M, 302k chuỗi).
- Data CC3M subset: `/home/tachau/dreamlip_cc3m/manifest.json` + `images/` (15,117 ảnh).

### Việc chưa làm / hạn chế cần nêu rõ nếu bị hỏi
- Chưa train được visual ViT-L/14-336 gốc của LLM2CLIP (giới hạn phần cứng 3060 12GB).
- Chưa test cross-dataset/zero-shot generalization (chỉ in-domain fine-tune + test).
- Adapter dùng bản **linear** đơn giản (không phải MLP 4-layer residual chuẩn của LLM2CLIP —
  đã thử bản MLP, bị overfit nặng trên dữ liệu nhỏ 15k ảnh, kết quả tệ hơn cả linear).
- CC3M subset chỉ 15k/2.87M ảnh gốc (do giới hạn băng thông/thời gian tải qua link web cũ).
- Mỗi cấu hình mới chạy 1 lần/1 seed — chưa có nhiều seed để đo độ ổn định thống kê.

---

## VÒNG 2 — Kiểm chứng ở batch lớn + Specificity Rate + Zero-shot (cập nhật 2026-07-08)

Sau vòng 1 (kết luận "ours thắng vanilla" ở batch=8), làm thêm 3 việc để kiểm chứng độ vững:
(a) chạy lại ở **batch lớn** (96 thay vì 8 — model llm2clip_text nhẹ, đo được batch tối đa ~128
trên 3060 vì Llama-8B đã precompute offline, chỉ ViT-B/16 + adapter nằm trong graph train),
(b) đo **Specificity Rate** (metric riêng của SPECS), (c) **zero-shot cross-dataset** trên
COCO val + Flickr30K (đúng protocol DreamLIP/LLM2CLIP tự báo cáo). Thêm cờ
`--w_pos/--w_longer/--w_hinge` để tinh chỉnh trọng số 4 loss.

### Phát hiện lõi: lợi thế của "ours" chủ yếu là hiện tượng batch nhỏ

**In-domain DOCCI, retrieval R@1:**

| Batch | Loss | T2I | I2T |
|---|---|---|---|
| 8 | vanilla | 74.2 | 75.3 |
| 8 | **ours** | **80.1** | **78.5** |
| 96 | **vanilla** | **77.6** | **79.5** |
| 96 | ours (1,1,1,1) | 73.4 | 74.8 |
| 96 | ours (loss_w 1,0.2,0.2,0.3, 20ep) | 74.3 | 75.5 |

-> Ở batch=8 ours thắng; ở batch=96 **vanilla thắng**, và hạ trọng số sub-loss chỉ cải thiện
chút ít (73.4->74.3), **không đảo ngược** được. Nguyên nhân đã xác định: `loss_w=(1,1,1,1)` chưa
bao giờ được tinh chỉnh; ở batch lớn `L_long` bão hòa ~0 rất nhanh còn `L_pos` vẫn cao (~0.05,
gấp ~50-100 lần), nên gradient bị `L_pos` hút, kéo model rời khỏi mục tiêu retrieval câu đầy đủ.

**Specificity Rate (SPECS metric):**

| | vanilla | ours | delta |
|---|---|---|---|
| DOCCI batch=8 | 87.1 | **98.8** | +11.7 |
| CC3M batch=8 | 69.9 | **99.8** | +29.8 |
| DOCCI batch=96 | 86.2 | 87.2 | +1.0 (gần như hòa) |

**Zero-shot COCO val (train DOCCI, không fine-tune), R@1:**

| | T2I | I2T |
|---|---|---|
| ours b8 | **21.4** | 16.2 |
| vanilla b8 | 14.7 | **20.0** |
| ours b96 | 17.3 | 16.1 |
| vanilla b96 | **18.3** | **22.2** |

**Zero-shot Flickr30K (train DOCCI), R@1:** (lưu ý: gallery ~31k ảnh toàn bộ Flickr, lớn hơn
protocol chuẩn 1k -> số tuyệt đối thấp hơn paper; nhưng so sánh tương đối giữa các ckpt hợp lệ
vì cùng gallery)

| | T2I | I2T |
|---|---|---|
| ours b8 | **13.9** | 8.4 |
| vanilla b8 | 7.4 | **11.4** |
| ours b96 | 8.7 | 9.1 |
| vanilla b96 | **11.3** | **13.6** |

### Kết luận vòng 2 (trung thực, đa chiều)

1. **Ở batch nhỏ (8):** ours có lợi thế nhưng là ĐÁNH ĐỔI, không phải thắng toàn diện:
   thắng in-domain retrieval + Specificity Rate + zero-shot **T2I**, nhưng **thua zero-shot I2T**.
2. **Ở batch lớn (96):** lợi thế của ours **biến mất gần hết** — vanilla thắng in-domain,
   Specificity Rate hòa, và **vanilla thắng cả 2 chiều zero-shot** trên cả COCO lẫn Flickr.
3. Mẫu hình "ours mạnh specificity/T2I, yếu I2T/generalization" **khớp chính xác cảnh báo trong
   paper SPECS**: *"strong specificity performance degrades retrieval/classification benchmarks"*.
4. **Hệ quả cho việc scale lên tài nguyên lớn:** không thể bê nguyên config batch nhỏ. Trước khi
   train quy mô lớn cần: (i) thiết kế lại cách cân bằng 4 loss (adaptive/uncertainty weighting
   thay vì hằng số tay, hoặc warm-up L_long trước rồi mới thêm sub-loss), (ii) chạy nhiều seed,
   (iii) cân nhắc mục tiêu thật là gì (specificity/caption-quality theo SPECS, hay retrieval
   thuần theo DreamLIP/LLM2CLIP — hai mục tiêu này đang xung đột trong kết quả).

### Sản phẩm bổ sung vòng 2
- Code: `train/eval_specificity_rate.py`, `train/eval_zeroshot_retrieval.py`,
  `train/download_sharegpt4v_coco_subset.py` (đã tải 16k ảnh COCO+ShareGPT4V caption, chưa train),
  `train/datasets_config/sharegpt4v_coco.py`, cờ `--grad_accum_steps`/`--w_pos`/`--w_longer`/`--w_hinge`.
- Eval data: `/home/tachau/eval_data/coco/` (val2017 5k), `/home/tachau/eval_data/flickr30k/` (31k),
  cache Llama cho caption của chúng đã build sẵn.
- Checkpoint batch=96: `train/docci_bigbatch_vanilla/`, `train/docci_bigbatch_ours/`,
  `train/docci_bigbatch_ours_tuned/`.

### Việc chưa làm (vòng 2)
- Chưa chạy ShareGPT4V (data 16k đã sẵn) — đang chờ quyết định cách cân bằng loss ở batch lớn
  trước khi train, tránh lặp lại kết quả "vanilla thắng" mà không kiểm soát được biến.
- Chưa làm zero-shot ImageNet classification, chưa làm semantic segmentation (mIoU/SAN — việc lớn).
- Zero-shot dùng 1 caption/ảnh (không phải 5-caption protocol chuẩn của COCO) -> số chỉ mang tính
  tương đối giữa các checkpoint, không so trực tiếp được với bảng trong paper.
