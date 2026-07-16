# Deploy 5-benchmark zero-shot eval pipeline lên server

Tài liệu này mô tả toàn bộ hạ tầng eval-theo-paper (Flickr30K/COCO/Urban1K/SG4V-1K/DOCCI) đã
xây xong và verify tại máy local, để bạn đưa lên server và chạy tiếp ở quy mô lớn hơn (checkpoint
ViT-L thật, ShareGPT4V-full).

## 0. Bối cảnh — tại sao có tài liệu này

Yêu cầu gốc: so sánh **LLM2CLIP + loss Rethinking** (`ours`) vs **LLM2CLIP + loss gốc**
(`vanilla`) theo đúng benchmark paper LLM2CLIP (arXiv:2411.04997 Table 2) dùng — Flickr30K,
COCO, ShareGPT4V-1K, Urban1K, DOCCI — thay vì chỉ so trên tập test tự tách (in-domain, gallery
quá lớn, không cùng thang đo với paper).

**Phát hiện quan trọng giữa chừng**: 2 checkpoint đầu tiên được eval (`llm2clip_text_docci_50ep`,
`llm2clip_text_vanilla_b16_50ep`) dùng **CLIP gốc OpenAI ViT-B/16** (`clip.load("ViT-B/16")` —
xem `model/model_llm2clip.py:154`), KHÔNG phải weight LLM2CLIP thật của Microsoft. Đây là platform
tự dựng để test loss rẻ tiền, không đại diện cho "LLM2CLIP". Checkpoint dùng **weight LLM2CLIP
thật** (`LLM2CLIPReleasedTeacher`, ViT-L/14@336) chỉ có bản `vanilla`
(`llm2clip_released_vanilla`) — bản `ours` tương ứng chưa từng được train. Đang train bù local
(xem mục 6).

## 1. Kiến trúc file — 7 file đã sửa/tạo mới

| File | Vai trò | Trạng thái |
|---|---|---|
| `train/eval_paper_benchmarks.py` | **File chính**: load checkpoint (2 kiểu kiến trúc), chạy R@1/5/10 T2I+I2T trên 1 hoặc cả 5 benchmark, multi-caption-safe, lưu tăng dần theo từng benchmark (resumable) | Mới, đã test qua 3 checkpoint |
| `train/precompute_llm2vec_embeddings.py` | Đã có sẵn từ trước (part-file, resumable). Thêm mới: `--eval_mode` + 4 hàm loader (`load_flickr30k_eval_captions`, `load_coco_eval_captions`, `load_urban1k_eval_captions`, `load_sg4v1k_eval_captions`) — chỉ collect caption thô, KHÔNG chạy qua `star_bar_long_text_split`/`make_base_longer` (logic đó chỉ dành cho augmentation lúc train) | Đã test cả 4 dataset |
| `train/assemble_paper_comparison.py` | Gộp nhiều file `--out_json` (từ eval_paper_benchmarks.py) + số cited từ paper thành 1 bảng markdown/csv | Mới, đã test |
| `train/paper_reference_numbers.json` | Số Table 2 của paper (verify qua WebFetch arXiv:2411.04997), tránh gõ tay lặp lại sai | Mới |
| `train/eval_zeroshot_retrieval.py` | Sửa 1 bug: `load_flickr30k()` thiếu filter `split=="test"` — đọc nhầm cả 31k ảnh (train+val+test) thay vì đúng 1K test theo Karpathy split | Đã fix |
| `train/urban1k.py` | Tham số hóa `clip.load(model_name)` (trước hardcode `"ViT-B/16"`, giờ dùng được với `"ViT-L/14@336px"` cho checkpoint released). Thêm `use_full_split=True` để lấy đủ 1000 ảnh (bản gốc chỉ chia 80/20 train/val) | Đã fix |
| `train/datasets_config/docci.py` | Thêm biến môi trường `DOCCI_DATA_ROOT` override (mặc định vẫn là path server `/cm/archive/luongtk/docci/`, không phá gì nếu không set biến) | Đã fix, đã test local |

## 2. Cách đưa 7 file lên server

Đã đóng gói sẵn thành **1 script duy nhất** tại máy local:
`/home/tachau/Rethinking_project/train/DEPLOY_TO_SERVER.sh` (1433 dòng, ghi đè/tạo cả 7 file).

**Copy toàn bộ nội dung file đó, paste vào terminal server** (đứng đúng thư mục `train/` trên
server, ví dụ `~/Nhan_folder/train`), rồi Enter. Script tự in `--- writing <tên file> ---` cho
từng file và kết thúc bằng `DONE: all 7 files written`.

Nếu bạn muốn tôi in nguyên nội dung `DEPLOY_TO_SERVER.sh` ra chat để copy (dài ~1400 dòng), báo
tôi — hoặc nếu 2 máy (local Claude Code này và server) có đường mạng thấy nhau, dùng
`scp`/`rsync` trực tiếp file đó lên sẽ nhanh hơn nhiều so với paste tay.

## 3. Data cần có trên server (Flickr30K/COCO/Urban1K)

SG4V-1K và DOCCI dùng lại data/JSON đã có sẵn trên server (`/cm/archive/luongtk/...`), không cần
tải gì thêm. 3 benchmark còn lại:

### Urban1K — tải qua HuggingFace (đã verify hoạt động)
```bash
export http_proxy="http://<PROXY_USER>:<PROXY_PASS>@<PROXY_HOST>:<PROXY_PORT>"
export https_proxy="$http_proxy"; export HTTP_PROXY="$http_proxy"; export HTTPS_PROXY="$http_proxy"

mkdir -p <work_dir>/eval_data/urban1k
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='BeichenZhang/Urban1k', filename='Urban1k.zip', repo_type='dataset', local_dir='<work_dir>/eval_data/urban1k')
print('downloaded:', p)
"
cd <work_dir>/eval_data/urban1k && unzip -q Urban1k.zip
# kết quả: <work_dir>/eval_data/urban1k/Urban1k/{image,caption}/  (1000 ảnh + 1000 caption)
```

### COCO val2017 — URL chính thức
```bash
mkdir -p <work_dir>/eval_data/coco
cd <work_dir>/eval_data/coco
curl -O http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
# cần: <work_dir>/eval_data/coco/val2017/*.jpg + <work_dir>/eval_data/coco/annotations/captions_val2017.json
```

### Flickr30K — nguồn đã verify: `nlphuji/flickr30k` trên HuggingFace
File `flickr_annotations_30k.csv` (khớp đúng tên/format đã dùng ở máy local) nằm trong dataset
này. Cách lấy đúng layout (CSV + thư mục `flickr30k-images/`) cần bạn kiểm chứng vì tôi chưa test
đường này trên server — dataset HF này trả về qua `datasets.load_dataset`, ảnh nhúng trong
parquet/arrow, không phải sẵn file `.jpg` rời như local đang có. 2 hướng:
1. **Đơn giản nhất nếu 2 máy nối được nhau**: `rsync`/`scp` thẳng `/home/tachau/eval_data/flickr30k/`
   (đã có sẵn, đúng layout, đã verify 1000 ảnh test split) từ máy local này sang server.
2. Nếu không nối được: tải qua `datasets.load_dataset("nlphuji/flickr30k")`, lọc `split=="test"`,
   rồi tự ghi từng ảnh ra `.jpg` theo đúng tên cột `filename` + copy cột `raw`/`split`/`img_id`
   thành `flickr_annotations_30k.csv` — tôi có thể viết script "materialize" này nếu bạn chọn
   hướng 2, báo tôi.

## 4. Build cache embedding cho 5 benchmark (giống hệt cách đã làm ở local)

```bash
cd ~/Nhan_folder/train  # hoặc đúng thư mục train/ trên server
mkdir -p <work_dir>/eval_data/eval_caches

# 4 benchmark dùng --eval_mode (script mới)
for ds in flickr30k coco urban1k; do
  python precompute_llm2vec_embeddings.py --dataset $ds --eval_mode \
    --batch_size 256 --part_size 50000 --no_4bit \
    --out_dir <work_dir>/eval_data/eval_caches/$ds \
    --eval_data_root <work_dir>/eval_data
done

# SG4V-1K: server có đủ file gốc (full 1.246M JSON + ảnh) nên dùng full_json mode
# (khác máy local phải dùng --sg4v_source manifest vì thiếu ảnh gốc)
python precompute_llm2vec_embeddings.py --dataset sg4v1k --eval_mode \
  --batch_size 256 --part_size 50000 --no_4bit \
  --out_dir <work_dir>/eval_data/eval_caches/sg4v1k \
  --sg4v_source full_json

# DOCCI: KHÔNG cần build — nếu cache DOCCI training (part-file) đã có sẵn trên server
# (từ track in-domain), symlink thẳng vào, y hệt cách làm ở local:
ln -s <đường_dẫn_cache_docci_training_có_sẵn> <work_dir>/eval_data/eval_caches/docci
```
Lưu ý: trên server (A100 80GB) dùng `--no_4bit` (bf16) được luôn, không cần 4-bit như máy local
3060 12GB.

## 5. Chạy eval

```bash
python eval_paper_benchmarks.py --benchmark all --released \
  --cache_dir <work_dir>/eval_data/eval_caches \
  --eval_data_root <work_dir>/eval_data \
  --docci_json /cm/archive/luongtk/docci/captioner_docci.json \
  --docci_image_root /cm/archive/luongtk/docci \
  --sg4v_source full_json \
  --sharegpt4v_full_json /cm/archive/luongtk/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json \
  --sharegpt4v_image_root /cm/archive/luongtk/sharegpt4v/data \
  --llm2clip_released_ckpt <ckpt_dir>/ViT-L-336 \
  --batch_size 64 \
  --out_json eval_results/released_server.json
```
Chạy `--released` (checkpoint gốc, không train) TRƯỚC làm sanity-check — máy local đã làm bước
này, khớp sát paper (lệch <3 điểm % ở 4/5 benchmark). Sau đó đổi `--ckpt <path> --arch {text,released}`
cho từng checkpoint thật cần so (ShareGPT4V-full vanilla/ours khi train xong, hoặc
`llm2clip_released_vanilla`/`llm2clip_released_ours` nếu transfer từ local lên).

## 6. Trạng thái đang chạy ở máy local (song song, không cần chờ)

`llm2clip_released_ours` — checkpoint **LLM2CLIP thật** (ViT-L/14@336 đóng băng) + loss Rethinking,
DOCCI, 20 epoch — đang train tại local (PID còn sống, theo dõi nền), ETA ~11-12 tiếng. Khi xong sẽ
eval qua đúng harness này, ghép với `llm2clip_released_vanilla` đã có → **cặp so sánh LLM2CLIP
thật đầu tiên hoàn chỉnh**. Không phụ thuộc vào việc deploy server — 2 việc chạy độc lập.

## 7. Việc chưa làm / cần bạn quyết

- Flickr30K trên server: chưa xác định được cách lấy chắc chắn (mục 3) — cần bạn chọn rsync hay
  materialize-từ-HF.
- Chưa biết checkpoint ShareGPT4V-full/DOCCI-server (track in-domain) đã train xong chưa — nếu
  xong, dùng đúng lệnh mục 5 để eval.
- Nếu muốn so cả `vanilla` ViT-L @336 lẫn `ours` ViT-L @336 trên SERVER (không chỉ local, quy mô
  DOCCI nhỏ) thì cần transfer checkpoint `llm2clip_released_vanilla`/`llm2clip_released_ours`
  (mỗi cái vài trăm MB — ViT-L visual đóng băng nên chỉ lưu phần adapter+logit_scale, không lưu
  cả ViT-L) từ local lên server, hoặc train lại thẳng trên server nếu muốn quy mô lớn hơn DOCCI.
