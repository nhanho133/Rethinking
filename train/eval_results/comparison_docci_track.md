| Config | Flickr I2T | Flickr T2I | COCO I2T | COCO T2I | SG4V I2T | SG4V T2I | Urban1K I2T | Urban1K T2I | DOCCI I2T | DOCCI T2I |
|---|---|---|---|---|---|---|---|---|---|---|
| CLIP-L/14-336 baseline (cited, paper Table 2) *(cited)* | 87.7 | 67.0 | 58.0 | 37.1 | 86.2 | 84.0 | 72.8 | 57.0 | 67.4 | 65.7 |
| LLM2CLIP-15M, ViT-L/14@336 (cited, paper Table 2 -- more data than ours, see caveat) *(cited)* | 91.2 | 82.1 | 65.5 | 53.6 | 98.1 | 98.4 | 90.3 | 93.2 | 87.7 | 89.0 |
| vanilla_ViT-B16_DOCCI | 46.1 | 26.5 | 29.2 | 15.4 | 82.6 | 67.6 | 65.2 | 50.4 | 75.4 | 74.2 |
| ours_ViT-B16_DOCCI | 43.3 | 41.2 | 24.3 | 22.5 | 82.5 | 73.2 | 68.1 | 57.1 | 78.5 | 80.2 |
| released_ViT-L336_no-finetune | 94.1 | 82.2 | 66.9 | 53.8 | 99.4 | 98.9 | 96.2 | 96.6 | 89.9 | 90.7 |

**Lưu ý**: 2 hàng *(cited)* lấy thẳng từ paper Table 2, dùng nhiều data hơn (15M cặp) và không có bản '336px + CC3M-only' được công bố để so khớp tuyệt đối -- chỉ mang tính tham khảo, KHÔNG phải baseline kiểm soát chặt. Phép so sánh có kiểm soát thật sự (cùng data, cùng checkpoint xuất phát, cùng lịch train) là giữa các hàng đo được (không có *(cited)*) với nhau.
