"""
Assembles the final comparison table: paper-cited reference rows (from
paper_reference_numbers.json) + our own measured rows (from one or more
eval_paper_benchmarks.py --out_json result files), across the 5 benchmarks
(Flickr30K, COCO, SG4V-1K, Urban1K, DOCCI) x {I2T, T2I} x R@1.

Usage:
  python assemble_paper_comparison.py \
    --result vanilla:results/vanilla.json \
    --result full:results/full.json \
    --result released:results/released.json \
    --out eval_results/comparison.md
"""
import argparse
import json
import os

BENCHMARKS = ["flickr30k", "coco", "sg4v1k", "urban1k", "docci"]
BENCHMARK_LABELS = {"flickr30k": "Flickr", "coco": "COCO", "sg4v1k": "SG4V",
                     "urban1k": "Urban1K", "docci": "DOCCI"}
REF_ROWS = [
    ("clip_l14_336_baseline", "CLIP-L/14-336 baseline (cited, paper Table 2)"),
    ("llm2clip_15m_336", "LLM2CLIP-15M, ViT-L/14@336 (cited, paper Table 2 -- more data than ours, see caveat)"),
]


def load_measured_row(path):
    """path = eval_paper_benchmarks.py --out_json output: list of per-benchmark dicts."""
    data = json.load(open(path, encoding="utf8"))
    row = {}
    for entry in data:
        b = entry["benchmark"]
        row[b] = {"I2T_R1": entry.get("I2T_R1"), "T2I_R1": entry.get("T2I_R1")}
    return row


def fmt(v):
    return f"{v*100:.1f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", default=[],
                    help="label:path_to_out_json, repeatable (e.g. --result vanilla:results/vanilla.json)")
    ap.add_argument("--reference_json", default=os.path.join(os.path.dirname(__file__), "paper_reference_numbers.json"))
    ap.add_argument("--out", required=True, help="Output .md path (a .csv is written alongside)")
    args = ap.parse_args()

    ref = json.load(open(args.reference_json, encoding="utf8"))

    rows = []  # (label, {benchmark: {I2T_R1, T2I_R1}}, is_cited)
    for key, label in REF_ROWS:
        rows.append((label, ref[key], True))
    for item in args.result:
        label, path = item.split(":", 1)
        rows.append((label, load_measured_row(path), False))

    header = ["Config"] + [f"{BENCHMARK_LABELS[b]} {d}" for b in BENCHMARKS for d in ("I2T", "T2I")]
    lines_md = ["| " + " | ".join(header) + " |",
                "|" + "---|" * len(header)]
    lines_csv = [",".join(header)]

    for label, row, is_cited in rows:
        cells = [label + (" *(cited)*" if is_cited else "")]
        cells_csv = [label + (" (cited)" if is_cited else "")]
        for b in BENCHMARKS:
            d = row.get(b, {})
            cells.append(fmt(d.get("I2T_R1")))
            cells.append(fmt(d.get("T2I_R1")))
            cells_csv.append(str(d.get("I2T_R1", "")))
            cells_csv.append(str(d.get("T2I_R1", "")))
        lines_md.append("| " + " | ".join(cells) + " |")
        lines_csv.append(",".join(cells_csv))

    footnote = (
        "\n**Lưu ý**: 2 hàng *(cited)* lấy thẳng từ paper Table 2, dùng nhiều data hơn "
        "(15M cặp) và không có bản '336px + CC3M-only' được công bố để so khớp tuyệt đối "
        "-- chỉ mang tính tham khảo, KHÔNG phải baseline kiểm soát chặt. Phép so sánh có "
        "kiểm soát thật sự (cùng data, cùng checkpoint xuất phát, cùng lịch train) là giữa "
        "các hàng đo được (không có *(cited)*) với nhau.\n"
    )

    md = "\n".join(lines_md) + "\n" + footnote
    with open(args.out, "w", encoding="utf8") as f:
        f.write(md)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines_csv) + "\n")

    print(md)
    print(f"[done] wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
