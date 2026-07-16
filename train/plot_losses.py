"""
Task 2 & 3 (PDF: Test case Rethinking project, 9/7/2026)
Tạo biểu đồ loss từ file log train.py sinh ra.

Cách dùng:
    # 1 log, có breakdown (loss_mode=full, use_hinge=True) -> 3 hàm loss + total/hinge
    python plot_losses.py <log_file> [--out <png_path>]

    # 1 log, không breakdown (--without_hinge_loss hoặc loss_mode=vanilla) -> chỉ total_loss
    python plot_losses.py <log_file_vanilla>

    # so sánh nhiều run (vd: baseline vs weight_hinge_loss 0.5 vs without_hinge_loss)
    python plot_losses.py --compare run1.log:baseline run2.log:weight_0.5 run3.log:no_hinge --out compare.png

Log line có 2 dạng:
  "Epoch 0 ▶ total_loss: 12.1191 ▶ L_long: 8.4613 ▶ L_pos: 9.1947 ▶ L_longer: 9.0586 ▶ L_hinge: 0.0238 ▶ logit scale: 100.0000 --- tau: 0.0100"
  "Epoch 0 ▶ Loss: 0.9822"   (use_hinge=False: --without_hinge_loss hoặc loss_mode=vanilla)
"""
import argparse
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BREAKDOWN_RE = re.compile(
    r"Epoch (?P<epoch>\d+) ▶ total_loss: (?P<total>[\d.]+) ▶ "
    r"L_long: (?P<L_long>[\d.]+) ▶ L_pos: (?P<L_pos>[\d.]+) ▶ "
    r"L_longer: (?P<L_longer>[\d.]+) ▶ L_hinge: (?P<L_hinge>[\d.]+)"
)
VANILLA_RE = re.compile(r"Epoch (?P<epoch>\d+) ▶ Loss: (?P<total>[\d.]+)")


def parse_log(log_path):
    """Trả về (steps dict, has_breakdown bool)."""
    steps = {"total_loss": [], "L_long": [], "L_pos": [], "L_longer": [], "L_hinge": []}
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = BREAKDOWN_RE.search(line)
            if m:
                steps["total_loss"].append(float(m.group("total")))
                steps["L_long"].append(float(m.group("L_long")))
                steps["L_pos"].append(float(m.group("L_pos")))
                steps["L_longer"].append(float(m.group("L_longer")))
                steps["L_hinge"].append(float(m.group("L_hinge")))
                continue
            m = VANILLA_RE.search(line)
            if m:
                steps["total_loss"].append(float(m.group("total")))

    has_breakdown = len(steps["L_long"]) > 0
    return steps, has_breakdown


def plot_single(steps, has_breakdown, out_path, title):
    if not has_breakdown:
        n_points = len(steps["total_loss"])
        if n_points == 0:
            raise ValueError(f"Không tìm thấy dòng log loss nào khớp định dạng trong {title}.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(n_points), steps["total_loss"], label="total_loss", color="black")
        ax.set_xlabel("log point (mỗi 50 step)")
        ax.set_ylabel("loss")
        ax.set_title(f"{title} — total_loss (không có breakdown, use_hinge=False)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path} ({n_points} points, vanilla/no-breakdown)")
        return

    n_points = len(steps["L_long"])
    x = list(range(n_points))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, steps["L_long"], label="L_long")
    ax.plot(x, steps["L_pos"], label="L_pos")
    ax.plot(x, steps["L_longer"], label="L_longer")
    ax.set_ylabel("loss")
    ax.set_title(f"{title} — 3 hàm loss chính")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, steps["total_loss"], label="total_loss", color="black")
    ax2.plot(x, steps["L_hinge"], label="L_hinge", color="tab:red")
    ax2.set_xlabel("log point (mỗi 50 step)")
    ax2.set_ylabel("loss")
    ax2.set_title("total_loss & L_hinge")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path} ({n_points} points)")


def plot_compare(runs, out_path):
    """runs: list of (log_path, label)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for log_path, label in runs:
        steps, _ = (parse_csv(log_path) if log_path.endswith(".csv") else parse_log(log_path))
        n_points = len(steps["total_loss"])
        if n_points == 0:
            print(f"[warn] {log_path}: no loss lines found, skipping")
            continue
        ax.plot(range(n_points), steps["total_loss"], label=label)
    ax.set_xlabel("log point (mỗi 50 step)")
    ax.set_ylabel("total_loss")
    ax.set_title("So sánh total_loss giữa các cấu hình hinge loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def parse_csv(csv_path):
    """Đọc loss_log.csv do train.py tự sinh (step,epoch,total_loss,L_long,L_pos,L_longer,L_hinge,logit_scale)."""
    import csv as _csv
    steps = {"total_loss": [], "L_long": [], "L_pos": [], "L_longer": [], "L_hinge": []}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            for k in steps:
                steps[k].append(float(row[k]))
    has_breakdown = any(steps["L_pos"]) or any(steps["L_longer"]) or any(steps["L_hinge"])
    return steps, has_breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", nargs="?",
                    help="loss_log.csv (khuyến nghị) HOẶC file .log/stdout do train.py sinh")
    ap.add_argument("--out", default=None, help="Đường dẫn file PNG output")
    ap.add_argument("--compare", nargs="+", default=None,
                     help="So sánh nhiều log: path1:label1 path2:label2 ...")
    args = ap.parse_args()

    if args.compare:
        runs = []
        for item in args.compare:
            if ":" in item:
                path, label = item.rsplit(":", 1)
            else:
                path, label = item, item
            runs.append((path, label))
        out_path = args.out or "loss_compare.png"
        plot_compare(runs, out_path)
        return

    if not args.log_file:
        ap.error("cần log_file (loss_log.csv hoặc .log) hoặc --compare")

    # auto: .csv -> parse_csv (loss_log.csv của train.py), còn lại -> parse_log (stdout)
    if args.log_file.endswith(".csv"):
        steps, has_breakdown = parse_csv(args.log_file)
    else:
        steps, has_breakdown = parse_log(args.log_file)
    out_path = args.out or (args.log_file.rsplit(".", 1)[0] + "_losses.png")
    plot_single(steps, has_breakdown, out_path, title=args.log_file)


if __name__ == "__main__":
    main()
