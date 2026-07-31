"""Lightning CSVLogger metrics.csv -> training curves PNG (train losses + val metrics).

    python -m scripts.plot_metrics outputs/<date>/<time>/lightning_logs/version_0/metrics.csv demo/curves.png

CSVLogger writes one row per log event with many empty cells (metrics log at different cadences),
so we collect each metric's (x, value) pairs skipping blanks. Left panel = train losses vs step,
right panel = val metrics vs epoch.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _series(rows, metric, xkey):
    xs, ys = [], []
    for r in rows:
        v = r.get(metric, "")
        if v in ("", None):
            continue
        try:
            y = float(v)
        except ValueError:
            continue
        x = r.get(xkey) or r.get("step") or ""
        xs.append(float(x) if x != "" else len(xs))
        ys.append(y)
    return xs, ys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "demo/training_curves.png"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {path}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    for m in ("train/loss", "train/ce", "train/lovasz", "train/center", "train/offset"):
        x, y = _series(rows, m, "step")
        if y:
            ax0.plot(x, y, label=m, linewidth=1)
    ax0.set(title="train losses", xlabel="step", ylabel="loss")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    for m in ("val/mIoU", "val/PQ", "val/PQ_dagger", "val/SQ", "val/RQ"):
        x, y = _series(rows, m, "epoch")
        if y:
            ax1.plot(x, y, marker="o", markersize=3, label=m)
    ax1.set(title="val metrics", xlabel="epoch", ylabel="score")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({len(rows)} log rows)")


if __name__ == "__main__":
    main()
