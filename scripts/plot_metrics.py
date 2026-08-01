"""Lightning CSVLogger metrics.csv -> training curves PNG (train losses + val metrics).

    python -m scripts.plot_metrics <metrics.csv> [out.png]

Auto-discovers every `train/*` column (left panel, vs step) and `val/*` column (right panel, vs
epoch) from the CSV header, and prints how many points each has — so a missing/empty series is
obvious. CSVLogger writes one row per log event with many empty cells (metrics log at different
cadences), so we skip blanks per column.
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
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows in {path}")

    train_cols = [c for c in cols if c.startswith("train/")]
    val_cols = [c for c in cols if c.startswith("val/")]
    print(f"columns ({len(cols)}): {cols}")
    print(f"train metrics: {train_cols or '(none)'}")
    print(f"val metrics:   {val_cols or '(none)'}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    for m in train_cols:
        x, y = _series(rows, m, "step")
        print(f"  {m}: {len(y)} pts")
        if y:
            ax0.plot(x, y, label=m, linewidth=1)
    ax0.set(title="train", xlabel="step", ylabel="loss")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    for m in val_cols:
        x, y = _series(rows, m, "epoch")
        print(f"  {m}: {len(y)} pts")
        if y:
            ax1.plot(x, y, marker="o", markersize=3, label=m)
    ax1.set(title="val", xlabel="epoch", ylabel="score")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({len(rows)} log rows)")


if __name__ == "__main__":
    main()
