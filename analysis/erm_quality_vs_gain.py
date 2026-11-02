"""Source ERM quality vs FeatureRater gain (contribution iii).

For each ERM checkpoint (one point per checkpoint) give:
  LABEL  ERM_CHECKPOINT  EMBEDDING_CACHE_DIR  FEATURERATER_ROOT  UNIFORM_ROOT
The ERM's own test WGA/accuracy is computed from its linear head on the cached
test embeddings (no images, no training). The gain is the paired difference
FeatureRater - uniform retraining (same seeds), under one selection rule.

  python analysis/erm_quality_vs_gain.py \
     --point ep5  <erm_ep5.pt>  <cache_ep5>  <fr_root_ep5>  <uniform_root_ep5> \
     --point ep100 <erm_ep100.pt> <cache_ep100> <fr_root_ep100> <uniform_root_ep100> \
     --rule val_worst_class_accuracy --out erm_quality
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import load_embedding_cache, read_experiment, write_csv  # noqa: E402
from analysis.compare import bootstrap_ci, paired  # noqa: E402
from analysis.weight_vs_error import erm_head  # noqa: E402


def erm_test_metrics(erm_model, cache_dir):
    data, _ = load_embedding_cache(Path(cache_dir) / "test.pt")
    w, b = erm_head(erm_model)
    with torch.no_grad():
        pred = (data["embeddings"].float() @ w.T + b).argmax(1)
    correct = (pred == data["labels"].long()).numpy()
    groups = data["groups"].long().numpy()
    accs = {int(gid): float(correct[groups == gid].mean()) for gid in np.unique(groups)}
    return {"erm_test_accuracy": float(correct.mean()), "erm_test_wga": min(accs.values()),
            **{f"erm_test_group_{k}_accuracy": v for k, v in accs.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--point", nargs=5, action="append", required=True,
                   metavar=("LABEL", "ERM_CKPT", "CACHE_DIR", "FR_ROOT", "UNIFORM_ROOT"))
    p.add_argument("--rule", default="val_worst_class_accuracy")
    p.add_argument("--metric", default="test_wga")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for label, erm, cache, fr_root, uni_root in args.point:
        fr, uni = read_experiment(fr_root, args.rule), read_experiment(uni_root, args.rule)
        keys, diffs, unmatched = paired(fr, uni, args.metric)
        lo, hi = bootstrap_ci(diffs)
        row = {"label": label, "erm_checkpoint": erm, **erm_test_metrics(erm, cache),
               "featurerater_mean": float(np.mean([r[args.metric] for r in fr])),
               "uniform_mean": float(np.mean([r[args.metric] for r in uni])),
               "paired_gain_mean": float(diffs.mean()) if len(diffs) else float("nan"),
               "gain_ci95_low": lo, "gain_ci95_high": hi, "n_paired_seeds": len(diffs),
               "unpaired_seeds": " ".join(map(str, unmatched))}
        rows.append(row)
        print(f"{label}: ERM test WGA {100 * row['erm_test_wga']:.1f}  gain {100 * row['paired_gain_mean']:+.2f} "
              f"[{100 * lo:+.2f}, {100 * hi:+.2f}] (n={len(diffs)})")
    out = Path(args.out)
    write_csv(out.with_suffix(".csv"), rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = [100 * r["erm_test_wga"] for r in rows]
    y = [100 * r["paired_gain_mean"] for r in rows]
    yerr = [[100 * (r["paired_gain_mean"] - r["gain_ci95_low"]) for r in rows],
            [100 * (r["gain_ci95_high"] - r["paired_gain_mean"]) for r in rows]]
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    for xi, yi, r in zip(x, y, rows):
        ax.annotate(r["label"], (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(xlabel="Source ERM test WGA (%)", ylabel=f"FeatureRater − uniform, {args.metric} (pts)",
           title=f"Gain vs source ERM quality ({args.rule})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("wrote", out.with_suffix(".csv"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
