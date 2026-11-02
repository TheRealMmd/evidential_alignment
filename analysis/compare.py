"""Paired comparison: FeatureRater minus each control, on the SAME seeds.

Rows are joined on (seed, split_seed). If two experiments used different data
splits for the same seed (e.g. VARY_DATA_SPLIT_WITH_SEED differs between the
notebooks), those seeds are NOT paired and the script says so.

Reports the mean paired difference, a percentile bootstrap CI over seeds, the
number of seeds where the difference is positive, and an exact Wilcoxon
signed-rank p-value (scipy). With 5-7 seeds these are descriptive, not proof.

  python analysis/compare.py --reference "FeatureRater=<root>" \
      --control "Uniform=<root>" --control "Random=<root>" \
      --rule val_worst_class_accuracy --metric test_wga --out <prefix>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import GROUP_LABEL_RULES, parse_named, read_experiment, write_csv  # noqa: E402


def paired(reference, control, metric):
    ref = {(r["seed"], r["split_seed"]): r[metric] for r in reference}
    con = {(r["seed"], r["split_seed"]): r[metric] for r in control}
    keys = sorted(set(ref) & set(con))
    unmatched_seeds = sorted({k[0] for k in set(ref) ^ set(con)})
    diffs = np.array([ref[k] - con[k] for k in keys], dtype=float)
    return keys, diffs, unmatched_seeds


def bootstrap_ci(diffs, n_boot=10000, seed=0, level=0.95):
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def wilcoxon_p(diffs):
    if len(diffs) < 2 or np.allclose(diffs, 0):
        return float("nan")
    from scipy import stats
    return float(stats.wilcoxon(diffs, alternative="two-sided", method="exact").pvalue)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="NAME=experiment_root (the method)")
    p.add_argument("--control", action="append", required=True, help="NAME=experiment_root (repeatable)")
    p.add_argument("--rule", default="val_worst_class_accuracy")
    p.add_argument("--metric", default="test_wga")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    (ref_name, ref_root), = parse_named([args.reference])
    reference = read_experiment(ref_root, args.rule)
    rows, plot_data = [], []
    for name, root in parse_named(args.control):
        keys, diffs, unmatched = paired(reference, read_experiment(root, args.rule), args.metric)
        lo, hi = bootstrap_ci(diffs)
        row = {"reference": ref_name, "control": name, "metric": args.metric, "selection_rule": args.rule,
               "uses_val_group_labels": args.rule in GROUP_LABEL_RULES,
               "n_paired_seeds": len(diffs), "paired_seeds": " ".join(str(k[0]) for k in keys),
               "unpaired_seeds": " ".join(map(str, unmatched)),
               "mean_difference": float(diffs.mean()) if len(diffs) else float("nan"),
               "ci95_low": lo, "ci95_high": hi,
               "n_positive": int((diffs > 0).sum()), "n_negative": int((diffs < 0).sum()),
               "wilcoxon_p": wilcoxon_p(diffs)}
        rows.append(row)
        plot_data.append((name, diffs))
        print(f"{ref_name} - {name}: {100 * row['mean_difference']:+.2f} pts "
              f"[{100 * lo:+.2f}, {100 * hi:+.2f}] over {len(diffs)} paired seeds; "
              f"{row['n_positive']} up / {row['n_negative']} down; p={row['wilcoxon_p']:.3g}"
              + (f"; NOT paired (different split or missing): seeds {unmatched}" if unmatched else ""))
    out = Path(args.out)
    write_csv(out.with_suffix(".csv"), rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(plot_data), 4))
    for i, (name, diffs) in enumerate(plot_data):
        jitter = np.linspace(-0.12, 0.12, max(len(diffs), 1))[:len(diffs)]
        ax.scatter(i + jitter, 100 * diffs, s=22, zorder=3)
        if len(diffs):
            ax.errorbar(i + 0.3, 100 * diffs.mean(),
                        yerr=[[100 * (diffs.mean() - rows[i]["ci95_low"])], [100 * (rows[i]["ci95_high"] - diffs.mean())]],
                        fmt="o", color="k", capsize=4)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xticks(range(len(plot_data)), [n for n, _ in plot_data])
    ax.set_ylabel(f"{ref_name} − control, {args.metric} (pts)")
    ax.set_title(f"Paired by seed; rule {args.rule}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("wrote", out.with_suffix(".csv"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
