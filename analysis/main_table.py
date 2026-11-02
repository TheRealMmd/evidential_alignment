"""Main results table: WGA and average accuracy, mean ± std over seeds.

Rows come from final-classifier experiment roots (multi-seed notebooks:
FeatureRater, uniform retraining = random-weight notebook with
WEIGHT_MODE="constant", random weights, minority weights, untrained rater)
and from baseline result files written by run_baselines.py.

Every row uses the SAME selection rule (default: val_worst_class_accuracy,
label-free). Rules that read validation group labels are flagged.

Example (Colab, after the runs):
  python analysis/main_table.py \
    --exp "FeatureRater=/content/drive/.../frozen_rater_final_multiseed_multiselection/<run>" \
    --exp "Uniform retraining=/content/drive/.../fixed_random_weight_final_multiseed_multiselection/constant_0.5_..." \
    --baseline "AFR=/content/drive/.../baselines/afr" \
    --rule val_worst_class_accuracy --out /content/drive/.../tables/main
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import (GROUP_LABEL_RULES, WATERBIRDS_TRAIN_GROUP_COUNTS, expand_baseline_paths,  # noqa: E402
                             mean_std, parse_named, read_baseline, read_experiment, write_csv)

METRICS = (("test_wga", "WGA"), ("test_accuracy", "Avg. acc."), ("test_adjusted_accuracy", "Adj. avg. acc."))


def summarize(name, kind, rows, rule):
    out = {"method": name, "kind": kind, "selection_rule": rule,
           "uses_val_group_labels": rule in GROUP_LABEL_RULES,
           "seeds": " ".join(str(r["seed"]) for r in rows),
           "split_seeds": " ".join(sorted({str(r.get("split_seed", "")) for r in rows}))}
    for key, _ in METRICS:
        mean, std, n = mean_std([r.get(key) for r in rows])
        out[f"{key}_mean"], out[f"{key}_std"], out[f"{key}_n"] = mean, std, n
    groups = sorted({k for r in rows for k in r if k.startswith("test_group_")})
    for key in groups:
        mean, std, _ = mean_std([r.get(key) for r in rows])
        out[f"{key}_mean"], out[f"{key}_std"] = mean, std
    return out


def fmt(mean, std, n):
    if n == 0 or mean != mean:
        return "--"
    return f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}"


def latex(rows, rule):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "Method & " + " & ".join(label for _, label in METRICS) + r" \\", r"\midrule"]
    for r in rows:
        cells = [fmt(r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_n"]) for k, _ in METRICS]
        mark = "$^\\dagger$" if r["uses_val_group_labels"] else ""
        lines.append(f"{r['method']}{mark} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% Selection rule: {rule}. Mean $\\pm$ std over seeds (n per cell in the CSV). "
              "$^\\dagger$ uses validation group labels. Adj. avg. = test group accuracies weighted by "
              "training group proportions; '--' where per-group results are unavailable."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", action="append", default=[], help="NAME=experiment_root (repeatable)")
    p.add_argument("--baseline", action="append", default=[], help="NAME=result file, folder or glob (repeatable)")
    p.add_argument("--rule", default="val_worst_class_accuracy")
    p.add_argument("--train-group-counts", type=int, nargs="+", default=list(WATERBIRDS_TRAIN_GROUP_COUNTS))
    p.add_argument("--out", required=True, help="Output prefix (writes <out>.csv, <out>_per_seed.csv, <out>.tex)")
    args = p.parse_args()
    if not args.exp and not args.baseline:
        p.error("pass at least one --exp or --baseline")
    summary, per_seed = [], []
    for name, root in parse_named(args.exp):
        rows = read_experiment(root, args.rule, args.train_group_counts)
        summary.append(summarize(name, "final_classifier", rows, args.rule))
        per_seed += [{"method": name, "source": root, **r} for r in rows]
    for name, spec in parse_named(args.baseline):
        rows = read_baseline(expand_baseline_paths(spec), args.rule)
        summary.append(summarize(name, "baseline", rows, args.rule))
        per_seed += [{"method": name, **r} for r in rows]
    out = Path(args.out)
    write_csv(out.with_suffix(".csv"), summary)
    write_csv(out.parent / (out.name + "_per_seed.csv"), per_seed)
    out.with_suffix(".tex").write_text(latex(summary, args.rule), encoding="utf-8")
    for r in summary:
        def plain(key):
            return fmt(r[f"{key}_mean"], r[f"{key}_std"], r[f"{key}_n"]).replace(" $\\pm$ ", " ± ")
        print(f"{r['method']:<28} WGA {plain('test_wga'):>13}  acc {plain('test_accuracy'):>13}  "
              f"seeds [{r['seeds']}]  split seeds [{r['split_seeds']}]")
    print("wrote", out.with_suffix(".csv"), out.with_suffix(".tex"))


if __name__ == "__main__":
    main()
