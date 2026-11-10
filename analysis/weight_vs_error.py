"""Within-group weighting: rater weight vs the ERM model's loss and errors.

CPU-only, no training. Uses the cached ERM embeddings, the ERM checkpoint's
linear head (the same model that produced the embeddings) and a frozen rater.

Outputs (prefix --out):
  _by_group.csv      per group: n, weight mean/quantiles, mass share, ERM error rate,
                     Spearman(weight, ERM loss) INSIDE the group
  _group_x_correct.csv  weight by group x ERM-correct
  _topq.csv          for the top-q% weighted samples: share that are minority-group
                     samples (precision), share of minority samples captured (recall),
                     the same for an ERM error set of equal size, and their overlap
  _scatter.png       weight vs ERM loss, coloured by group

  python analysis/weight_vs_error.py --cache-dir <ERM_DIR>/embeddings \
      --erm-model <erm.pt> --rater <final_rater.pt> --split train_no_aug --out wve_train
  # the final classifier's own training set (val_subset1 of the notebook split):
  python analysis/weight_vs_error.py ... --split classifier_train --classifier-split-val 0.6 --classifier-split-seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import load_embedding_cache, load_rater_checkpoint, rater_weights, write_csv  # noqa: E402
from analysis.meta_set_overlap import notebook_val_split  # noqa: E402


def erm_head(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_sd", ckpt)
    w = next(v for k, v in state.items() if k.endswith("fc.weight"))
    b = next(v for k, v in state.items() if k.endswith("fc.bias"))
    return w.float(), b.float()


def load_split(cache_dir, split, split_val, split_seed):
    cache_dir = Path(cache_dir)
    if split == "classifier_train":
        data, _ = load_embedding_cache(cache_dir / "val.pt")
        idx, _ = notebook_val_split(len(data["labels"]), split_val, split_seed)
        return {k: v[idx] for k, v in data.items()}
    data, _ = load_embedding_cache(cache_dir / f"{split}.pt")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--erm-model", required=True)
    p.add_argument("--rater", required=True)
    p.add_argument("--split", default="train_no_aug", help="train_no_aug, val, test or classifier_train")
    p.add_argument("--classifier-split-val", type=float, default=0.6)
    p.add_argument("--classifier-split-seed", type=int, default=0)
    p.add_argument("--normalization", default="global", choices=("global", "none"))
    p.add_argument("--transform", default="sigmoid", choices=("sigmoid", "softmax"))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--minority-groups", type=int, nargs="+", default=[1, 2],
                   help="Waterbirds: 1 = landbird/water, 2 = waterbird/land")
    p.add_argument("--top-percent", type=float, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = load_split(args.cache_dir, args.split, args.classifier_split_val, args.classifier_split_seed)
    z, y, g = data["embeddings"].float(), data["labels"].long(), data["groups"].long().numpy()
    rater, ckpt = load_rater_checkpoint(args.rater)
    with torch.no_grad():
        raw = torch.cat([rater(z[i:i + 1024]) for i in range(0, len(z), 1024)]).numpy()
        w_head, b_head = erm_head(args.erm_model)
        logits = z @ w_head.T + b_head
        erm_loss = F.cross_entropy(logits, y, reduction="none").numpy()
        erm_correct = (logits.argmax(1) == y).numpy()
    weights = rater_weights(raw, args.normalization, args.transform, args.temperature)
    total = weights.sum()

    by_group, gx = [], []
    for gid in sorted(np.unique(g)):
        m = g == gid
        rho = stats.spearmanr(weights[m], erm_loss[m]).statistic if m.sum() > 2 else float("nan")
        by_group.append({"group": int(gid), "n": int(m.sum()), "sample_share": float(m.mean()),
                         "weight_mass_share": float(weights[m].sum() / total),
                         "weight_mean": float(weights[m].mean()), "weight_q25": float(np.quantile(weights[m], .25)),
                         "weight_median": float(np.median(weights[m])), "weight_q75": float(np.quantile(weights[m], .75)),
                         "erm_error_rate": float(1 - erm_correct[m].mean()),
                         "spearman_weight_vs_erm_loss_within_group": float(rho)})
        for correct in (True, False):
            mm = m & (erm_correct == correct)
            gx.append({"group": int(gid), "erm_correct": correct, "n": int(mm.sum()),
                       "weight_mean": float(weights[mm].mean()) if mm.any() else float("nan"),
                       "weight_std": float(weights[mm].std()) if mm.any() else float("nan")})

    minority = np.isin(g, args.minority_groups)
    error_set = ~erm_correct
    topq = []
    order_w = np.argsort(-weights, kind="stable")
    order_l = np.argsort(-erm_loss, kind="stable")
    for q in args.top_percent:
        k = max(1, int(round(len(weights) * q / 100)))
        top_w = np.zeros(len(weights), bool); top_w[order_w[:k]] = True
        top_l = np.zeros(len(weights), bool); top_l[order_l[:k]] = True
        topq.append({"top_percent": q, "k": k,
                     "rater_minority_precision": float(minority[top_w].mean()),
                     "rater_minority_recall": float((minority & top_w).sum() / max(minority.sum(), 1)),
                     "erm_loss_minority_precision": float(minority[top_l].mean()),
                     "erm_loss_minority_recall": float((minority & top_l).sum() / max(minority.sum(), 1)),
                     "overlap_rater_vs_erm_loss_topk": float((top_w & top_l).sum() / k),
                     "share_of_top_rater_that_erm_misclassifies": float(error_set[top_w].mean())})
    topq.append({"top_percent": "ERM error set", "k": int(error_set.sum()),
                 "erm_loss_minority_precision": float(minority[error_set].mean()) if error_set.any() else float("nan"),
                 "erm_loss_minority_recall": float((minority & error_set).sum() / max(minority.sum(), 1))})

    out = Path(args.out)
    write_csv(out.parent / (out.name + "_by_group.csv"), by_group)
    write_csv(out.parent / (out.name + "_group_x_correct.csv"), gx)
    write_csv(out.parent / (out.name + "_topq.csv"), topq)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for gid in sorted(np.unique(g)):
        m = g == gid
        axes[0].scatter(erm_loss[m] + 1e-6, weights[m], s=6, alpha=0.5, label=f"group {gid} (n={m.sum()})")
    axes[0].set(xscale="log", xlabel="ERM per-sample CE (log)", ylabel="rater weight",
                title=f"{args.split}: weight vs ERM loss")
    axes[0].legend(fontsize=8)
    labels = [f"g{r['group']} {'ok' if r['erm_correct'] else 'err'}" for r in gx]
    axes[1].bar(range(len(gx)), [r["weight_mean"] for r in gx], yerr=[r["weight_std"] for r in gx], capsize=3)
    axes[1].set_xticks(range(len(gx)), labels, rotation=30)
    axes[1].set(ylabel="mean rater weight", title="by group and ERM correctness")
    fig.tight_layout()
    fig.savefig(out.parent / (out.name + "_scatter.png"), dpi=160)
    for r in by_group:
        print(f"group {r['group']}: n={r['n']:<5} weight mean {r['weight_mean']:.3f} "
              f"(mass {100 * r['weight_mass_share']:.1f}% vs samples {100 * r['sample_share']:.1f}%), "
              f"ERM err {100 * r['erm_error_rate']:.1f}%, Spearman(w, ERM loss) {r['spearman_weight_vs_erm_loss_within_group']:+.3f}")
    print("rater best_meta_step:", ckpt.get("best_meta_step"))
    print("wrote", out.parent / (out.name + "_by_group.csv"))


if __name__ == "__main__":
    main()
