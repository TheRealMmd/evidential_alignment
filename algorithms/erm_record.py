"""ERM record: how the frozen ERM classifier itself performs on each split.

The rater and the final classifier are built on embeddings from one ERM
checkpoint. This module evaluates that checkpoint's own linear head on the
cached embeddings, so every rater run and every embedding cache carries the
ERM baseline it started from (per-group accuracy, WGA, average and adjusted
average accuracy). It only reads tensors; it does not touch any RNG.
"""

import csv
import math

import torch
import torch.nn.functional as F


def split_metrics(weight, bias, embeddings, labels, groups, train_group_share=None):
    """Metrics of the linear head (weight, bias) on one split."""
    with torch.no_grad():
        logits = embeddings.float().cpu() @ weight.float().cpu().T + bias.float().cpu()
        labels = labels.long().cpu().reshape(-1)
        groups = groups.long().cpu().reshape(-1)
        ce = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(dim=1).eq(labels).double()
    per_group = {}
    for g in torch.unique(groups).tolist():
        mask = groups.eq(g)
        per_group[int(g)] = {"n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                             "mean_ce": float(ce[mask].mean())}
    per_class = {}
    for c in torch.unique(labels).tolist():
        mask = labels.eq(c)
        per_class[int(c)] = {"n": int(mask.sum()), "accuracy": float(correct[mask].mean())}
    accs = [v["accuracy"] for v in per_group.values()]
    out = {
        "n": int(labels.numel()),
        "accuracy": float(correct.mean()),
        "mean_ce": float(ce.mean()),
        "worst_group_accuracy": min(accs),
        "mean_group_accuracy": sum(accs) / len(accs),
        "worst_class_accuracy": min(v["accuracy"] for v in per_class.values()),
        "per_group": per_group,
        "per_class": per_class,
    }
    if train_group_share:
        shared = [g for g in per_group if g in train_group_share]
        total = sum(train_group_share[g] for g in shared)
        out["adjusted_average_accuracy"] = (
            sum(train_group_share[g] * per_group[g]["accuracy"] for g in shared) / total
            if total > 0 else math.nan)
    return out


def build_record(weight, bias, splits, train_split=None, checkpoint=None):
    """splits: {name: (embeddings, labels, groups)}; train_split names the split whose
    group proportions define adjusted average accuracy (the usual Waterbirds metric)."""
    share = None
    if train_split in splits:
        groups = splits[train_split][2].long().reshape(-1)
        share = {int(g): float(groups.eq(g).double().mean()) for g in torch.unique(groups).tolist()}
    return {
        "what": "the frozen ERM checkpoint's own linear head evaluated on the cached embeddings",
        "checkpoint": checkpoint or {},
        "adjusted_average_weights_from": train_split if share else None,
        "train_group_share": share,
        "splits": {name: split_metrics(weight, bias, *tensors, train_group_share=share)
                   for name, tensors in splits.items()},
    }


def write_record(record, json_path, csv_path):
    """JSON with everything; CSV with one row per (split, group) plus a summary row per split."""
    import json
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "group", "n", "accuracy", "mean_ce", "worst_group_accuracy",
                    "mean_group_accuracy", "adjusted_average_accuracy", "worst_class_accuracy"])
        for name, m in record["splits"].items():
            w.writerow([name, "all", m["n"], m["accuracy"], m["mean_ce"], m["worst_group_accuracy"],
                        m["mean_group_accuracy"], m.get("adjusted_average_accuracy", ""),
                        m["worst_class_accuracy"]])
            for g, gm in sorted(m["per_group"].items()):
                w.writerow([name, g, gm["n"], gm["accuracy"], gm["mean_ce"], "", "", "", ""])


def format_record(record):
    """Short human-readable table for logs."""
    lines = []
    for name, m in record["splits"].items():
        groups = "  ".join(f"g{g}={gm['accuracy']:.4f}(n={gm['n']})" for g, gm in sorted(m["per_group"].items()))
        adj = m.get("adjusted_average_accuracy")
        adj_text = f" adj_avg={adj:.4f}" if adj is not None else ""
        lines.append(f"[ERM record] {name}: acc={m['accuracy']:.4f} WGA={m['worst_group_accuracy']:.4f}"
                     f"{adj_text} | {groups}")
    return lines
