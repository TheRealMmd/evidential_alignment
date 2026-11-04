"""ERM record: how the frozen ERM classifier itself performs on each split.

The rater and the final classifier are built on embeddings from one ERM
checkpoint. This module evaluates that checkpoint's own linear head on the
cached embeddings, so every rater run and every embedding cache carries the
ERM baseline it started from (per-group accuracy, WGA, average and adjusted
average accuracy). It only reads tensors; it does not touch any RNG.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _json_safe(value):
    """Recursively convert values to standard Python JSON-serializable types."""

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    # np.float32, np.float64, np.int64, etc.
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch.is_tensor(value):
        value = value.detach().cpu()

        if value.ndim == 0:
            return value.item()

        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def split_metrics(
    weight,
    bias,
    embeddings,
    labels,
    groups,
    train_group_share=None,
):
    """Metrics of the linear head (weight, bias) on one split."""

    with torch.no_grad():
        logits = (
            embeddings.float().cpu()
            @ weight.float().cpu().T
            + bias.float().cpu()
        )

        labels = (
            labels.long()
            .cpu()
            .reshape(-1)
        )

        groups = (
            groups.long()
            .cpu()
            .reshape(-1)
        )

        ce = F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        correct = (
            logits.argmax(dim=1)
            .eq(labels)
            .double()
        )

    per_group = {}

    for g in torch.unique(groups).tolist():
        mask = groups.eq(g)

        per_group[int(g)] = {
            "n": int(mask.sum().item()),
            "accuracy": float(
                correct[mask].mean().item()
            ),
            "mean_ce": float(
                ce[mask].mean().item()
            ),
        }

    per_class = {}

    for c in torch.unique(labels).tolist():
        mask = labels.eq(c)

        per_class[int(c)] = {
            "n": int(mask.sum().item()),
            "accuracy": float(
                correct[mask].mean().item()
            ),
        }

    accs = [
        value["accuracy"]
        for value in per_group.values()
    ]

    out = {
        "n": int(labels.numel()),
        "accuracy": float(
            correct.mean().item()
        ),
        "mean_ce": float(
            ce.mean().item()
        ),
        "worst_group_accuracy": float(
            min(accs)
        ),
        "mean_group_accuracy": float(
            sum(accs) / len(accs)
        ),
        "worst_class_accuracy": float(
            min(
                value["accuracy"]
                for value in per_class.values()
            )
        ),
        "per_group": per_group,
        "per_class": per_class,
    }

    if train_group_share:
        shared = [
            g
            for g in per_group
            if g in train_group_share
        ]

        total = sum(
            train_group_share[g]
            for g in shared
        )

        out["adjusted_average_accuracy"] = (
            float(
                sum(
                    train_group_share[g]
                    * per_group[g]["accuracy"]
                    for g in shared
                )
                / total
            )
            if total > 0
            else math.nan
        )

    return out


def build_record(
    weight,
    bias,
    splits,
    train_split=None,
    checkpoint=None,
):
    """Build the frozen-ERM evaluation record.

    Args:
        weight:
            Linear-head weight tensor.

        bias:
            Linear-head bias tensor.

        splits:
            Mapping:
                {name: (embeddings, labels, groups)}

        train_split:
            Split whose group proportions define adjusted average accuracy.

        checkpoint:
            Optional checkpoint metadata.
    """

    share = None

    if train_split in splits:
        groups = (
            splits[train_split][2]
            .long()
            .reshape(-1)
        )

        share = {
            int(g): float(
                groups.eq(g)
                .double()
                .mean()
                .item()
            )
            for g in torch.unique(
                groups
            ).tolist()
        }

    record = {
        "what": (
            "the frozen ERM checkpoint's own linear head "
            "evaluated on the cached embeddings"
        ),
        "checkpoint": (
            checkpoint or {}
        ),
        "adjusted_average_weights_from": (
            train_split
            if share
            else None
        ),
        "train_group_share": share,
        "splits": {
            name: split_metrics(
                weight,
                bias,
                *tensors,
                train_group_share=share,
            )
            for name, tensors
            in splits.items()
        },
    }

    # Important:
    # checkpoint metadata can contain np.float32 / np.int64 / tensors.
    # Normalize everything at the serialization boundary.
    return _json_safe(record)


def write_record(
    record,
    json_path,
    csv_path,
):
    """Write JSON plus one-row-per-group CSV."""

    safe_record = _json_safe(
        record
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            safe_record,
            f,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "group",
            "n",
            "accuracy",
            "mean_ce",
            "worst_group_accuracy",
            "mean_group_accuracy",
            "adjusted_average_accuracy",
            "worst_class_accuracy",
        ])

        for name, metrics in safe_record["splits"].items():
            writer.writerow([
                name,
                "all",
                metrics["n"],
                metrics["accuracy"],
                metrics["mean_ce"],
                metrics["worst_group_accuracy"],
                metrics["mean_group_accuracy"],
                metrics.get(
                    "adjusted_average_accuracy",
                    "",
                ),
                metrics["worst_class_accuracy"],
            ])

            for g, group_metrics in sorted(
                metrics["per_group"].items(),
                key=lambda item: int(
                    item[0]
                ),
            ):
                writer.writerow([
                    name,
                    g,
                    group_metrics["n"],
                    group_metrics["accuracy"],
                    group_metrics["mean_ce"],
                    "",
                    "",
                    "",
                    "",
                ])


def format_record(record):
    """Short human-readable table for logs."""

    record = _json_safe(
        record
    )

    lines = []

    for name, metrics in record["splits"].items():

        groups = "  ".join(
            (
                f"g{g}="
                f"{group_metrics['accuracy']:.4f}"
                f"(n={group_metrics['n']})"
            )
            for g, group_metrics
            in sorted(
                metrics["per_group"].items(),
                key=lambda item: int(
                    item[0]
                ),
            )
        )

        adjusted = metrics.get(
            "adjusted_average_accuracy"
        )

        adjusted_text = (
            f" adj_avg={adjusted:.4f}"
            if adjusted is not None
            else ""
        )

        lines.append(
            (
                f"[ERM record] {name}: "
                f"acc={metrics['accuracy']:.4f} "
                f"WGA={metrics['worst_group_accuracy']:.4f}"
                f"{adjusted_text} | "
                f"{groups}"
            )
        )

    return lines
