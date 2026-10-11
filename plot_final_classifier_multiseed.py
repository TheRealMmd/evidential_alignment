#!/usr/bin/env python3
"""Aggregate completed final-classifier runs across seeds.

Usage:
    python plot_final_classifier_multiseed.py \
        --runs RUN_SEED0 RUN_SEED1 RUN_SEED2 \
        --output OUTPUT_DIR

The script only reads completed runs. It never launches training and does not
interpolate missing epochs.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_FIELDS = ("accuracy", "wga", "loss", "worst_group_loss")
METRIC_LABELS = (
    "Accuracy",
    "Worst-group accuracy",
    "Mean CE loss",
    "Worst-group mean CE loss",
)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    )


def normalized_condition(config):
    """Allow seed and concrete Rater checkpoint paths to differ across seeds."""
    cfg = dict(config)
    cfg.pop("seed", None)
    cfg.pop("rater_checkpoint", None)
    cfg.pop("gradcam", None)  # diagnostic-only settings do not affect training

    signature = dict(cfg.get("rater_signature", {}))
    # These naturally differ across independently trained Rater seeds.
    signature.pop("best_meta_step", None)
    signature.pop("best_outer_loss", None)
    cfg["rater_signature"] = signature
    return cfg


def read_run(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "plots" / "data" / "metrics.csv"
    groups_path = run_dir / "plots" / "data" / "group_metrics.csv"

    for path in (config_path, summary_path, metrics_path, groups_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir.resolve()),
        "seed": int(config["seed"]),
        "config": config,
        "summary": summary,
        "metrics": read_csv(metrics_path),
        "groups": read_csv(groups_path),
    }


def aggregate_runs(run_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_run(path) for path in run_dirs]

    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each run must have a unique seed.")

    reference = None
    for run in runs:
        condition = normalized_condition(run["config"])
        if reference is None:
            reference = condition
        elif condition != reference:
            raise ValueError(
                "Runs differ in training settings beyond seed / checkpoint path. "
                "Aggregate one scientific condition at a time."
            )

    write_json(
        output_dir / "runs.json",
        {
            "condition": reference,
            "runs": [
                {"seed": run["seed"], "run_dir": run["run_dir"]}
                for run in runs
            ],
        },
    )

    # ------------------------------------------------------------
    # Overall trajectory metrics
    # ------------------------------------------------------------
    grouped = defaultdict(list)
    for run in runs:
        for row in run["metrics"]:
            if row["phase"] != "trajectory":
                continue
            key = (row["role"], row["split"], int(row["epoch"]))
            grouped[key].append(row)

    aggregate = []
    for (role, split, epoch), rows in sorted(grouped.items()):
        item = {
            "role": role,
            "split": split,
            "epoch": epoch,
            "n_seeds": len(rows),
        }
        for field in METRIC_FIELDS:
            item[field + "_mean"], item[field + "_std"] = mean_std(
                [float(row[field]) for row in rows]
            )
        aggregate.append(item)

    write_csv(output_dir / "trajectory_metrics_mean_std.csv", aggregate)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, field, label in zip(axes.flat, METRIC_FIELDS, METRIC_LABELS):
        for role in ("train", "val", "test"):
            values = sorted(
                [row for row in aggregate if row["role"] == role],
                key=lambda row: row["epoch"],
            )
            if not values:
                continue
            x = np.asarray([row["epoch"] for row in values])
            mean = np.asarray([row[field + "_mean"] for row in values])
            std = np.asarray([row[field + "_std"] for row in values])
            counts = [row["n_seeds"] for row in values]
            ax.plot(x, mean, label=f"{role} (n={min(counts)}–{max(counts)})")
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Final classifier trajectory: mean ± seed SD")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_metrics_mean_std.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Selected model metrics
    # ------------------------------------------------------------
    selected_grouped = defaultdict(list)
    for run in runs:
        for row in run["metrics"]:
            if row["phase"] == "selected":
                selected_grouped[(row["role"], row["split"])].append(row)

    selected = []
    for (role, split), rows in sorted(selected_grouped.items()):
        item = {"role": role, "split": split, "n_seeds": len(rows)}
        for field in METRIC_FIELDS:
            item[field + "_mean"], item[field + "_std"] = mean_std(
                [float(row[field]) for row in rows]
            )
        selected.append(item)

    write_csv(output_dir / "selected_metrics_mean_std.csv", selected)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    roles = [row["role"] for row in selected]
    for ax, field, label in zip(axes.flat, METRIC_FIELDS, METRIC_LABELS):
        means = [row[field + "_mean"] for row in selected]
        stds = [row[field + "_std"] for row in selected]
        ax.bar(roles, means, yerr=stds, capsize=3)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Selected final classifier: mean ± seed SD")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_metrics_mean_std.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Per-group trajectory metrics
    # ------------------------------------------------------------
    grouped = defaultdict(list)
    for run in runs:
        for row in run["groups"]:
            if row["phase"] != "trajectory":
                continue
            key = (
                row["role"], row["split"], int(row["group"]), int(row["epoch"])
            )
            grouped[key].append(row)

    group_aggregate = []
    for (role, split, group, epoch), rows in sorted(grouped.items()):
        item = {
            "role": role,
            "split": split,
            "group": group,
            "epoch": epoch,
            "n_seeds": len(rows),
        }
        for field in ("accuracy", "loss"):
            item[field + "_mean"], item[field + "_std"] = mean_std(
                [float(row[field]) for row in rows]
            )
        group_aggregate.append(item)

    write_csv(output_dir / "group_metrics_mean_std.csv", group_aggregate)

    for role in ("train", "val", "test"):
        role_rows = [row for row in group_aggregate if row["role"] == role]
        if not role_rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, field, label in (
            (axes[0], "accuracy", "Group accuracy"),
            (axes[1], "loss", "Group mean CE"),
        ):
            for group in sorted({row["group"] for row in role_rows}):
                values = sorted(
                    [row for row in role_rows if row["group"] == group],
                    key=lambda row: row["epoch"],
                )
                x = np.asarray([row["epoch"] for row in values])
                mean = np.asarray([row[field + "_mean"] for row in values])
                std = np.asarray([row[field + "_std"] for row in values])
                ax.plot(x, mean, label=f"Group {group}")
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{role}: per-group mean ± seed SD")
        fig.tight_layout()
        fig.savefig(output_dir / f"group_metrics_{role}_mean_std.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return {
        "trajectory": aggregate,
        "selected": selected,
        "groups": group_aggregate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate completed final-classifier runs across seeds."
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate_runs(args.runs, args.output)


if __name__ == "__main__":
    main()
