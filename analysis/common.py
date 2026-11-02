"""Shared helpers for the analysis scripts. CPU-only; never trains anything.

Inputs are the folders the notebooks and main.py already write:
  * final-classifier experiment roots (multi-seed notebooks):
      selection_results_all_seeds.csv, seed_<s>/summary.json,
      seed_<s>/plots/data/group_metrics.csv
  * baseline result files written by main.py --result_path (see run_baselines.py)
  * rater runs (final_rater.pt, plots/data/...)
"""

import csv
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Rules that never read group labels (selection on val_subset2).
LABEL_FREE_RULES = (
    "val_worst_class_accuracy", "val_accuracy", "val_mean_class_accuracy",
    "val_loss", "val_worst_class_loss",
)
GROUP_LABEL_RULES = ("val_wga", "val_mean_group_accuracy", "val_worst_group_loss")

# Waterbirds training-set group sizes (metadata.csv, split 0; group = 2*y + place):
# landbird/land, landbird/water, waterbird/land, waterbird/water. Used for the
# conventional "adjusted average accuracy" (test group accuracies weighted by
# training group proportions). Pass --train-group-counts for other datasets.
WATERBIRDS_TRAIN_GROUP_COUNTS = (3498, 184, 56, 1057)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def parse_named(items, what="NAME=PATH"):
    """['FeatureRater=/a', 'Uniform=/b'] -> [('FeatureRater', '/a'), ...] keeping order."""
    pairs = []
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Expected {what}, got {item!r}")
        name, path = item.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    return pairs


def adjusted_average(group_accuracy, train_group_counts):
    counts = np.asarray(train_group_counts, dtype=float)
    weights = counts / counts.sum()
    if len(group_accuracy) != len(weights) or any(not math.isfinite(a) for a in group_accuracy):
        return float("nan")
    return float(np.dot(weights, group_accuracy))


def read_experiment(root, rule, train_group_counts=WATERBIRDS_TRAIN_GROUP_COUNTS):
    """Per-seed test metrics of one final-classifier experiment under one selection rule.

    Returns rows with seed, split_seed, selected_epoch, test_accuracy, test_wga,
    test_mean_group_accuracy, test_adjusted_accuracy, test_group_<g>_accuracy.
    """
    root = Path(root)
    path = root / "selection_results_all_seeds.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: pass a multi-seed experiment root.")
    rows = [r for r in read_csv(path) if r["selection_rule"] == rule]
    if not rows:
        rules = sorted({r["selection_rule"] for r in read_csv(path)})
        raise SystemExit(f"Rule {rule!r} not in {path}; available: {rules}")
    out = []
    for r in rows:
        seed = int(r["seed"])
        row = {
            "seed": seed, "split_seed": int(r.get("split_seed", seed)),
            "selected_epoch": int(float(r["selected_epoch"])),
            "test_accuracy": to_float(r["test_accuracy"]),
            "test_wga": to_float(r["test_wga"]),
            "test_mean_group_accuracy": to_float(r.get("test_mean_group_accuracy")),
        }
        group_csv = root / f"seed_{seed}" / "plots" / "data" / "group_metrics.csv"
        group_acc = []
        if group_csv.exists():
            selected = [g for g in read_csv(group_csv)
                        if g["phase"] == "selected" and g["selection_rule"] == rule and g["role"] == "test"]
            for g in sorted(selected, key=lambda g: int(g["group"])):
                row[f"test_group_{int(g['group'])}_accuracy"] = to_float(g["accuracy"])
                group_acc.append(to_float(g["accuracy"]))
        row["test_adjusted_accuracy"] = (adjusted_average(group_acc, train_group_counts)
                                         if group_acc else float("nan"))
        summary = root / f"seed_{seed}" / "summary.json"
        if summary.exists():
            info = json.loads(summary.read_text(encoding="utf-8"))
            row["elapsed_seconds"] = info.get("elapsed_seconds")
            row["peak_gpu_memory_bytes"] = info.get("peak_gpu_memory_bytes")
        out.append(row)
    return sorted(out, key=lambda r: r["seed"])


# ---------------------------------------------------------------- baselines
_SEL_TO_RULE = (
    ("worst_cls_acc", "val_worst_class_accuracy"),
    ("worst_group_acc", "val_wga"),
    ("avg_cls_diff", "val_class_gap"),
    ("unbiased_group_acc", "val_mean_group_accuracy"),
    ("acc", "val_accuracy"),
)
_METRIC_MAP = (
    ("unbiased_group_acc", "test_mean_group_accuracy"),
    ("worst_group_acc", "test_wga"),
    ("worst_cls_acc", "test_worst_class_accuracy"),
    ("avg_cls_diff", "test_class_gap"),
    ("acc", "test_accuracy"),
)


def _map_suffix(key, table):
    for suffix, name in table:
        if key.endswith("_" + suffix) or key == suffix:
            return name
    return None


def parse_baseline_results(path):
    """Parse main.py --result_path files (afr/dfr/jtt/evidential_alignment test()).

    Header line: '<method> <dataset> ... seed:<s> ...'. Result lines:
    '[test (<val metric>:<value>)]: <split>_acc:0.95, ...' or '[test (latest)]: ...'.
    """
    rows, header = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("["):
            parts = line.split()
            header = {"method": parts[0]}
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    header[k] = v
            continue
        m = re.match(r"\[(\S+) \(([^):]+)(?::([-0-9.eE]+))?\)\]:\s*(.*)", line)
        if not m:
            continue
        split, selector, _, rest = m.groups()
        if split != "test":
            continue
        rule = "final_epoch" if selector == "latest" else _map_suffix(selector, _SEL_TO_RULE)
        row = {"method": header.get("method"), "seed": int(header.get("seed", -1)),
               "selection_rule": rule, "source_file": str(path)}
        for item in rest.split(","):
            if ":" not in item:
                continue
            k, v = item.strip().rsplit(":", 1)
            name = _map_suffix(k.strip(), _METRIC_MAP)
            if name:
                row[name] = to_float(v)
        rows.append(row)
    return rows


def read_baseline(paths, rule):
    rows = []
    for path in paths:
        rows.extend(r for r in parse_baseline_results(path) if r["selection_rule"] == rule)
    if not rows:
        raise SystemExit(f"No rows for rule {rule!r} in {paths}")
    for r in rows:
        r.setdefault("test_adjusted_accuracy", float("nan"))
    return sorted(rows, key=lambda r: r["seed"])


def expand_baseline_paths(spec):
    """A file, a directory of *.txt result files, or a glob."""
    p = Path(spec)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*.txt"))
    import glob
    matches = sorted(glob.glob(spec))
    if not matches:
        raise SystemExit(f"No baseline result files match {spec}")
    return matches


# ---------------------------------------------------------------- raters
def load_rater_models():
    """Import algorithms/rater_models.py without importing the algorithms package."""
    spec = importlib.util.spec_from_file_location("rater_models", REPO / "algorithms" / "rater_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_rater(capacity, input_dim):
    models = load_rater_models()
    if capacity == "small":
        return models.EmbeddingRaterSmall(input_dim)
    if capacity == "medium":
        return models.EmbeddingRaterMedium(input_dim)
    raise ValueError(f"Unknown rater capacity {capacity!r}")


def infer_capacity(state):
    first = state.get("net.1.weight")
    if first is None:
        raise ValueError("Cannot infer rater capacity from checkpoint.")
    return "small" if first.shape[0] == 256 else "medium"


def load_rater_checkpoint(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["rater_sd"]
    capacity = str(ckpt.get("rater_capacity") or infer_capacity(state)).lower()
    rater = build_rater(capacity, state["net.1.weight"].shape[1])
    rater.load_state_dict(state)
    rater.eval()
    return rater, ckpt


def rater_weights(raw, normalization="global", transform="sigmoid", temperature=1.0, scale=1.0):
    """Same mapping as the final-classifier notebooks (global normalization over the split)."""
    raw = np.asarray(raw, dtype=np.float64)
    if normalization == "global":
        z = (raw - raw.mean()) / (raw.std() + 1e-6)
    elif normalization == "none":
        z = raw
    else:
        raise ValueError("normalization must be global or none for split-level analysis")
    if transform == "sigmoid":
        return scale / (1.0 + np.exp(-z / temperature))
    if transform == "softmax":
        e = np.exp((z - z.mean()) / temperature - ((z - z.mean()) / temperature).max())
        return scale * e / e.sum()
    raise ValueError(transform)


def load_embedding_cache(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: payload[k] for k in ("embeddings", "labels", "groups", "attrs")}, payload.get("cache_metadata", {})


def mean_std(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, int(len(values))
