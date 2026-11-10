"""Which rater checkpoint was exported? List every final_rater.pt under a folder.

Flags raters whose exported checkpoint is the initial, never-updated network:
best_meta_step == 1 under the min-outer-loss rule (the pre-update rater of
step 1), or 0 under fresh-head selection / exported controls. Older checkpoints
carry no 'selection_rule'; they used min_outer_loss.

  python analysis/rater_inventory.py --root /content/drive/MyDrive/FeatureRater --out raters.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import write_csv  # noqa: E402


def describe(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rule = ckpt.get("selection_rule") or "min_outer_loss (pre-selection_rule export)"
    step = ckpt.get("best_meta_step")
    initial = ckpt.get("selected_is_initial_rater")
    if initial is None and step is not None:
        initial = (step == 1 and rule.startswith("min_outer_loss")) or step == 0
    return {
        "path": str(path), "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"),
        "outer_loss_type": ckpt.get("outer_loss_type"),
        "outer_top_loss_percent": ckpt.get("outer_top_loss_percent_per_class"),
        "selection_rule": rule, "best_meta_step": step, "meta_steps": ckpt.get("meta_steps"),
        "best_outer_loss": ckpt.get("best_outer_loss"), "selected_is_initial_rater": initial,
        "weighting": ckpt.get("weighting"), "temperature": ckpt.get("temperature"),
        "capacity": ckpt.get("rater_capacity"),
        "refresh_every": ckpt.get("refresh_every"), "git_commit": (ckpt.get("provenance") or {}).get("git_commit"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()
    rows = [describe(path) for path in sorted(Path(args.root).rglob("final_rater.pt"))]
    if not rows:
        raise SystemExit(f"No final_rater.pt under {args.root}")
    for r in rows:
        flag = "UNTRAINED (initial rater)" if r["selected_is_initial_rater"] else ""
        print(f"{r['best_meta_step']!s:>5}/{r['meta_steps']!s:<5} {str(r['outer_loss_type']):<20} {flag:<26} {r['path']}")
    if args.out:
        write_csv(args.out, rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
