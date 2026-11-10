"""Untrained-rater control: export the rater network at initialization.

Writes a final_rater.pt with the SAME format the final-classifier notebooks
load (rater_sd, rater_capacity, weighting, temperature), but
with weights straight from initialization (best_meta_step = 0,
outer_loss_type = "none_untrained"). Point RATER_CHECKPOINT in
Frozen_Rater_Final_Classifier_MultiSeed_MultiSelection.ipynb at it to run the
control through the identical classifier protocol. Use several --init-seed
values; a random MLP on ERM embeddings already gives group-dependent weights.

  python analysis/export_init_rater.py --capacity medium --init-seed 0 \
      --out /content/drive/.../untrained_raters/medium_seed0/final_rater.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import build_rater  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capacity", choices=("small", "medium"), default="medium")
    p.add_argument("--input-dim", type=int, default=2048)
    p.add_argument("--init-seed", type=int, required=True)
    p.add_argument("--weighting", default="sigmoid")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.init_seed)
    rater = build_rater(args.capacity, args.input_dim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "rater_sd": {k: v.detach().cpu().clone() for k, v in rater.state_dict().items()},
        "rater_capacity": args.capacity, "feature_dim": args.input_dim,
        "weighting": args.weighting, "temperature": args.temperature,
        "outer_loss_type": "none_untrained", "best_meta_step": 0, "best_outer_loss": None,
        "meta_steps": 0, "selected_is_initial_rater": True, "init_seed": args.init_seed,
        "note": "Untrained-rater control exported by analysis/export_init_rater.py; never meta-trained.",
    }, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
