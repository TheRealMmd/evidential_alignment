"""Qualitative figure: the highest- and lowest-weighted training images.

Maps embedding-cache rows back to images through metadata.csv (cache rows follow
the metadata order of the split; the script checks labels and groups match and
stops if they do not).

  python analysis/top_bottom_examples.py --cache-dir <embedding cache>/resolution_224 \
      --rater <final_rater.pt> --dataset-dir /content/datasets/waterbird_complete95_forest2water2 \
      --split train_no_aug --n 12 --out top_bottom_train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import load_rater_checkpoint, rater_weights, write_csv  # noqa: E402
from analysis.meta_set_overlap import notebook_val_split  # noqa: E402
from analysis.weight_vs_error import load_split  # noqa: E402

GROUP_NAMES = {0: "landbird/land", 1: "landbird/water", 2: "waterbird/land", 3: "waterbird/water"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--rater", required=True)
    p.add_argument("--dataset-dir", required=True, help="folder with metadata.csv and the images")
    p.add_argument("--split", default="train_no_aug", choices=("train_no_aug", "val", "test", "classifier_train"))
    p.add_argument("--classifier-split-val", type=float, default=0.6)
    p.add_argument("--classifier-split-seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = load_split(args.cache_dir, args.split, args.classifier_split_val, args.classifier_split_seed)
    meta = pd.read_csv(Path(args.dataset_dir) / "metadata.csv")
    split_id = {"train_no_aug": 0, "val": 1, "test": 2, "classifier_train": 1}[args.split]
    rows = meta[meta["split"] == split_id].reset_index(drop=True)
    if args.split == "classifier_train":
        idx, _ = notebook_val_split(len(rows), args.classifier_split_val, args.classifier_split_seed)
        rows = rows.iloc[idx].reset_index(drop=True)
    n_places = meta["place"].nunique()
    groups = (rows["y"].values * n_places + rows["place"].values).astype(int)
    if len(rows) != len(data["labels"]) or not np.array_equal(rows["y"].values, data["labels"].numpy()) \
            or not np.array_equal(groups, data["groups"].numpy()):
        raise SystemExit("Cache rows do not match metadata order for this split; refusing to label images.")

    rater, ckpt = load_rater_checkpoint(args.rater)
    with torch.no_grad():
        z = data["embeddings"].float()
        raw = torch.cat([rater(z[i:i + 1024]) for i in range(0, len(z), 1024)]).numpy()
    weights = rater_weights(raw, "global", "sigmoid", args.temperature)
    order = np.argsort(-weights, kind="stable")
    picks = {"top": order[:args.n], "bottom": order[::-1][:args.n]}

    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, args.n, figsize=(1.9 * args.n, 4.6))
    table = []
    for r, (kind, ids) in enumerate(picks.items()):
        for c, i in enumerate(ids):
            ax = axes[r, c]
            ax.imshow(Image.open(Path(args.dataset_dir) / rows.loc[i, "img_filename"]).convert("RGB"))
            ax.set_title(f"{GROUP_NAMES.get(groups[i], groups[i])}\nw={weights[i]:.2f}", fontsize=7)
            ax.axis("off")
            table.append({"rank_kind": kind, "rank": c + 1, "row": int(i), "img_filename": rows.loc[i, "img_filename"],
                          "group": int(groups[i]), "label": int(rows.loc[i, "y"]), "weight": float(weights[i]),
                          "raw_score": float(raw[i])})
        axes[r, 0].text(-0.15, 0.5, "highest" if kind == "top" else "lowest", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle(f"{args.split}: highest- and lowest-weighted images (rater best_meta_step={ckpt.get('best_meta_step')})",
                 fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"), dpi=170)
    write_csv(out.with_suffix(".csv"), table)
    print("wrote", out.with_suffix(".png"), out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
