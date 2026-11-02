"""D_meta composition and its overlap with the final classifier's splits.

The rater's D_meta (val_subset1 at --split_val 0.4, drawn by data_utils with
np.random.shuffle) and the final classifier's splits (val_subset1/val_subset2
at SPLIT_VAL 0.6, drawn with RandomState(split_seed) in the notebook) both come
from the same official validation set. This script reports, without training:
  * group counts and proportions of D_meta, the classifier's training set and its
    selection set (and of the full train/val sets for reference);
  * how many D_meta images are also in the classifier's training / selection set.

D_meta indices come from a rater run made with the current rater.py
(plots/data/sample_indices/full_split_indices.json) or, for older runs, from the
cached split file data_utils wrote next to the dataset
(<dataset_dir>/val_split_0.40_seed0.pickle).

  python analysis/meta_set_overlap.py --metadata /content/datasets/waterbird_complete95_forest2water2/metadata.csv \
      --rater-split-pickle /content/datasets/waterbird_complete95_forest2water2/val_split_0.40_seed0.pickle \
      --classifier-split-val 0.6 --classifier-split-seeds 0 --out overlap
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import write_csv  # noqa: E402


def notebook_val_split(n, split_val, split_seed):
    """Exactly ensure_val_subsets() in the final-classifier notebooks."""
    indices = np.arange(n)
    np.random.RandomState(split_seed).shuffle(indices)
    k = int(n * split_val)
    return indices[:k], indices[k:]


def composition(name, groups, n_groups):
    counts = np.bincount(groups, minlength=n_groups)
    row = {"set": name, "n": int(len(groups))}
    for g in range(n_groups):
        row[f"group_{g}_n"] = int(counts[g])
        row[f"group_{g}_share"] = float(counts[g] / max(len(groups), 1))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="metadata.csv (columns y, place, split)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rater-run", help="rater output dir with plots/data/sample_indices/full_split_indices.json")
    src.add_argument("--rater-split-pickle", help="val_split_<r>_seed<s>.pickle written by data_utils")
    p.add_argument("--classifier-split-val", type=float, default=0.6)
    p.add_argument("--classifier-split-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.metadata)
    n_places = meta["place"].nunique()
    groups_all = (meta["y"].values * n_places + meta["place"].values).astype(int)
    n_groups = int(groups_all.max()) + 1
    val_groups = groups_all[meta["split"].values == 1]
    train_groups = groups_all[meta["split"].values == 0]

    if args.rater_run:
        info = json.loads((Path(args.rater_run) / "plots" / "data" / "sample_indices"
                           / "full_split_indices.json").read_text())
        if "val_indexes1" not in info:
            raise SystemExit("This rater run did not use a validation subset as D_meta "
                             "(no val_indexes1); its D_meta cannot overlap the validation splits.")
        meta_idx = np.asarray(info["val_indexes1"], dtype=int)
    else:
        with open(args.rater_split_pickle, "rb") as fin:
            meta_idx = np.asarray(pickle.load(fin)[0], dtype=int)

    rows = [composition("train (all)", train_groups, n_groups),
            composition("validation (all)", val_groups, n_groups),
            composition("D_meta (rater outer split)", val_groups[meta_idx], n_groups)]
    overlaps = []
    meta_set = set(meta_idx.tolist())
    for seed in args.classifier_split_seeds:
        tr, sel = notebook_val_split(len(val_groups), args.classifier_split_val, seed)
        rows.append(composition(f"classifier train (val_subset1, split seed {seed})", val_groups[tr], n_groups))
        rows.append(composition(f"classifier selection (val_subset2, split seed {seed})", val_groups[sel], n_groups))
        o_tr, o_sel = len(meta_set & set(tr.tolist())), len(meta_set & set(sel.tolist()))
        overlaps.append({"classifier_split_seed": seed, "d_meta_size": len(meta_idx),
                         "classifier_train_size": len(tr), "classifier_selection_size": len(sel),
                         "d_meta_in_classifier_train": o_tr, "d_meta_in_classifier_selection": o_sel,
                         "share_of_selection_set_seen_by_rater": o_sel / len(sel),
                         "share_of_classifier_train_seen_by_rater": o_tr / len(tr)})
        print(f"split seed {seed}: {o_sel}/{len(sel)} selection images and {o_tr}/{len(tr)} "
              f"classifier-training images are also in D_meta ({len(meta_idx)} images).")
    out = Path(args.out)
    write_csv(out.parent / (out.name + "_composition.csv"), rows)
    write_csv(out.parent / (out.name + "_overlap.csv"), overlaps)
    for r in rows:
        shares = " ".join(f"g{g}={100 * r[f'group_{g}_share']:.1f}%" for g in range(n_groups))
        print(f"{r['set']:<52} n={r['n']:<5} {shares}")
    print("wrote", out.parent / (out.name + "_composition.csv"), out.parent / (out.name + "_overlap.csv"))


if __name__ == "__main__":
    main()
