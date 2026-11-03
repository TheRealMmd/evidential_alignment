"""Run the repo's baselines in the SAME setting as the FeatureRater final classifier.

Same frozen ERM checkpoint, same 2048-d embeddings, training on val_subset1 and
checkpoint selection on val_subset2 with SPLIT_VAL=0.6 by default, and (with
--match-notebook-split) the exact validation split the final-classifier
notebooks use: RandomState(split_seed) over the validation set, written as the
val_split_<r>_seed<s>.pickle that data/data_utils.py reads.

Each run writes <out-dir>/<method>/seed<s>.txt (main.py --result_path; parse
with main_table.py/compare.py) and seed<s>_compute.json (wall-clock, peak GPU
memory sampled with nvidia-smi every second: whole-GPU usage, approximate).

GroupDRO is not implemented in this repository, so it is not offered here.
DFR runs with --group_balanced True: it uses validation group labels and is a
group-supervised upper bound, labelled dfr_group_balanced.

Colab example:
  python analysis/run_baselines.py --methods afr evidential_alignment dfr \
      --seeds 0 1 2 3 4 --erm-model "$ERM_MODEL" \
      --dataset-dir /content/datasets/waterbird_complete95_forest2water2 --patch-config \
      --match-notebook-split fixed:0 --out-dir /content/drive/MyDrive/FeatureRater/waterbirds/erm-r50-e100-s0/baselines
"""

import argparse
import json
import pickle
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.meta_set_overlap import notebook_val_split  # noqa: E402

METHOD_ARGS = {
    # README of this repository (Evidential Alignment, KDD 2025).
    "evidential_alignment": ["--batch_size", "128", "--optimizer_cls", "sgd",
                             "--optimizer_cls_kwargs", "lr=0.01", "weight_decay=0", "momentum=0.0",
                             "--edl_lr", "0.01", "--alignment_lr", "0.01", "--reg_weight", "0.1",
                             "--annealing_step", "10", "--edl_epochs", "50", "--alignment_epochs", "100",
                             "--balance_classes", "True"],
    "afr": [],
    "dfr": ["--group_balanced", "True"],
    # End-to-end image training (slow): trains on the full training split.
    "jtt": ["--train_split", "train"],
}
RESULT_NAME = {"dfr": "dfr_group_balanced"}


def patch_config(dataset, dataset_dir):
    path = REPO / "config.py"
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(rf'("{re.escape(dataset)}"\s*:\s*)["\'][^"\']*["\']',
                     lambda m: m.group(1) + json.dumps(str(dataset_dir)), text, count=1)
    if n != 1:
        raise SystemExit(f"Could not find the {dataset!r} entry in config.py dataset_paths")
    path.write_text(new, encoding="utf-8")
    print(f"config.py: {dataset} -> {dataset_dir}")


def write_split(dataset_dir, split_val, run_seed, split_seed, force):
    import pandas as pd
    meta = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    n_val = int((meta["split"] == 1).sum())
    first, second = notebook_val_split(n_val, split_val, split_seed)
    path = Path(dataset_dir) / f"val_split_{split_val:.2f}_seed{run_seed}.pickle"
    if path.exists():
        with open(path, "rb") as fin:
            old = pickle.load(fin)
        if np.array_equal(np.asarray(old[0]), first) and np.array_equal(np.asarray(old[1]), second):
            return path
        if not force:
            raise SystemExit(f"{path} exists with a DIFFERENT split. Rerun with --force to replace it "
                             f"(the old file is kept as .bak).")
        path.rename(path.with_suffix(".pickle.bak"))
    with open(path, "wb") as fout:
        pickle.dump((first, second), fout)
    return path


class GpuPeak:
    def __init__(self):
        self.peak_mib, self._stop = None, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                used = max(int(x) for x in out.split())
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except Exception:
                return
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["afr", "evidential_alignment", "dfr"], choices=sorted(METHOD_ARGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--erm-model", required=True)
    p.add_argument("--dataset", default="waterbirds")
    p.add_argument("--dataset-dir", help="needed for --match-notebook-split and --patch-config")
    p.add_argument("--split-val", type=float, default=0.6)
    p.add_argument("--train-split", default="val_subset1")
    p.add_argument("--match-notebook-split", default="fixed:0",
                   help="fixed:<split_seed> (VARY_DATA_SPLIT_WITH_SEED=False), vary (split seed = run seed), or none")
    p.add_argument("--force", action="store_true", help="replace an existing, different split pickle")
    p.add_argument("--patch-config", action="store_true", help="point config.py dataset_paths at --dataset-dir")
    p.add_argument("--extra", action="append", default=[], help='METHOD="--flag value ..." appended to that method')
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    extra = {}
    for item in args.extra:
        method, flags = item.split("=", 1)
        extra.setdefault(method, []).extend(shlex.split(flags))
    if args.patch_config and not args.dry_run:
        patch_config(args.dataset, args.dataset_dir)
    out_dir = Path(args.out_dir)
    for seed in args.seeds:
        if args.match_notebook_split != "none":
            if not args.dataset_dir:
                raise SystemExit("--match-notebook-split needs --dataset-dir")
            split_seed = seed if args.match_notebook_split == "vary" else int(args.match_notebook_split.split(":")[1])
            if not args.dry_run:
                print("validation split:", write_split(args.dataset_dir, args.split_val, seed, split_seed, args.force))
        for method in args.methods:
            name = RESULT_NAME.get(method, method)
            method_dir = out_dir / name
            method_dir.mkdir(parents=True, exist_ok=True)
            result = method_dir / f"seed{seed}.txt"
            cmd = [sys.executable, "-u", "main.py", "--algorithm", method, "--dataset", args.dataset,
                   "--backbone", "resnet50", "--pretrained", "True", "--erm_model", args.erm_model,
                   "--mode", "train", "--split_val", str(args.split_val), "--train_split", args.train_split,
                   "--test_split", "test", "--seed", str(seed), "--save_folder", str(method_dir / f"run_seed{seed}"),
                   "--result_path", str(result)] + METHOD_ARGS[method] + extra.get(method, [])
            print("$", " ".join(shlex.quote(c) for c in cmd))
            if args.dry_run:
                continue
            if result.exists():
                result.rename(result.with_suffix(f".txt.{int(time.time())}.old"))
            started = time.monotonic()
            with GpuPeak() as gpu:
                code = subprocess.run(cmd, cwd=str(REPO)).returncode
            info = {"method": name, "seed": seed, "returncode": code, "wall_clock_seconds": time.monotonic() - started,
                    "peak_gpu_memory_mib_nvidia_smi": gpu.peak_mib, "command": cmd,
                    "note": "wall clock includes data loading and the repo's own embedding extraction/caching"}
            (method_dir / f"seed{seed}_compute.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if code != 0:
                raise SystemExit(f"{method} seed {seed} failed with exit code {code}")


if __name__ == "__main__":
    main()
