"""Compute-cost table across stages: embedding extraction, rater meta-training,
final classifier (per seed; wall clock only), and baselines.

  python analysis/collect_compute.py \
     --embedding-cache <cache>/resolution_224 \
     --rater-run "FeatureRater=<drive rater run dir>" \
     --exp "FeatureRater classifier=<final-classifier root>" --exp "Uniform=<root>" \
     --baseline "AFR=<baselines>/afr" --baseline "EA=<baselines>/evidential_alignment" \
     --out compute
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import mean_std, parse_named, write_csv  # noqa: E402


def row(stage, name, seconds, mem_bytes, gpu, n):
    s_mean, s_std, _ = mean_std(seconds)
    mem = [m for m in mem_bytes if m is not None]
    return {"stage": stage, "method": name, "n_runs": n,
            "wall_clock_minutes_mean": s_mean / 60, "wall_clock_minutes_std": s_std / 60,
            "peak_gpu_memory_gb_max": (max(mem) / 1024 ** 3) if mem else float("nan"),
            "gpu": ", ".join(sorted({g for g in gpu if g})) or "unknown"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-cache", action="append", default=[])
    p.add_argument("--rater-run", action="append", default=[], help="NAME=rater output dir (repeatable)")
    p.add_argument("--exp", action="append", default=[], help="NAME=final-classifier experiment root")
    p.add_argument("--baseline", action="append", default=[], help="NAME=folder with seed*_compute.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for cache in args.embedding_cache:
        info = json.loads((Path(cache) / "_EXTRACTION_COMPUTE.json").read_text())
        rows.append(row("embedding extraction", Path(cache).name, [info["embedding_extraction_seconds"]],
                        [info.get("peak_gpu_memory_bytes")], [info.get("gpu_name")], 1))
    for name, run in parse_named(args.rater_run):
        summary = json.loads((Path(run) / "plots" / "data" / "summary.json").read_text())
        compute = summary.get("compute")
        if not compute:
            raise SystemExit(f"{run} has no compute record (run made before the compute logging was added).")
        rows.append(row("rater meta-training", name, [compute["meta_training_seconds"]],
                        [compute.get("peak_gpu_memory_bytes")], [compute.get("gpu_name")], 1))
    for name, root in parse_named(args.exp):
        seeds = sorted(Path(root).glob("seed_*/summary.json"))
        infos = [json.loads(s.read_text()) for s in seeds]
        rows.append(row("final classifier (per seed)", name, [i.get("elapsed_seconds") for i in infos],
                        [i.get("peak_gpu_memory_bytes") for i in infos], [i.get("gpu_name") for i in infos], len(infos)))
    for name, folder in parse_named(args.baseline):
        infos = [json.loads(s.read_text()) for s in sorted(Path(folder).glob("seed*_compute.json"))]
        mem = [i["peak_gpu_memory_mib_nvidia_smi"] * 1024 ** 2 if i.get("peak_gpu_memory_mib_nvidia_smi") else None
               for i in infos]
        rows.append(row("baseline (per seed, whole run)", name, [i["wall_clock_seconds"] for i in infos],
                        mem, [], len(infos)))
    out = Path(args.out)
    write_csv(out.with_suffix(".csv"), rows)
    for r in rows:
        print(f"{r['stage']:<32} {r['method']:<24} n={r['n_runs']:<3} "
              f"{r['wall_clock_minutes_mean']:.1f} ± {r['wall_clock_minutes_std']:.1f} min  "
              f"peak {r['peak_gpu_memory_gb_max']:.2f} GB  {r['gpu']}")
    print("wrote", out.with_suffix(".csv"),
          "(rater/classifier memory: torch.cuda.max_memory_allocated; baselines: nvidia-smi whole-GPU)")


if __name__ == "__main__":
    main()
