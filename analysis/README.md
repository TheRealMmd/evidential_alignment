# Analysis and controls for the FeatureRater paper

Everything here reads artifacts the notebooks and `main.py` already write. Nothing trains
except `run_baselines.py`, which launches the repository's own baselines.
Scripts need `numpy pandas scipy matplotlib torch` (all present in Colab).
`plot_final_classifier_multiseed.py` (moved here from the repository root) re-plots completed final-classifier runs.

## Stage 1: rater meta-training (`algorithms/rater.py`)
New behaviour is **off by default**; existing runs are bit-identical (checked on a CPU
synthetic run: same outer losses, metrics and exported rater). Turn options on with
environment variables (cell 6 of `notebooks/Train Rater/Run_Rater_Classwise_Outer_Loss_ERM_Embeddings.ipynb`):

| Variable | Effect |
|---|---|
| `RATER_UNIFORM_REFERENCE=1` | Uniform-weight reference heads: same init and batches as each inner head, constant weight = batch-mean rater weight. Logged as `reference_*` in `metrics.csv` and `reference_outer_loss` in `meta_history.csv`. |
| `RATER_REFRESH_STEPS=K` | Staggered re-initialization of the inner heads every K meta steps (the paper's "periodically re-initialized"). 0 = never (previous behaviour). |
| `RATER_SELECTION=fresh_heads` | Export the rater whose weights give the lowest outer objective when *fresh* heads are trained for `RATER_FRESH_HEAD_EPOCHS` epochs. Not confounded by how long the persistent heads have trained. Also `last`; default `min_outer_loss`. |
| `RATER_FRESH_HEAD_EVAL=1` | Log the fresh-head objective at every eval step without selecting by it. |
| `RATER_OUTER_SPLIT=train_no_aug_subset2` | Use a held-out part of the (skewed) training set as D_meta. Run with `--split_train 0.8 --train_split train_subset1`. |
| `RATER_OUTER_MAX_SAMPLES=N` | Seeded subsample of D_meta (meta-set size sweep; compare with `python algorithms/rater.py --runs ... --compare calibration_size`). |
| `RATER_ERM_FINGERPRINT=<12 hex>` (+ `RATER_ERM_MD5`) | Keep the embedding-cache name of an ERM checkpoint that was moved on Drive (its path is part of the computed fingerprint). The md5 check refuses a different file. Unset = previous behaviour. |
| `RATER_OUTER_LOSS=top_loss` | CVaR over all outer examples: mean of the top `RATER_OUTER_TOP_LOSS_PERCENT` losses, classes ignored. |
| `RATER_OUTER_LOSS=worst_class_top_loss` | Worst class's top-loss tail mean (instead of the class average). |
| `RATER_OUTER_LOSS=classwise_logsumexp` | Normalized log-sum-exp within each class (`RATER_OUTER_LSE_BETA`), classes averaged equally. |
| `RATER_OUTER_LOSS=mean_plus_std` | Mean CE + `RATER_OUTER_STD_COEF` (default 1.0) × std of CE. |

Always logged now: rater gradient norm before clipping, whether clipping fired, update norm,
weighted inner loss, weight ESS fraction and normalized entropy (`meta_history.csv`); weight
rank stability between snapshots and ESS (`score_correlations.csv`); population plot
`plots/meta_metrics/population.png` and `plots/meta_selection.png` (selected step marked);
`selected_is_initial_rater` + a WARNING in `summary.txt` when the exported rater is the
untrained one; compute (wall-clock, peak GPU memory, GPU) and provenance (git commit,
sha256 of the executed `rater.py`) in `final_rater.pt` and `plots/data/summary.json`;
full-dataset indices of D_meta in `plots/data/sample_indices/full_split_indices.json`.
The embedding cache name now uses the dataset name (`resnet50_<dataset>_erm_<fp>`), so other
datasets work; Waterbirds caches keep their old names.

## Stage 2 notebooks
* `Final_Classifier_Fixed_Random_Weight_MultiSeed.ipynb`: `WEIGHT_MODE = "constant"` is
  **uniform-weight retraining** with the same protocol (weight 0.5 for every sample).
  Set `VARY_DATA_SPLIT_WITH_SEED` to the same value as the rater notebook (it is `True` here and
  `False` there by default) or the seeds cannot be paired.
* Untrained-rater control: export a rater at initialization and point `RATER_CHECKPOINT` at it.

## Scripts
```bash
cd /content/evidential_alignment
D=/content/drive/MyDrive/evidential_alignment_colab
CACHE=$D/embeddings_waterbirds_erm/waterbirds/resnet50_waterbirds_erm_1055b1e8de29/resolution_224
ERM=$D/erm_base_models/waterbirds_erm_resnet50_imagenet_100ep_best_val_acc.pt
WB=/content/datasets/waterbird_complete95_forest2water2

# Which raters were exported at step 1 (= untrained)?
python analysis/rater_inventory.py --root $D --out $D/tables/raters.csv

# Untrained-rater control (3 init seeds), then run the multi-seed notebook on each
for s in 0 1 2; do python analysis/export_init_rater.py --capacity medium --init-seed $s \
  --erm-fingerprint 1055b1e8de29 --out $D/untrained_raters/medium_seed$s/final_rater.pt; done

# D_meta composition and overlap with the classifier's splits
python analysis/meta_set_overlap.py --metadata $WB/metadata.csv \
  --rater-split-pickle $WB/val_split_0.40_seed0.pickle --classifier-split-val 0.6 \
  --classifier-split-seeds 0 --out $D/tables/meta_overlap

# Baselines in the same setting (same ERM, same val_subset1/val_subset2 as the notebooks)
python analysis/run_baselines.py --methods afr evidential_alignment dfr --seeds 0 1 2 3 4 \
  --erm-model $ERM --dataset-dir $WB --patch-config --match-notebook-split fixed:0 --out-dir $D/baselines

# Main table and paired comparison (same label-free selection rule everywhere)
python analysis/main_table.py --rule val_worst_class_accuracy \
  --exp "FeatureRater=<rater experiment root>" --exp "Uniform retraining=<constant root>" \
  --exp "Random weights=<random root>" --exp "Untrained rater=<init root>" \
  --baseline "AFR=$D/baselines/afr" --baseline "Evidential Alignment=$D/baselines/evidential_alignment" \
  --baseline "DFR (group labels)=$D/baselines/dfr_group_balanced" --out $D/tables/main
python analysis/compare.py --reference "FeatureRater=<root>" --control "Uniform=<root>" \
  --control "Untrained rater=<root>" --out $D/tables/paired

# Mechanism: weights vs ERM loss inside each group; top/bottom images
python analysis/weight_vs_error.py --cache-dir $CACHE --erm-model $ERM --rater <final_rater.pt> \
  --split train_no_aug --out $D/tables/wve_train
python analysis/top_bottom_examples.py --cache-dir $CACHE --rater <final_rater.pt> --dataset-dir $WB \
  --split train_no_aug --out $D/figures/top_bottom_train

# Source ERM quality vs gain (one --point per ERM checkpoint)
python analysis/erm_quality_vs_gain.py --point ep100 $ERM $CACHE <fr_root> <uniform_root> --out $D/figures/erm_quality

# Compute cost
python analysis/collect_compute.py --embedding-cache $CACHE --rater-run "FeatureRater=<rater run>" \
  --exp "FeatureRater classifier=<root>" --baseline "AFR=$D/baselines/afr" --out $D/tables/compute
```
Notes: GroupDRO is not implemented in this repository. JTT (`--methods jtt`) trains end to end on
images and is slow. Baseline peak memory comes from `nvidia-smi` (whole GPU), the others from
`torch.cuda.max_memory_allocated`.
