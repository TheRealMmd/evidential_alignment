# FeatureRater

Meta-learned, embedding-based sample weighting for group robustness without group labels.

A frozen feature extractor maps every input to an embedding. A small **rater** network scores
each embedding, and the scores become sample weights. Following the meta-learning principle of
DataRater (Calian et al., 2025), the rater is trained by differentiating a group-free held-out
loss through the weighted training of several linear classification heads. The learned rater is
then frozen and used to weight the training of a newly initialized final classifier. Group labels
are used only for evaluation.

## Pipeline

| Stage | What it does | Notebook | Code |
|---|---|---|---|
| 0. ERM backbone | Fine-tunes an ImageNet ResNet-50 end to end with plain ERM; its checkpoint defines the frozen embeddings. | `notebooks/ERM Backbone/Train_Waterbirds_ERM_Base_Model_For_Evidential_Alignment.ipynb` | `main.py --algorithm erm` (`algorithms/erm.py`) |
| 1. Rater meta-training | Caches the ERM embeddings once, then meta-trains the rater (bilevel, selectable outer loss) and exports `final_rater.pt`. | `notebooks/Train Rater/Run_Rater_Classwise_Outer_Loss_ERM_Embeddings.ipynb` | `main.py --algorithm rater` (`algorithms/rater.py`, `algorithms/rater_models.py`) |
| 2. Final classifier | Trains a fresh `Linear(2048, C)` on frozen embeddings with the frozen rater's weights, over several seeds, with several checkpoint-selection rules. | `notebooks/Final Classifier/Frozen_Rater_Final_Classifier_MultiSeed_MultiSelection.ipynb` | self-contained notebook |

Controls for stage 2 (same protocol, no rater): `Final_Classifier_Fixed_Random_Weight_MultiSeed.ipynb`
(random weights, or `WEIGHT_MODE = "constant"` for uniform-weight retraining) and
`Final_Classifier_Fixed_Minority_Group_Weight_MultiSeed.ipynb` (fixed minority-group upweighting; uses
training group labels by design). `notebooks/Analysis/FeatureRater_TSNE_Visualization.ipynb` draws a
t-SNE of the embeddings coloured by group and by rater weight.

The notebooks run in Google Colab: they clone this repository, read/write a Google Drive folder
(`evidential_alignment_colab/`), and pass rater settings through `RATER_*` environment variables.

## Repository layout

```
main.py, config.py, utils.py     entry point, command-line options, logging/seeding
algorithms/
  rater.py, rater_models.py      FeatureRater meta-training (stage 1)
  erm.py                         ERM backbone (stage 0)
  afr.py, dfr.py, jtt.py, evidential_alignment.py   baselines
  algorithm.py, register.py, utils.py               shared base code
data/                            Waterbirds, CelebA (wb_data.py), MultiNLI, CivilComments, samplers
models/                          ResNet-50 (resnet.py), timm/ViT backbones, BERT
notebooks/                       the Colab notebooks above
analysis/                        tables, paired comparisons, controls, baseline launcher (see analysis/README.md)
```

## Setup

```bash
pip install -r requirements.txt
```
Set each dataset folder in `config.py` (`dataset_paths`). Waterbirds:
<https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz>; CelebA:
<https://www.kaggle.com/datasets/jessicali9530/celeba-dataset> with the DFR metadata
(<https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/celeba_metadata.csv>);
CivilComments downloads through `wilds`; MultiNLI via `download` in `data/multinli.py`.

## Running without notebooks

```bash
# Stage 0: ERM backbone
python main.py --dataset waterbirds --algorithm erm --backbone resnet50 --pretrained True \
  --batch_size 128 --optimizer sgd --optimizer_kwargs lr=0.001 momentum=0.9 weight_decay=0.0001 \
  --scheduler none --epoch 100 --save_folder runs/erm --train_split train --mode train

# Stage 1: cache embeddings once, then meta-train the rater
export RATER_ERM_MODEL=runs/erm/<run>/best_val_acc_model.pt RATER_EMBEDDING_CACHE_ROOT=runs/embeddings
RATER_PRECOMPUTE_EMBEDDINGS_ONLY=1 python main.py --dataset waterbirds --algorithm rater --backbone resnet50 \
  --pretrained True --erm_model $RATER_ERM_MODEL --split_val 0.4 --mode train --save_folder runs/pre
RATER_OUTER_LOSS=classwise_top_loss RATER_WEIGHTING=sigmoid RATER_TEMPERATURE=1.0 RATER_META_STEPS=500 \
  python main.py --dataset waterbirds --algorithm rater --backbone resnet50 --pretrained True \
  --erm_model $RATER_ERM_MODEL --split_val 0.4 --train_split train --epoch 500 --mode train --save_folder runs/rater
```
Outer objectives (`RATER_OUTER_LOSS` / `OUTER_LOSS`). Class labels only: `classwise_top_loss` (default),
`class_balanced`, `worst_class_loss`, `logsumexp`, `top_loss` (CVaR over all examples), `worst_class_top_loss`,
`classwise_logsumexp`, `mean_plus_std` (mean + `RATER_OUTER_STD_COEF` × std), `standard_ce`. With group
labels (reference rows only): `worst_group_loss`, `group_balanced_loss`.

Stage 2 is the final-classifier notebook. Rater options (outer loss, weighting, temperature, learning
rates, uniform reference heads, head refresh, rater selection rule, D_meta options) are listed in
`analysis/README.md`; all new options are off by default.

## Drive layout (experiment outputs)
The notebooks read and write under `MyDrive/evidential_alignment_colab/`:
```
archives/                         dataset tarballs
waterbirds/<ERM_ID>/              one folder per ERM backbone (e.g. erm-r50-e100-s0)
  <stable ERM checkpoint>.pt      + manifest, results, Evidential Alignment feature files
  erm_..._pretrained/             the ERM training run
  embeddings/waterbirds/resnet50_waterbirds_erm_<fingerprint>/resolution_224/
  raters/<rater run>/             final_rater.pt, logs, plots
    classifiers/<run>/            final classifiers trained with this rater
  controls/<run>/                 no-rater controls (random, constant, minority weights)
  baselines/  analysis/
unsorted/                         older runs whose ERM or rater is not recorded
```
Each notebook sets `ERM_ID` and falls back to the pre-November paths when the new folders do not
exist. A moved ERM checkpoint keeps its embedding cache through `RATER_ERM_FINGERPRINT` (+ `RATER_ERM_MD5`),
set in the Train Rater notebook (`ERM_PINS`).

## What each stage saves

**Stage 0 (ERM)**, in `<save_folder>/erm_<dataset>_resnet50_..._pretrained/`: `best_val_acc_model.pt`
(used downstream), `best_val_{worst_cls_acc,worst_group_acc,avg_cls_diff}_model.pt`, `latest_model.pt`,
`model_epoch<k>.pt` (every `--save_freq`), `erm_train_val_test_history.csv` (per-epoch metrics), `log.txt`,
and the `--result_path` text file.
Per-group results of the checkpoint used downstream are written in stage 1 (ERM record, below).

**Stage 1 (rater)**, in the run folder:
- `final_rater.pt` (rater weights + settings, selected step, compute, provenance, ERM record), `summary.txt`, `log.txt`
- **ERM record** (how the frozen ERM checkpoint itself does: accuracy, WGA, average and adjusted average
  accuracy, per-group and per-class accuracy, mean CE): `ERM_RECORD.json`/`.csv` next to the embedding
  cache (every cached split; written by the embedding step), `plots/data/erm_record.{json,csv}` for the
  splits the run uses, and a summary in `log.txt`
- `plots/data/`: `metrics.csv` and `group_metrics.csv` (accuracy, WGA, loss, worst-group loss of every
  inner head on train / D_meta / held-out val / test over meta-steps), `meta_history.csv` (outer loss and
  meta-update diagnostics per step), `group_statistics.csv` (per-group weight statistics over time),
  `score_correlations.csv`, `outer_objective_details.jsonl`, `score_snapshots/*.npz` (per-sample scores and
  weights), `run_config.json`, `summary.json`, `sample_indices/`
- plots: `meta_metrics/inner_<i>.png` and `population.png` (inner-head curves), `meta_training_curves/`
  (outer loss, per-group mean score and weight), `raw_score_histogram/` and `final_score_histogram_*`
  (score/weight histograms by group, per evaluation step), `meta_rate_distributions/` (box and ECDF per
  group), `rate_vs_loss/` and `final_score_vs_loss_*` (score vs loss), `meta_selection.png`,
  `gradcam/selected/` (Grad-CAM of the selected rater and inner heads, one image per group and split)
- `plots/checkpoints/`: `best_meta.pt`, `latest_meta.pt` (rater, inner heads, optimizer, RNG), optional `meta_<step>.pt`
- Across runs: `python algorithms/rater.py --runs RUN... --output DIR` (mean ± std over seeds),
  `--compare calibration_size` (sensitivity to D_meta size).

**Stage 2 (final classifier)**, in the experiment folder:
- per seed (`seed_<s>/`): `plots/data/metrics.csv`, `group_metrics.csv`, `class_metrics.csv`,
  `history.csv`, `weight_stats.csv`; curves over epochs (`overall_`, `group_metrics_*`, `class_metrics_*`);
  weight histograms by group per epoch; `selection_results.csv` and `oracle_results.csv`;
  `selected_models/<rule>.pt` and `checkpoints/` (`by_selection/`, oracle, `latest.pt`); Grad-CAM of the
  selected classifiers in `plots/gradcam/<rule>/` (cell 11, `ENABLE_GRADCAM`); `summary.json`/`summary.txt` (incl. elapsed time)
- aggregate: `selection_results_all_seeds.csv`, `selection_metrics_mean_std.csv`, oracle files,
  `trajectory_metrics_mean_std.csv`, `aggregate_plots/` (curves and selection rules, mean ± std),
  `summary.txt`; `analysis/plot_final_classifier_multiseed.py` re-plots completed runs.

## Baselines and analysis

`analysis/run_baselines.py` runs AFR, Evidential Alignment, DFR (group-labelled upper bound) and optionally
JTT on the same embeddings and splits; `analysis/main_table.py` and `analysis/compare.py` build the main
table and paired comparisons. See `analysis/README.md`.

## Acknowledgements

The data pipeline, backbones and baseline implementations come from the Evidential Alignment
repository (Ye, Zheng and Zhang, KDD 2025). The meta-learning principle follows DataRater
(Calian et al., 2025, arXiv:2505.17895).

```bibtex
@inproceedings{ye2025improving,
  title={Improving Group Robustness on Spurious Correlation via Evidential Alignment},
  author={Ye, Wenqian and Zheng, Guangtao and Zhang, Aidong},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={3610--3621},
  year={2025}
}
```
