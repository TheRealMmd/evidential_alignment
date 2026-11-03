# Drive layout

The shared Google Drive folder holds data and experiment outputs only; code and notebooks live in this repository.
Owner: Mohammad (`colab.mohammadian@gmail.com`). Name: `FeatureRater` (was `evidential_alignment_colab`; the notebooks accept both names).

## Current structure
```
FeatureRater/
  archives/                         dataset tarballs
  code/                             (one legacy file; code belongs in the repo)
  waterbirds/
    erm-r50-e100-s0/                ERM backbone 1055b1e8de29 (sheet: Backbone 1)
      waterbirds_erm_resnet50_imagenet_100ep_best_val_acc.pt   + manifest, results, EA feature files (*.npy)
      erm_waterbirds_resnet50_train_128B_100E_seed0_pretrained/   the training run (other checkpoints too)
      embeddings/waterbirds/resnet50_waterbirds_erm_1055b1e8de29/resolution_224/
      raters/<rater run>/           final_rater.pt, logs, plots
        classifiers/<run>/          every final classifier trained with this rater
      controls/<run>/               no-rater classifiers (uniform, random, minority weights, fresh linear)
      baselines/<method>/           AFR, EA, JTT, DFR …
      analysis/<kind>/              t-SNE and other figures
    erm-r50-e100-s0-augmented/      augmented ERM (not used yet)
    imagenet-r50/embeddings/        ImageNet-only features
  unsorted/                         runs whose ERM or rater is not recorded (GERNE, BAM, early 4–5 Oct runs)
```

## Where a new run goes
| Run | Folder | Set by |
|---|---|---|
| ERM backbone | `waterbirds/<ERM_ID>/` | `ERM_ID` in the ERM notebook |
| Rater | `waterbirds/<ERM_ID>/raters/<objective>_<time>/` | Train Rater notebook (`ERM_ID`) |
| Classifier with a rater | `…/raters/<rater run>/classifiers/<run>/` | multi-seed notebook (`RATER_RUN`) |
| Control (no rater) | `waterbirds/<ERM_ID>/controls/<run>/` | random / minority notebooks |
| Baseline | `waterbirds/<ERM_ID>/baselines/<method>/` | `analysis/run_baselines.py --out-dir` |

`ERM_ID` names a backbone: `<arch>-e<epochs>-s<seed>[-variant]`. A new backbone gets a new folder; nothing else changes.
Each notebook resolves paths in its first cells and falls back to the pre-November locations.

## Old → new (reorganization of 3 Nov 2026)
Only moves: no file was deleted, copied or edited. All 15,894 files and 13.07 GB were verified unchanged afterwards. The old top-level folders were left empty.

| Old top-level folder | Now |
|---|---|
| `GERNE_checkpoints/` (1 item) | `unsorted/` |
| `bam_checkpoints/` (1 item) | `unsorted/` |
| `embeddings/` (1 item) | ImageNet-only cache → `waterbirds/imagenet-r50/embeddings/` |
| `embeddings_waterbirds_erm/` (1 item) | `waterbirds/erm-r50-e100-s0/embeddings/waterbirds/…` |
| `erm_base_models/` (24 items) | plain ERM → `waterbirds/erm-r50-e100-s0/` (checkpoint, manifest, results, EA feature files, training run); augmented ERM → `waterbirds/erm-r50-e100-s0-augmented/` |
| `erm_fresh_classifier_baselines/` (1 item) | `waterbirds/erm-r50-e100-s0/controls/` |
| `evidential_alignment_runs/` (2 items) | `waterbirds/erm-r50-e100-s0/baselines/evidential_alignment/` |
| `feature_rater_tsne/` (7 items) | `waterbirds/erm-r50-e100-s0/analysis/tsne/` |
| `fixed_minority_group_weight_final_multiseed_multiselection/` (3 items) | `waterbirds/erm-r50-e100-s0/controls/` |
| `fixed_random_weight_final_multiseed_multiselection/` (1 item) | `waterbirds/erm-r50-e100-s0/controls/` |
| `frozen_new_rater_final_runs/` (1 item) | `unsorted/` |
| `frozen_rater_final_multiseed_multiselection/` (3 items) | multi-seed classifier runs → `raters/<the rater it used>/classifiers/` |
| `frozen_rater_final_runs_configurable/` (31 items) | classifier runs → `raters/<the rater it used>/classifiers/` (2 empty runs → `unsorted/`) |
| `rater_erm_minimal_runs/` (1 item) | `unsorted/` |
| `rater_erm_runs/` (1 item) | `unsorted/` |
| `waterbirds_raters/` (13 items) | each rater run → `waterbirds/erm-r50-e100-s0/raters/<run>/` |

### Every moved item
| Old path | New path |
|---|---|
| `GERNE_checkpoints` | `unsorted/GERNE_checkpoints` |
| `bam_checkpoints` | `unsorted/bam_checkpoints` |
| `embeddings/waterbirds/resnet50_imagenet_pretrained` | `waterbirds/imagenet-r50/embeddings/resnet50_imagenet_pretrained` |
| `embeddings_waterbirds_erm/waterbirds` | `waterbirds/erm-r50-e100-s0/embeddings/waterbirds` |
| `erm_base_models/augmented_checkpoint_backups/backup_20261103_130907` | `waterbirds/erm-r50-e100-s0-augmented/checkpoint_backups/backup_20261103_130907` |
| `erm_base_models/augmented_runs/erm_waterbirds_resnet50_train_128B_100E_seed0_pretrained` | `waterbirds/erm-r50-e100-s0-augmented/erm_waterbirds_resnet50_train_128B_100E_seed0_pretrained` |
| `erm_base_models/embeds_test.npy` | `waterbirds/erm-r50-e100-s0/embeds_test.npy` |
| `erm_base_models/embeds_train_no_aug.npy` | `waterbirds/erm-r50-e100-s0/embeds_train_no_aug.npy` |
| `erm_base_models/embeds_val.npy` | `waterbirds/erm-r50-e100-s0/embeds_val.npy` |
| `erm_base_models/erm_waterbirds_resnet50_train_128B_100E_seed0_pretrained` | `waterbirds/erm-r50-e100-s0/erm_waterbirds_resnet50_train_128B_100E_seed0_pretrained` |
| `erm_base_models/groups_test.npy` | `waterbirds/erm-r50-e100-s0/groups_test.npy` |
| `erm_base_models/groups_train_no_aug.npy` | `waterbirds/erm-r50-e100-s0/groups_train_no_aug.npy` |
| `erm_base_models/groups_val.npy` | `waterbirds/erm-r50-e100-s0/groups_val.npy` |
| `erm_base_models/info_test.txt` | `waterbirds/erm-r50-e100-s0/info_test.txt` |
| `erm_base_models/info_train_no_aug.txt` | `waterbirds/erm-r50-e100-s0/info_train_no_aug.txt` |
| `erm_base_models/info_val.txt` | `waterbirds/erm-r50-e100-s0/info_val.txt` |
| `erm_base_models/labels_test.npy` | `waterbirds/erm-r50-e100-s0/labels_test.npy` |
| `erm_base_models/labels_train_no_aug.npy` | `waterbirds/erm-r50-e100-s0/labels_train_no_aug.npy` |
| `erm_base_models/labels_val.npy` | `waterbirds/erm-r50-e100-s0/labels_val.npy` |
| `erm_base_models/pred_probs_test.npy` | `waterbirds/erm-r50-e100-s0/pred_probs_test.npy` |
| `erm_base_models/pred_probs_train_no_aug.npy` | `waterbirds/erm-r50-e100-s0/pred_probs_train_no_aug.npy` |
| `erm_base_models/pred_probs_val.npy` | `waterbirds/erm-r50-e100-s0/pred_probs_val.npy` |
| `erm_base_models/waterbirds_erm_augmented_training_results.txt` | `waterbirds/erm-r50-e100-s0-augmented/waterbirds_erm_augmented_training_results.txt` |
| `erm_base_models/waterbirds_erm_resnet50_imagenet_100ep_augmented_best_val_acc.pt` | `waterbirds/erm-r50-e100-s0-augmented/waterbirds_erm_resnet50_imagenet_100ep_augmented_best_val_acc.pt` |
| `erm_base_models/waterbirds_erm_resnet50_imagenet_100ep_augmented_manifest.json` | `waterbirds/erm-r50-e100-s0-augmented/waterbirds_erm_resnet50_imagenet_100ep_augmented_manifest.json` |
| `erm_base_models/waterbirds_erm_resnet50_imagenet_100ep_best_val_acc.pt` | `waterbirds/erm-r50-e100-s0/waterbirds_erm_resnet50_imagenet_100ep_best_val_acc.pt` |
| `erm_base_models/waterbirds_erm_resnet50_imagenet_100ep_manifest.json` | `waterbirds/erm-r50-e100-s0/waterbirds_erm_resnet50_imagenet_100ep_manifest.json` |
| `erm_base_models/waterbirds_erm_training_results.txt` | `waterbirds/erm-r50-e100-s0/waterbirds_erm_training_results.txt` |
| `erm_fresh_classifier_baselines/fresh_linear_erm_splitval_0.6_lr_0.01_20261007_114134` | `waterbirds/erm-r50-e100-s0/controls/fresh_linear_erm_splitval_0.6_lr_0.01_20261007_114134` |
| `evidential_alignment_runs/original_ea_saved_erm_weight_audit_20261002_183015` | `waterbirds/erm-r50-e100-s0/baselines/evidential_alignment/original_ea_saved_erm_weight_audit_20261002_183015` |
| `evidential_alignment_runs/original_ea_saved_erm_weight_audit_20261002_184209` | `waterbirds/erm-r50-e100-s0/baselines/evidential_alignment/original_ea_saved_erm_weight_audit_20261002_184209` |
| `feature_rater_tsne/test_rater_weight_20261028_071113_792504` | `waterbirds/erm-r50-e100-s0/analysis/tsne/test_rater_weight_20261028_071113_792504` |
| `feature_rater_tsne/test_rater_weight_20261028_073608_896972` | `waterbirds/erm-r50-e100-s0/analysis/tsne/test_rater_weight_20261028_073608_896972` |
| `feature_rater_tsne/test_rater_weight_20261028_074250_867245` | `waterbirds/erm-r50-e100-s0/analysis/tsne/test_rater_weight_20261028_074250_867245` |
| `feature_rater_tsne/train_no_aug_rater_weight_20261028_072214_002827` | `waterbirds/erm-r50-e100-s0/analysis/tsne/train_no_aug_rater_weight_20261028_072214_002827` |
| `feature_rater_tsne/val_rater_weight_20261028_071930_986678` | `waterbirds/erm-r50-e100-s0/analysis/tsne/val_rater_weight_20261028_071930_986678` |
| `feature_rater_tsne/val_subset1_rater_weight_20261028_071625_248186` | `waterbirds/erm-r50-e100-s0/analysis/tsne/val_subset1_rater_weight_20261028_071625_248186` |
| `feature_rater_tsne/val_subset2_rater_weight_20261028_071800_510406` | `waterbirds/erm-r50-e100-s0/analysis/tsne/val_subset2_rater_weight_20261028_071800_510406` |
| `fixed_minority_group_weight_final_multiseed_multiselection/majority_0.33_minority_1.0_lr_0.01_splitval_0.6_20261028_075627` | `waterbirds/erm-r50-e100-s0/controls/majority_0.33_minority_1.0_lr_0.01_splitval_0.6_20261028_075627` |
| `fixed_minority_group_weight_final_multiseed_multiselection/majority_0.5_minority_1.0_lr_0.01_splitval_0.6_20261013_132744` | `waterbirds/erm-r50-e100-s0/controls/majority_0.5_minority_1.0_lr_0.01_splitval_0.6_20261013_132744` |
| `fixed_minority_group_weight_final_multiseed_multiselection/majority_0.66_minority_1.0_lr_0.01_splitval_0.6_20261028_114944` | `waterbirds/erm-r50-e100-s0/controls/majority_0.66_minority_1.0_lr_0.01_splitval_0.6_20261028_114944` |
| `fixed_random_weight_final_multiseed_multiselection/random_uniform_0.0_1.0_lr_0.01_splitval_0.6_20261013_140627` | `waterbirds/erm-r50-e100-s0/controls/random_uniform_0.0_1.0_lr_0.01_splitval_0.6_20261013_140627` |
| `frozen_new_rater_final_runs` | `unsorted/frozen_new_rater_final_runs` |
| `frozen_rater_final_multiseed_multiselection/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_095011` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_095011` |
| `frozen_rater_final_multiseed_multiselection/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_103458` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_103458` |
| `frozen_rater_final_multiseed_multiselection/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_131900` | `waterbirds/erm-r50-e100-s0/raters/top10pct_per_class_all_sample_20261011_131220/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_131900` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_095201` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_095201` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_100100` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_100100` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261007_114555` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261007_114555` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233807` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233807` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_234001` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_234001` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113550` | `unsorted/frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113550` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113633` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113633` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_114051` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_114051` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_114447` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_114447` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_115209` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_115209` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_115551` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_115551` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_130531` | `waterbirds/erm-r50-e100-s0/raters/logsumexp_beta_1p0_all_sample_20261011_125739/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_130531` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_130835` | `waterbirds/erm-r50-e100-s0/raters/logsumexp_beta_1p0_all_sample_20261011_125739/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_130835` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_131119` | `waterbirds/erm-r50-e100-s0/raters/logsumexp_beta_1p0_all_sample_20261011_125739/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_131119` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_140118` | `waterbirds/erm-r50-e100-s0/raters/top10pct_per_class_all_sample_20261011_131220/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_140118` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_140457` | `waterbirds/erm-r50-e100-s0/raters/top10pct_per_class_all_sample_20261011_131220/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_140457` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_141149` | `waterbirds/erm-r50-e100-s0/raters/top10pct_per_class_all_sample_20261011_131220/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_141149` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261012_140654` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261012_140654` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_081613` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_081613` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_082036` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_082036` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_082346` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261013_082346` |
| `frozen_rater_final_runs_configurable/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.8_20261005_095444` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_all_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.8_20261005_095444` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_batch_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094953` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_misclassified_norm_batch_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094953` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094432` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094432` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094737` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261005_094737` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_232707` | `unsorted/frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_232707` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_232710` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_232710` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233054` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233054` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233614` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_233614` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_234217` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261009_234217` |
| `frozen_rater_final_runs_configurable/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113305` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941/classifiers/rating_misclassified_norm_global_sigmoid_temp_1.0_lr_0.01_splitval_0.6_20261011_113305` |
| `rater_erm_minimal_runs` | `unsorted/rater_erm_minimal_runs` |
| `rater_erm_runs` | `unsorted/rater_erm_runs` |
| `waterbirds_raters/class_balanced_runs/class_balanced_all_sample_20261012_113019` | `waterbirds/erm-r50-e100-s0/raters/class_balanced_all_sample_20261012_113019` |
| `waterbirds_raters/group_balanced_loss_runs/group_balanced_loss_all_sample_20261013_141048` | `waterbirds/erm-r50-e100-s0/raters/group_balanced_loss_all_sample_20261013_141048` |
| `waterbirds_raters/logsumexp_runs/logsumexp_beta_1p0_all_sample_20261011_125739` | `waterbirds/erm-r50-e100-s0/raters/logsumexp_beta_1p0_all_sample_20261011_125739` |
| `waterbirds_raters/rater_classwise_top_loss_runs/top100pct_per_class_all_sample_20261012_152903` | `waterbirds/erm-r50-e100-s0/raters/top100pct_per_class_all_sample_20261012_152903` |
| `waterbirds_raters/rater_classwise_top_loss_runs/top10pct_per_class_all_sample_20261011_131220` | `waterbirds/erm-r50-e100-s0/raters/top10pct_per_class_all_sample_20261011_131220` |
| `waterbirds_raters/rater_classwise_top_loss_runs/top30pct_per_class_all_sample_20261007_215941` | `waterbirds/erm-r50-e100-s0/raters/top30pct_per_class_all_sample_20261007_215941` |
| `waterbirds_raters/rater_classwise_top_loss_runs/top50pct_per_class_all_sample_20261012_101749` | `waterbirds/erm-r50-e100-s0/raters/top50pct_per_class_all_sample_20261012_101749` |
| `waterbirds_raters/rater_classwise_top_loss_runs/top70pct_per_class_all_sample_20261012_103006` | `waterbirds/erm-r50-e100-s0/raters/top70pct_per_class_all_sample_20261012_103006` |
| `waterbirds_raters/rater_meta_only_runs/ea_gated_full_epoch_on_all_20261005_081758` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_on_all_20261005_081758` |
| `waterbirds_raters/rater_meta_only_runs/ea_gated_full_epoch_random_inner_20261004_142804` | `waterbirds/erm-r50-e100-s0/raters/ea_gated_full_epoch_random_inner_20261004_142804` |
| `waterbirds_raters/standard_runs/standard_ce_all_sample_20261012_233626` | `waterbirds/erm-r50-e100-s0/raters/standard_ce_all_sample_20261012_233626` |
| `waterbirds_raters/worst_class_loss_runs/worst_class_loss_all_sample_20261012_111028` | `waterbirds/erm-r50-e100-s0/raters/worst_class_loss_all_sample_20261012_111028` |
| `waterbirds_raters/worst_group_runs/worst_group_loss_all_sample_20261012_170942` | `waterbirds/erm-r50-e100-s0/raters/worst_group_loss_all_sample_20261012_170942` |

A file-by-file map (old path, new path, Drive id) and a script that moves everything back are kept with the reorganization record in Project Station.
