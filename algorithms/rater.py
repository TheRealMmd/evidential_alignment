import os
import csv
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .algorithm import Algorithm
from .register import register_algorithm

from models.classifier import Classifier
from utils import log


# ============================================================
# Embedding dataset
# ============================================================

class EmbeddingTensorDataset(Dataset):
    """Dataset of frozen backbone embeddings and Waterbirds metadata."""

    def __init__(self, embeddings, labels, groups, attrs):
        self.embeddings = embeddings.float()
        self.y_array = labels.long()
        self.group_array = groups.long()
        self.confounder_array = attrs.long()
        self.n_classes = int(torch.unique(self.y_array).numel())

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.y_array[idx],
            self.group_array[idx],
            self.confounder_array[idx],
        )


# ============================================================
# Rater architectures
# ============================================================

class EmbeddingRaterSmall(nn.Module):
    """Small MLP rater for a ResNet-50 pooled embedding."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class EmbeddingRaterMedium(nn.Module):
    """Medium-capacity MLP rater for a ResNet-50 pooled embedding."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ============================================================
# Inner / downstream classifier
# ============================================================

class InnerLinearClassifier(nn.Module):
    """Linear classifier operating on frozen embeddings."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, z):
        return self.linear(z)


# ============================================================
# Rater algorithm
# ============================================================

@register_algorithm("rater")
class Rater(Algorithm):

    def _run_log(self, message):
        """
        Mirror messages to both the repository logger and experiment/log.txt.
        Before train() knows the output directory, this only logs normally.
        """
        log(message)

        path = getattr(
            self,
            "run_log_path",
            None,
        )

        if path:
            with open(
                path,
                "a",
                encoding="utf-8",
            ) as fout:
                fout.write(
                    str(message)
                    + "\n"
                )

    """
    Frozen trained feature-Rater used only for final-classifier training.

    Image x -> frozen Waterbirds-ERM ResNet-50 -> z -> r_eta(z) -> scalar score.

    Inner update:

        First, the CURRENT inner classifier predicts each sample.

        If the prediction is CORRECT:
            w_i = 1

        If the prediction is WRONG:
            the Rater is evaluated ONLY for that misclassified sample,
            and its raw score is converted to a rating using the selected
            Rater transform:

            SOFTMAX:
                r_i = softmax((score_i - mean_misclassified) / tau)

            SIGMOID:
                standardized_i =
                    (score_i - mean_misclassified)
                    / (std_misclassified + eps)
                r_i = sigmoid(standardized_i / tau)

            Then:
                w_i = r_i

        Thus the effective weight follows the Evidential-Alignment-style gate:

            w_i = 1                         if prediction is correct
            w_i = RaterRating(z_i)          if prediction is wrong

        The Rater is NOT used to determine the training weight of correctly
        classified samples.

        The requested weighted objective is retained:

            L_inner = sum_i w_i CE(h_theta(z_i), y_i)

        There is intentionally NO division by sum_i w_i.

        theta' = theta - alpha * grad_theta L_inner

    Outer update:
        L_outer = CE(h_theta'(z_val), y_val)
        eta <- eta - beta * grad_eta L_outer

    In addition to learning the rater, this implementation:
      * evaluates the persistent inner-model population during meta-training;
      * logs average accuracy and worst-group accuracy (WGA);
      * measures correlation between raw/final Rater training weights and loss;
      * saves per-group histograms for raw Rater scores and final training weights;
      * saves raw-score-vs-loss and final-weight-vs-loss scatter plots;
      * optionally saves intermediate rater/inner-model checkpoints;
      * after meta-training, restores and freezes the BEST trained Rater;
      * uses an EA-style calibration protocol for the final classifier:
            val_subset1 -> calibration/training,
            val_subset2 -> model selection,
            test        -> diagnostic evaluation after every final epoch;
      * test metrics are NEVER used for checkpoint selection;
      * trains a fresh final classifier with EA-style dynamic weighting:
            correct -> 1,
            misclassified -> trained-Rater rating;
      * saves the ACTUAL final-classifier weight histograms every epoch;
      * selects the downstream classifier by validation WGA by default;
      * evaluates the final classifier on validation/test splits;
      * uses a persistent shared embedding cache so the frozen ERM ResNet-50
        embeddings are extracted once and reused across timestamped runs.
    """

    def __init__(self, config):
        # The method works on frozen backbone features / last-layer models.
        config.last_layer = True
        super().__init__(config)

        if torch.cuda.is_available():
            self.device = f"cuda:{self.config.gpu}"
        else:
            self.device = "cpu"

        self.n_classes = self.datasets["train"].n_classes

        if self.config.backbone != "resnet50":
            self._run_log(
                f"[Rater] Warning: requested backbone is "
                f"{self.config.backbone}, not resnet50."
            )

        if not self.config.pretrained:
            raise ValueError(
                "This Rater experiment expects a ResNet-50 that was "
                "initialized from ImageNet and then ERM-finetuned on "
                "Waterbirds. Run with --pretrained True."
            )

        # ----------------------------------------------------
        # Frozen Waterbirds-ERM feature extractor
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT use a fresh ImageNet-only ResNet-50 here.
        #
        # We reconstruct the same Classifier architecture used by the
        # repository, then load the saved Waterbirds ERM checkpoint:
        #
        #   image
        #       -> ERM-finetuned ResNet-50 backbone
        #       -> 2048-D embedding z
        #       -> saved ERM linear classifier
        #
        # The backbone is frozen after loading. The saved ERM classifier
        # parameters are also copied and used to initialize the persistent
        # inner classifiers for Rater meta-learning.
        # ----------------------------------------------------
        self.erm_model_path = str(
            getattr(
                self.config,
                "erm_model",
                "",
            )
            or os.environ.get(
                "RATER_ERM_MODEL",
                "",
            )
        ).strip()

        if not self.erm_model_path:
            raise ValueError(
                "This Rater experiment requires the saved Waterbirds ERM "
                "checkpoint. Pass it with --erm_model PATH or set "
                "RATER_ERM_MODEL."
            )

        if not os.path.exists(self.erm_model_path):
            raise FileNotFoundError(
                f"ERM checkpoint does not exist: {self.erm_model_path}"
            )

        self.feature_model = Classifier(
            backbone=self.config.backbone,
            num_classes=self.n_classes,
            pretrained=True,
        ).to(self.device)

        erm_checkpoint = torch.load(
            self.erm_model_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(erm_checkpoint, dict)
            and "model_sd" in erm_checkpoint
        ):
            erm_state_dict = erm_checkpoint["model_sd"]
        else:
            erm_state_dict = erm_checkpoint

        filtered_erm_state_dict = {
            k: v
            for k, v in erm_state_dict.items()
            if not k.endswith("position_ids")
        }

        incompatible = self.feature_model.load_state_dict(
            filtered_erm_state_dict,
            strict=False,
        )

        self.erm_checkpoint_epoch = (
            erm_checkpoint.get("epoch", None)
            if isinstance(erm_checkpoint, dict)
            else None
        )

        self.erm_checkpoint_sel_metric = (
            erm_checkpoint.get("sel_metric", None)
            if isinstance(erm_checkpoint, dict)
            else None
        )

        if not isinstance(
            self.feature_model.fc,
            nn.Linear,
        ):
            raise TypeError(
                "Expected the saved ERM Classifier to have a linear .fc "
                "head, but got "
                f"{type(self.feature_model.fc).__name__}."
            )

        self.feature_dim = (
            self.feature_model.backbone.num_features
        )

        if (
            self.feature_model.fc.weight.shape[1]
            != self.feature_dim
        ):
            raise ValueError(
                "ERM classifier input dimension does not match the "
                "backbone embedding dimension."
            )

        # Exact ERM classifier parameters used as the base initialization
        # for the persistent inner classifiers.
        self.erm_head_weight = (
            self.feature_model.fc.weight
            .detach()
            .clone()
        )

        self.erm_head_bias = (
            self.feature_model.fc.bias
            .detach()
            .clone()
        )

        # Fingerprint the exact checkpoint file so this cache can never be
        # confused with the old ImageNet-only embedding cache.
        erm_stat = os.stat(
            self.erm_model_path
        )

        fingerprint_text = (
            f"{os.path.abspath(self.erm_model_path)}|"
            f"{erm_stat.st_size}|"
            f"{erm_stat.st_mtime_ns}"
        )

        self.erm_checkpoint_fingerprint = (
            hashlib.sha1(
                fingerprint_text.encode("utf-8")
            ).hexdigest()[:12]
        )

        self.feature_model.eval()

        for p in self.feature_model.parameters():
            p.requires_grad_(False)

        self._run_log(
            f"[Rater] Loaded frozen Waterbirds ERM model from: "
            f"{self.erm_model_path}"
        )

        self._run_log(
            f"[Rater] ERM checkpoint epoch="
            f"{self.erm_checkpoint_epoch}, "
            f"sel_metric={self.erm_checkpoint_sel_metric}, "
            f"fingerprint={self.erm_checkpoint_fingerprint}"
        )

        if incompatible.missing_keys:
            self._run_log(
                f"[Rater] ERM load missing keys: "
                f"{incompatible.missing_keys}"
            )

        if incompatible.unexpected_keys:
            self._run_log(
                f"[Rater] ERM load unexpected keys: "
                f"{incompatible.unexpected_keys}"
            )

        self._run_log(
            f"[Rater] Frozen ERM-finetuned "
            f"{self.config.backbone} backbone. "
            f"Embedding dimension = {self.feature_dim}"
        )

        # ----------------------------------------------------
        # Persistent SHARED embedding cache
        # ----------------------------------------------------
        #
        # Preferred Colab usage:
        #
        #   RATER_EMBEDDING_CACHE_ROOT=
        #       /content/drive/MyDrive/evidential_alignment_colab/embeddings
        #
        # The Rater automatically creates a representation-specific
        # namespace under that root. The cache is independent of the
        # timestamped experiment directory, so future runs reuse the same
        # train/val/test embeddings and do not run ResNet-50 again.
        # ----------------------------------------------------
        self.embedding_cache_root = str(
            getattr(
                self.config,
                "rater_embedding_cache_root",
                "",
            )
            or os.environ.get(
                "RATER_EMBEDDING_CACHE_ROOT",
                "",
            )
        ).strip()

        # One-time notebook preparation mode. When enabled, train() only
        # builds/validates the persistent embeddings and then returns.
        self.precompute_embeddings_only = (
            os.environ.get(
                "RATER_PRECOMPUTE_EMBEDDINGS_ONLY",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "y"}
        )

        # ----------------------------------------------------
        # Bilevel hyperparameters
        # ----------------------------------------------------
        self.meta_steps = int(
            getattr(self.config, "rater_meta_steps", self.config.epoch)
        )
        self.inner_steps = int(
            getattr(self.config, "rater_inner_steps", 2)
        )
        self.num_inner_models = int(
            os.environ.get(
                "RATER_NUM_INNER_MODELS",
                str(
                    getattr(
                        self.config,
                        "rater_num_inner_models",
                        2,
                    )
                ),
            )
        )
        self.inner_lr = float(
            getattr(self.config, "rater_inner_lr", 1e-2)
        )
        self.outer_lr = float(
            getattr(self.config, "rater_outer_lr", 3e-4)
        )
        self.temperature = float(
            os.environ.get(
                "RATER_TEMPERATURE",
                str(
                    getattr(
                        self.config,
                        "rater_temperature",
                        2.0,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Rater-rating transform for MISCLASSIFIED samples only.
        #
        # Correctly classified samples always receive weight = 1.
        # Misclassified samples receive a Rater-derived rating using the
        # selected softmax/sigmoid transform.
        #
        # Effective training rule:
        #       weight_i = 1                  if correct
        #       weight_i = rater_rating_i     if misclassified
        #
        # Inner loss remains:
        #       sum_i weight_i * CE_i
        #
        # There is NO division by sum(weights).
        # ----------------------------------------------------
        self.weighting = str(
            os.environ.get(
                "RATER_WEIGHTING",
                str(
                    getattr(
                        self.config,
                        "rater_weighting",
                        "softmax",
                    )
                ),
            )
        ).lower()

        self.refresh_steps = int(
            getattr(self.config, "rater_refresh_steps", 100)
        )
        self.grad_clip = float(
            getattr(self.config, "rater_grad_clip", 5.0)
        )
        self.inner_reg_weight = float(
            getattr(self.config, "rater_inner_reg_weight", 0.0)
        )
        self.outer_reg_weight = 0.0
        self.score_reg_weight = 0.0

        # ----------------------------------------------------
        # First ERM-initialized Rater experiment
        # ----------------------------------------------------
        #
        # Rater meta-objective:
        #
        #       L_meta = CE(
        #           h_theta'(z_outer),
        #           y_outer
        #       )
        #
        # There is NO gradient-magnitude auxiliary loss and no additional
        # Rater regularizer in the meta objective for this experiment.
        #
        # The inner classifiers start from the SAVED ERM classifier:
        #
        #   inner model 0:
        #       exact ERM classifier weights
        #
        #   inner models 1,2,...:
        #       ERM classifier + very small Gaussian perturbation
        #
        # This keeps the two-model population near the same useful ERM
        # solution without making their meta-trajectories exactly identical.
        # With RATER_NUM_INNER_MODELS=1, the sole inner model is the exact
        # saved ERM classifier.
        # ----------------------------------------------------
        self.grad_loss_weight = 0.0

        self.inner_init_noise_std = float(
            os.environ.get(
                "RATER_INNER_INIT_NOISE_STD",
                str(
                    getattr(
                        self.config,
                        "rater_inner_init_noise_std",
                        1e-3,
                    )
                ),
            )
        )

        if self.inner_init_noise_std < 0:
            raise ValueError(
                "RATER_INNER_INIT_NOISE_STD must be >= 0."
            )

        # This experiment intentionally keeps the previous ALL-SAMPLE
        # weighting rule unchanged.
        self.rate_all_samples = (
            os.environ.get(
                "RATER_RATE_ALL_SAMPLES",
                "1",
            ).strip().lower()
            in {"1", "true", "yes", "y"}
        )

        if not self.rate_all_samples:
            raise ValueError(
                "This experiment keeps the ALL-SAMPLE Rater weighting "
                "used by the attached implementation. Set "
                "RATER_RATE_ALL_SAMPLES=1."
            )

        self.rater_capacity = getattr(
            self.config, "rater_capacity", "medium"
        )

        # ----------------------------------------------------
        # Diagnostic plotting. By default, plots are saved at
        # the same frequency as --eval_freq. If config.py later
        # defines --rater_plot_freq, it will override eval_freq.
        # ----------------------------------------------------
        self.plot_freq = int(
            getattr(
                self.config,
                "rater_plot_freq",
                max(1, int(getattr(self.config, "eval_freq", 1))),
            )
        )
        # PNG plots only; never save per-plot CSV data.
        self.save_plot_data = False

        # ----------------------------------------------------
        # Final weighted classifier hyperparameters.
        # These have defaults, so config.py does not HAVE to
        # define them unless you want command-line control.
        # ----------------------------------------------------
        self.final_epochs = int(
            os.environ.get(
                "RATER_FINAL_EPOCHS",
                str(
                    getattr(
                        self.config,
                        "rater_final_epochs",
                        50,
                    )
                ),
            )
        )

        self.final_lr = float(
            os.environ.get(
                "RATER_FINAL_LR",
                str(
                    getattr(
                        self.config,
                        "rater_final_lr",
                        1e-3,
                    )
                ),
            )
        )

        self.final_momentum = float(
            os.environ.get(
                "RATER_FINAL_MOMENTUM",
                str(
                    getattr(
                        self.config,
                        "rater_final_momentum",
                        0.9,
                    )
                ),
            )
        )

        self.final_weight_decay = float(
            os.environ.get(
                "RATER_FINAL_WEIGHT_DECAY",
                str(
                    getattr(
                        self.config,
                        "rater_final_weight_decay",
                        1e-4,
                    )
                ),
            )
        )
        # For robustness experiments we select the downstream
        # classifier by held-out validation WGA by default.
        #
        # NOTE: WGA selection uses validation group labels. If you want a
        # group-label-free selection rule, set rater_final_selection to
        # "accuracy" or "loss" in config.py.
        self.final_selection = str(
            getattr(self.config, "rater_final_selection", "wga")
        ).lower()

        # Save final-classifier weight histograms every epoch by default.
        # No config.py change is required.
        self.final_plot_freq = int(
            os.environ.get(
                "RATER_FINAL_PLOT_FREQ",
                str(
                    getattr(
                        self.config,
                        "rater_final_plot_freq",
                        5,
                    )
                ),
            )
        )

        if self.inner_steps < 1:
            raise ValueError("rater_inner_steps must be >= 1.")
        if self.num_inner_models < 1:
            raise ValueError("rater_num_inner_models must be >= 1.")
        if self.temperature <= 0:
            raise ValueError("rater_temperature must be > 0.")

        if self.weighting not in {"softmax", "sigmoid"}:
            raise ValueError(
                "rater_weighting must be either 'softmax' or 'sigmoid'."
            )

        if self.final_selection not in {"accuracy", "wga", "loss"}:
            raise ValueError(
                "rater_final_selection must be one of: accuracy, wga, loss"
            )

        if self.final_plot_freq < 1:
            raise ValueError("rater_final_plot_freq must be >= 1.")

        # ----------------------------------------------------
        # EA-style calibration protocol for the FINAL classifier
        # ----------------------------------------------------
        #
        # With --split_val < 1, prepare_data() creates:
        #
        #   val_subset1 -> calibration set
        #   val_subset2 -> held-out selection set
        #
        # This updated Rater uses:
        #
        #   * original train_no_aug for inner-model training
        #   * val_subset1 as the Rater outer/calibration signal
        #   * val_subset1 to train the fresh final classifier
        #   * val_subset2 only for final-classifier checkpoint selection
        #   * test only for final reporting
        #
        # This keeps val_subset2 untouched during Rater meta-training.
        # ----------------------------------------------------
        self.use_calibration_final = bool(
            getattr(
                self.config,
                "rater_use_calibration_final",
                True,
            )
        )

        # ----------------------------------------------------
        # Rater
        # ----------------------------------------------------
        if self.rater_capacity == "small":
            self.rater = EmbeddingRaterSmall(self.feature_dim).to(self.device)
        elif self.rater_capacity == "medium":
            self.rater = EmbeddingRaterMedium(self.feature_dim).to(self.device)
        else:
            raise ValueError(
                f"Unknown rater capacity: {self.rater_capacity}"
            )

        self.outer_optimizer = torch.optim.Adam(
            self.rater.parameters(), lr=self.outer_lr
        )

        self.inner_models = []
        self.final_classifier = None

        self._run_log(
            f"[Rater] ALL-SAMPLE Rater transform = "
            f"{self.weighting}; temperature = {self.temperature}"
        )
        self._run_log(
            "[Rater] ALL samples are rated and weighted by the Rater; "
            "there is NO correct/misclassified gate."
        )
        self._run_log(
            "[Rater] Inner objective: L_inner = sum_i w_i * CE_i "
            "WITHOUT dividing by sum(w)."
        )
        self._run_log(
            "[Rater] Meta objective = OUTER CLASSIFICATION "
            "CROSS-ENTROPY ONLY."
        )
        self._run_log(
            f"[Rater] Inner population = {self.num_inner_models}; "
            f"model 0 starts from the EXACT ERM classifier; "
            f"models 1+ use ERM + Gaussian noise "
            f"(std={self.inner_init_noise_std:g})."
        )

        if self.config.check_point:
            self._load_rater_checkpoint(self.config.check_point)

    # ========================================================
    # Persistent shared embedding cache
    # ========================================================

    @staticmethod
    def _safe_cache_token(value):
        """Convert a value into a filesystem-safe cache token."""
        token = str(value)
        token = token.replace("/", "_")
        token = token.replace(chr(92), "_")
        token = token.replace(" ", "_")
        token = token.replace(".", "p")
        return token

    def _embedding_cache_namespace(self):
        """
        Namespace shared embeddings by the EXACT frozen ERM representation.

        This intentionally differs from the old ImageNet-only cache.
        """
        dataset_name = self._safe_cache_token(
            getattr(
                self.config,
                "dataset",
                "dataset",
            )
        )

        backbone_name = self._safe_cache_token(
            self.config.backbone
        )

        resolution = self._safe_cache_token(
            getattr(
                self.config,
                "resolution",
                224,
            )
        )

        representation_tag = (
            f"{backbone_name}_waterbirds_erm_"
            f"{self.erm_checkpoint_fingerprint}"
        )

        return os.path.join(
            dataset_name,
            representation_tag,
            f"resolution_{resolution}",
        )

    def _resolve_embedding_cache_dir(self, output_dir=None):
        """
        Return the embedding directory used by every train/test call.

        If RATER_EMBEDDING_CACHE_ROOT is set, timestamped experiments all
        share the same tensors. Otherwise fall back to the old per-run cache.
        """
        if self.embedding_cache_root:
            cache_dir = os.path.join(
                self.embedding_cache_root,
                self._embedding_cache_namespace(),
            )
        else:
            if output_dir is None:
                raise ValueError(
                    "No shared embedding cache root was configured and "
                    "output_dir is None."
                )

            cache_dir = os.path.join(
                output_dir,
                "embedding_cache",
            )

        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _embedding_cache_metadata(self, split):
        """Metadata used to validate persistent cache files."""
        metadata = {
            "dataset": str(
                getattr(self.config, "dataset", "")
            ),
            "split": str(split),
            "backbone": str(self.config.backbone),
            "pretrained": bool(self.config.pretrained),
            "resolution": int(
                getattr(self.config, "resolution", 224)
            ),
            "feature_dim": int(self.feature_dim),
            "n_classes": int(self.n_classes),
            "representation_source": (
                "waterbirds_erm_checkpoint"
            ),
            "erm_checkpoint_path": os.path.abspath(
                self.erm_model_path
            ),
            "erm_checkpoint_fingerprint": (
                self.erm_checkpoint_fingerprint
            ),
            "erm_checkpoint_epoch": (
                self.erm_checkpoint_epoch
            ),
        }

        if "subset" in str(split):
            metadata.update(
                {
                    "split_train": float(
                        getattr(self.config, "split_train", 1.0)
                    ),
                    "split_val": float(
                        getattr(self.config, "split_val", 1.0)
                    ),
                    "seed": int(
                        getattr(self.config, "seed", 0)
                    ),
                }
            )

        return metadata

    def _embedding_cache_filename(self, split):
        """
        Standard deterministic splits become:
            train_no_aug.pt
            val.pt
            test.pt

        Subset splits additionally encode split ratios and seed.
        """
        safe_split = self._safe_cache_token(split)

        if "subset" in str(split):
            split_train = self._safe_cache_token(
                getattr(self.config, "split_train", 1.0)
            )
            split_val = self._safe_cache_token(
                getattr(self.config, "split_val", 1.0)
            )
            seed = self._safe_cache_token(
                getattr(self.config, "seed", 0)
            )

            safe_split = (
                f"{safe_split}"
                f"_splittrain_{split_train}"
                f"_splitval_{split_val}"
                f"_seed_{seed}"
            )

        return f"{safe_split}.pt"

    @staticmethod
    def _cache_payload_has_required_tensors(payload):
        return (
            isinstance(payload, dict)
            and "embeddings" in payload
            and "labels" in payload
            and "groups" in payload
            and "attrs" in payload
        )

    def _cache_metadata_matches(self, payload, split):
        """Reject stale or incompatible shared cache files."""
        if not self._cache_payload_has_required_tensors(payload):
            return False

        embeddings = payload["embeddings"]

        if (
            not torch.is_tensor(embeddings)
            or embeddings.ndim != 2
            or embeddings.shape[1] != self.feature_dim
        ):
            return False

        expected = self._embedding_cache_metadata(split)
        actual = payload.get("cache_metadata", None)

        if actual is None:
            return False

        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                return False

        n = embeddings.shape[0]

        return all(
            torch.is_tensor(payload[key])
            and payload[key].shape[0] == n
            for key in ("labels", "groups", "attrs")
        )

    def _load_embedding_cache_file(self, cache_file, split):
        """
        Load one persistent cache file, returning None if it must be rebuilt.
        """
        if not os.path.exists(cache_file):
            return None

        self._run_log(
            f"[Rater embeddings] Found shared cache: {cache_file}"
        )

        try:
            payload = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            self._run_log(
                f"[Rater embeddings] Cache load failed "
                f"({type(exc).__name__}: {exc}). "
                f"Regenerating split '{split}'."
            )
            return None

        if not self._cache_metadata_matches(payload, split):
            self._run_log(
                f"[Rater embeddings] Cache metadata mismatch for "
                f"split '{split}'. Regenerating it once."
            )
            return None

        self._run_log(
            f"[Rater embeddings] USING stored embeddings for "
            f"'{split}' ({len(payload['labels'])} samples). "
            f"No backbone extraction is performed."
        )

        return EmbeddingTensorDataset(
            payload["embeddings"],
            payload["labels"],
            payload["groups"],
            payload["attrs"],
        )

    @torch.no_grad()
    def _extract_embedding_dataset(self, split, cache_dir=None):
        """
        Load a split from persistent storage.

        On the first ever request, extract with the frozen backbone once and
        save it. Every later run directly loads the stored tensor file.
        """
        if cache_dir is None:
            cache_dir = self._resolve_embedding_cache_dir()

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir,
            self._embedding_cache_filename(split),
        )

        cached_dataset = self._load_embedding_cache_file(
            cache_file,
            split,
        )

        if cached_dataset is not None:
            return cached_dataset

        if split not in self.dataloaders:
            raise ValueError(
                f"Unknown split '{split}'. Available splits: "
                f"{list(self.dataloaders.keys())}"
            )

        self._run_log(
            f"[Rater embeddings] Shared cache MISS for '{split}'. "
            f"Extracting it ONCE with frozen "
            f"{self.config.backbone}..."
        )

        loader = self.dataloaders[split]
        self.feature_model.eval()

        embeddings = []
        labels = []
        groups = []
        attrs = []

        for x, y, g, a in tqdm(
            loader,
            desc=f"ONE-TIME embedding extraction: {split}",
        ):
            x = x.to(
                self.device,
                non_blocking=True,
            )

            z = self.feature_model.backbone(x)

            embeddings.append(z.detach().cpu())
            labels.append(torch.as_tensor(y).cpu())
            groups.append(torch.as_tensor(g).cpu())
            attrs.append(torch.as_tensor(a).cpu())

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        groups = torch.cat(groups, dim=0)
        attrs = torch.cat(attrs, dim=0)

        payload = {
            "embeddings": embeddings,
            "labels": labels,
            "groups": groups,
            "attrs": attrs,
            "cache_metadata": self._embedding_cache_metadata(split),
        }

        tmp_file = cache_file + f".tmp.{os.getpid()}"
        torch.save(payload, tmp_file)
        os.replace(tmp_file, cache_file)

        self._run_log(
            f"[Rater embeddings] SAVED persistent shared "
            f"'{split}' embeddings ({len(labels)} samples):"
        )
        self._run_log(f"[Rater embeddings]   {cache_file}")
        self._run_log(
            "[Rater embeddings] Future runs will load this file "
            "directly and will NOT extract this split again."
        )

        return EmbeddingTensorDataset(
            embeddings,
            labels,
            groups,
            attrs,
        )

    def _precompute_shared_embeddings(self, output_dir):
        """
        Build/validate all deterministic embeddings without meta-training.

        Current Waterbirds workflow:
            train_no_aug
            val
            test

        Future subset/calibration splits are cached too when available.
        """
        cache_dir = self._resolve_embedding_cache_dir(output_dir)

        preferred_splits = [
            "train_no_aug",
            "val",
            "test",
            "train_subset1",
            "train_subset2",
            "val_subset1",
            "val_subset2",
        ]

        available = [
            split
            for split in preferred_splits
            if split in self.dataloaders
        ]

        self._run_log("[Rater embeddings] PRECOMPUTE-ONLY mode.")
        self._run_log(
            f"[Rater embeddings] Shared cache directory: {cache_dir}"
        )
        self._run_log(
            f"[Rater embeddings] Preparing splits: {available}"
        )

        for split in available:
            dataset = self._extract_embedding_dataset(
                split,
                cache_dir,
            )
            self._run_log(
                f"[Rater embeddings] READY: "
                f"{split} -> {len(dataset)} samples"
            )

        ready_file = os.path.join(cache_dir, "_READY.txt")

        with open(ready_file, "w") as f:
            f.write("Persistent Rater embedding cache is ready.\n")
            for split in available:
                f.write(
                    self._embedding_cache_filename(split)
                    + "\n"
                )

        self._run_log(
            f"[Rater embeddings] Cache preparation complete: "
            f"{ready_file}"
        )

    # ========================================================
    # Inner models
    # ========================================================

    def _new_inner_model(
        self,
        model_index=0,
    ):
        """
        Construct one persistent inner classifier.

        model_index == 0:
            exact saved ERM classifier.

        model_index > 0:
            saved ERM classifier + tiny Gaussian perturbation.

        The perturbation is only to avoid an exactly duplicated
        two-model population. The models still start in the local
        neighborhood of the SAME learned ERM classifier.
        """
        model = InnerLinearClassifier(
            input_dim=self.feature_dim,
            num_classes=self.n_classes,
        ).to(self.device)

        with torch.no_grad():
            model.linear.weight.copy_(
                self.erm_head_weight
            )

            model.linear.bias.copy_(
                self.erm_head_bias
            )

            if (
                model_index > 0
                and self.inner_init_noise_std > 0
            ):
                model.linear.weight.add_(
                    torch.randn_like(
                        model.linear.weight
                    )
                    * self.inner_init_noise_std
                )

                model.linear.bias.add_(
                    torch.randn_like(
                        model.linear.bias
                    )
                    * self.inner_init_noise_std
                )

        return model

    def _new_final_classifier(self):
        """
        Final downstream classifier initialized from the exact saved ERM head.

        The trained Rater is frozen; only this linear classifier is optimized.
        """
        model = InnerLinearClassifier(
            input_dim=self.feature_dim,
            num_classes=self.n_classes,
        ).to(self.device)

        with torch.no_grad():
            model.linear.weight.copy_(
                self.erm_head_weight
            )

            model.linear.bias.copy_(
                self.erm_head_bias
            )

        return model

    def _initialize_inner_population(self):
        self.inner_models = [
            self._new_inner_model(
                model_index=i
            )
            for i in range(
                self.num_inner_models
            )
        ]

        with torch.no_grad():
            for i, model in enumerate(
                self.inner_models
            ):
                weight_delta = (
                    model.linear.weight
                    - self.erm_head_weight
                )

                bias_delta = (
                    model.linear.bias
                    - self.erm_head_bias
                )

                self._run_log(
                    f"[Rater inner init {i}] "
                    f"||dW||={weight_delta.norm().item():.8f}, "
                    f"||db||={bias_delta.norm().item():.8f}"
                )

    @staticmethod
    def _next_batch(iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    # ========================================================
    # Raw Rater score -> rating transformation
    # ========================================================

    def _scores_to_weights(self, raw_scores):
        """
        Convert raw Rater outputs into ratings in (0, 1).

        In the EA-style rule this function is applied to MISCLASSIFIED
        samples only. Correct samples receive weight exactly 1.
        """

        if raw_scores.numel() == 0:
            return raw_scores

        if self.weighting == "softmax":
            centered_scores = raw_scores - raw_scores.mean()
            return torch.softmax(
                centered_scores / self.temperature,
                dim=0,
            )

        score_mean = raw_scores.mean()
        score_std = raw_scores.std(unbiased=False)
        standardized_scores = (
            raw_scores - score_mean
        ) / (
            score_std + 1e-6
        )
        return torch.sigmoid(
            standardized_scores / self.temperature
        )

    def _all_sample_training_weights(
        self,
        z,
        y,
        logits,
        raw_scores_all=None,
    ):
        """
        Rate EVERY sample.

        Correct examples are NOT forced to weight 1.

            raw_i = Rater(z_i)
            w_i   = transform(raw_i)

        `correct_mask` is returned only for diagnostics.
        """
        if raw_scores_all is None:
            raw_scores = self.rater(z)
        else:
            raw_scores = raw_scores_all

        weights = self._scores_to_weights(
            raw_scores
        )

        correct_mask = (
            logits.argmax(dim=1).eq(y)
        )

        return (
            weights,
            correct_mask,
            raw_scores,
        )


    def _ea_style_training_weights(
        self,
        z,
        y,
        logits,
        raw_scores_all=None,
    ):
        """
        Backward-compatible alias.

        In this experiment this NO LONGER performs EA-style
        correct/wrong gating. It delegates to the ALL-SAMPLE rule.
        """
        return self._all_sample_training_weights(
            z=z,
            y=y,
            logits=logits,
            raw_scores_all=raw_scores_all,
        )

    # ========================================================
    # Differentiable inner optimization
    # ========================================================

    @staticmethod
    def _within_class_percentile_targets(
        values,
        labels,
    ):
        """
        Percentile-rank scalar values independently within each class.

        Output:
            target in [0, 1]

        If only one example of a class is present in a minibatch,
        that sample gets the neutral target 0.5.
        """
        with torch.no_grad():
            targets = torch.empty_like(
                values,
                dtype=torch.float32,
            )

            for class_id in torch.unique(
                labels
            ):
                mask = labels.eq(
                    class_id
                )

                class_values = values[
                    mask
                ]

                n = class_values.numel()

                if n == 1:
                    class_targets = torch.full_like(
                        class_values,
                        0.5,
                        dtype=torch.float32,
                    )
                else:
                    order = torch.argsort(
                        class_values,
                        stable=True,
                    )

                    ranks = torch.empty(
                        n,
                        device=values.device,
                        dtype=torch.float32,
                    )

                    ranks[
                        order
                    ] = torch.linspace(
                        0.0,
                        1.0,
                        steps=n,
                        device=values.device,
                        dtype=torch.float32,
                    )

                    class_targets = ranks

                targets[
                    mask
                ] = class_targets

            return targets


    @torch.no_grad()
    def _per_sample_param_grad_norm(
        self,
        z,
        y,
        logits,
    ):
        """
        Exact per-sample parameter-gradient magnitude of the CURRENT
        linear classifier.

        logits_i = W z_i + b

        delta_i = softmax(logits_i) - onehot(y_i)

        dL_i/dW = delta_i outer z_i
        dL_i/db = delta_i

        Thus:

            ||dL_i/d(W,b)||_2
                = ||delta_i||_2 * sqrt(||z_i||_2^2 + 1)
        """
        probs = torch.softmax(
            logits.detach(),
            dim=1,
        )

        onehot = F.one_hot(
            y,
            num_classes=self.n_classes,
        ).to(
            probs.dtype
        )

        delta = (
            probs - onehot
        )

        delta_norm = torch.linalg.vector_norm(
            delta,
            ord=2,
            dim=1,
        )

        z_norm_sq = (
            z.detach()
            .pow(2)
            .sum(dim=1)
        )

        return (
            delta_norm
            * torch.sqrt(
                z_norm_sq + 1.0
            )
        )


    def _gradient_percentile_mse(
        self,
        raw_scores,
        z,
        y,
        logits,
    ):
        """
        Auxiliary Rater supervision.

            G_i = exact per-sample parameter-gradient norm
            q_i = within-class percentile of G_i
            r_i = sigmoid(raw_rater_score_i)

            L_grad = mean_i (r_i - q_i)^2

        ALL samples contribute.
        Group/background labels are never used.
        """
        grad_norm = (
            self._per_sample_param_grad_norm(
                z=z,
                y=y,
                logits=logits,
            )
        )

        target = (
            self._within_class_percentile_targets(
                grad_norm,
                y,
            )
        )

        prediction = torch.sigmoid(
            raw_scores
        )

        grad_mse = F.mse_loss(
            prediction,
            target,
            reduction="mean",
        )

        return (
            grad_mse,
            grad_norm.detach(),
            target.detach(),
            prediction,
        )


    def _inner_unroll(
        self,
        inner_model,
        train_iterator,
        train_loader,
    ):
        """
        Differentiable ALL-SAMPLE weighted classifier optimization.

        Inner objective:

            L_inner
                = sum_i w_i * CE(
                    h_theta(z_i),
                    y_i
                  )

        where:
            w_i = transform(Rater(z_i))

        There is intentionally NO division by sum(weights).

        The Rater receives its learning signal ONLY through the held-out
        outer classification cross-entropy in _meta_step().
        """
        fast_weight = (
            inner_model.linear.weight
            .detach()
            .clone()
            .requires_grad_(True)
        )

        fast_bias = (
            inner_model.linear.bias
            .detach()
            .clone()
            .requires_grad_(True)
        )

        last_raw_scores = None

        for _ in range(
            self.inner_steps
        ):
            batch, train_iterator = (
                self._next_batch(
                    train_iterator,
                    train_loader,
                )
            )

            z, y, _, _ = batch

            z = z.to(
                self.device,
                non_blocking=True,
            )

            y = y.to(
                self.device,
                non_blocking=True,
            )

            logits = F.linear(
                z,
                fast_weight,
                fast_bias,
            )

            per_sample_loss = F.cross_entropy(
                logits,
                y,
                reduction="none",
            )

            raw_scores = self.rater(
                z
            )

            weights = self._scores_to_weights(
                raw_scores
            )

            inner_loss = (
                per_sample_loss
                * weights
            ).sum()

            if self.inner_reg_weight > 0:
                inner_reg = (
                    fast_weight.pow(2).sum()
                    + fast_bias.pow(2).sum()
                )

                inner_loss = (
                    inner_loss
                    + self.inner_reg_weight
                    * inner_reg
                )

            grad_w, grad_b = torch.autograd.grad(
                inner_loss,
                [
                    fast_weight,
                    fast_bias,
                ],
                create_graph=True,
            )

            fast_weight = (
                fast_weight
                - self.inner_lr
                * grad_w
            )

            fast_bias = (
                fast_bias
                - self.inner_lr
                * grad_b
            )

            last_raw_scores = (
                raw_scores
            )

        fast_params = {
            "weight": fast_weight,
            "bias": fast_bias,
        }

        zero_aux = torch.zeros(
            (),
            device=self.device,
        )

        return (
            fast_params,
            train_iterator,
            last_raw_scores,
            zero_aux,
        )
    # ========================================================
    # One bilevel/meta step
    # ========================================================

    def _meta_step(
        self,
        train_iterator,
        val_iterator,
        train_loader,
        val_loader,
    ):
        """
        One classification-only Rater meta update.

        For every persistent inner classifier:

            theta'
                = differentiable weighted inner update

            L_outer
                = CE(
                    h_theta'(z_outer),
                    y_outer
                  )

        Across the inner-model population:

            L_meta
                = mean_m L_outer^(m)

        This is the ENTIRE Rater objective in this first experiment.
        There is no gradient-MSE auxiliary target and no Rater-side
        regularization term.
        """
        self.rater.train()

        outer_batch, val_iterator = (
            self._next_batch(
                val_iterator,
                val_loader,
            )
        )

        z_outer, y_outer, _, _ = (
            outer_batch
        )

        z_outer = z_outer.to(
            self.device,
            non_blocking=True,
        )

        y_outer = y_outer.to(
            self.device,
            non_blocking=True,
        )

        outer_losses = []
        fast_parameter_sets = []

        for inner_model in self.inner_models:
            (
                fast_params,
                train_iterator,
                _,
                _,
            ) = self._inner_unroll(
                inner_model,
                train_iterator,
                train_loader,
            )

            outer_logits = F.linear(
                z_outer,
                fast_params["weight"],
                fast_params["bias"],
            )

            outer_loss = F.cross_entropy(
                outer_logits,
                y_outer,
            )

            outer_losses.append(
                outer_loss
            )

            fast_parameter_sets.append(
                fast_params
            )

        outer_ce = torch.stack(
            outer_losses
        ).mean()

        # CLASSIFICATION-ONLY META OBJECTIVE.
        meta_loss = outer_ce

        self.outer_optimizer.zero_grad()

        meta_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.rater.parameters(),
                self.grad_clip,
            )

        self.outer_optimizer.step()

        # Persist the inner classifiers' updated states so the next
        # meta-step continues their learning trajectory.
        with torch.no_grad():
            for (
                inner_model,
                fast_params,
            ) in zip(
                self.inner_models,
                fast_parameter_sets,
            ):
                inner_model.linear.weight.copy_(
                    fast_params[
                        "weight"
                    ].detach()
                )

                inner_model.linear.bias.copy_(
                    fast_params[
                        "bias"
                    ].detach()
                )

        return (
            float(
                meta_loss.detach().item()
            ),
            float(
                outer_ce.detach().item()
            ),
            0.0,
            train_iterator,
            val_iterator,
        )
    # ========================================================
    # Evaluation helpers
    # ========================================================

    @torch.no_grad()
    def _evaluate_classifier(self, model, dataset):
        """
        Evaluate an embedding classifier.

        Returns standard loss/accuracy plus Waterbirds-style per-group
        accuracy and worst-group accuracy.
        """

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        group_correct = {}
        group_total = {}

        for z, y, g, _ in loader:
            z = z.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)

            logits = model(z)
            loss = F.cross_entropy(
                logits,
                y,
                reduction="sum",
            )
            pred = logits.argmax(dim=1)

            total_loss += loss.item()
            total_correct += (pred == y).sum().item()
            total_examples += y.numel()

            for group_id in torch.unique(g):
                gid = int(group_id.item())
                mask = g == group_id
                group_correct[gid] = (
                    group_correct.get(gid, 0)
                    + (pred[mask] == y[mask]).sum().item()
                )
                group_total[gid] = (
                    group_total.get(gid, 0)
                    + mask.sum().item()
                )

        if total_examples == 0:
            raise RuntimeError("Cannot evaluate an empty dataset.")

        group_accuracy = {
            gid: group_correct[gid] / group_total[gid]
            for gid in sorted(group_total)
            if group_total[gid] > 0
        }

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
            "group_accuracy": group_accuracy,
            "worst_group_accuracy": min(group_accuracy.values())
            if group_accuracy else float("nan"),
        }

    def _evaluate_inner_population(self, dataset, verbose=True):
        """Evaluate every persistent inner model and aggregate metrics."""

        metrics = []

        for i, model in enumerate(self.inner_models):
            result = self._evaluate_classifier(model, dataset)
            metrics.append(result)

            if verbose:
                group_str = ", ".join(
                    f"g{gid}={100.0 * acc:.2f}%"
                    for gid, acc in result["group_accuracy"].items()
                )
                self._run_log(
                    f"[Inner model {i}] "
                    f"loss={result['loss']:.6f}, "
                    f"acc={100.0 * result['accuracy']:.2f}%, "
                    f"WGA={100.0 * result['worst_group_accuracy']:.2f}% "
                    f"({group_str})"
                )

        mean_acc = float(np.mean([m["accuracy"] for m in metrics]))
        mean_wga = float(
            np.mean([m["worst_group_accuracy"] for m in metrics])
        )
        mean_loss = float(np.mean([m["loss"] for m in metrics]))
        best_acc = float(max(m["accuracy"] for m in metrics))
        best_wga = float(
            max(m["worst_group_accuracy"] for m in metrics)
        )

        if verbose:
            self._run_log(
                f"[Inner population] mean_loss={mean_loss:.6f}, "
                f"mean_acc={100.0 * mean_acc:.2f}%, "
                f"mean_WGA={100.0 * mean_wga:.2f}%, "
                f"best_acc={100.0 * best_acc:.2f}%, "
                f"best_WGA={100.0 * best_wga:.2f}%"
            )

        return {
            "models": metrics,
            "mean_loss": mean_loss,
            "mean_accuracy": mean_acc,
            "mean_wga": mean_wga,
            "best_accuracy": best_acc,
            "best_wga": best_wga,
        }

    @torch.no_grad()
    def _score_loss_relationship(self, dataset, models=None):
        """
        Diagnostic relationship between raw Rater scores, effective
        EA-style weights, classifier loss, and correctness.

        For each inner model:
            correct -> weight 1
            wrong   -> Rater rating

        `final_scores` is the mean effective weight across the current
        inner-model population.
        """

        if models is None:
            models = self.inner_models

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.rater.eval()
        for model in models:
            model.eval()

        all_scores = []
        all_final_scores = []
        all_losses = []
        all_correct = []
        all_groups = []
        all_labels = []
        all_attrs = []

        for z, y, g, a in loader:
            z = z.to(self.device, non_blocking=True)
            y_device = y.to(self.device, non_blocking=True)

            # Raw scores for all samples are used only for diagnostics.
            raw_scores = self.rater(z)

            model_losses = []
            model_correct = []
            model_weights = []

            for model in models:
                logits = model(z)
                per_loss = F.cross_entropy(
                    logits,
                    y_device,
                    reduction="none",
                )

                effective_weights, correct_mask, _ = (
                    self._all_sample_training_weights(
                        z=z,
                        y=y_device,
                        logits=logits,
                        raw_scores_all=raw_scores,
                    )
                )

                model_losses.append(per_loss)
                model_correct.append(correct_mask.float())
                model_weights.append(effective_weights)

            mean_loss = torch.stack(model_losses, dim=0).mean(dim=0)
            mean_correct = torch.stack(model_correct, dim=0).mean(dim=0)
            mean_weight = torch.stack(model_weights, dim=0).mean(dim=0)

            all_scores.append(raw_scores.cpu())
            all_final_scores.append(mean_weight.cpu())
            all_losses.append(mean_loss.cpu())
            all_correct.append(mean_correct.cpu())
            all_groups.append(g.cpu())
            all_labels.append(y.cpu())
            all_attrs.append(a.cpu())

        scores = torch.cat(all_scores).numpy()
        final_scores = torch.cat(all_final_scores).numpy()
        losses = torch.cat(all_losses).numpy()
        correctness = torch.cat(all_correct).numpy()
        groups = torch.cat(all_groups).numpy()
        labels = torch.cat(all_labels).numpy()
        attrs = torch.cat(all_attrs).numpy()

        def _corr(x, y):
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                return (
                    float(stats.spearmanr(x, y).statistic),
                    float(stats.pearsonr(x, y).statistic),
                )
            return 0.0, 0.0

        spearman_loss, pearson_loss = _corr(scores, losses)
        spearman_final_loss, pearson_final_loss = _corr(
            final_scores, losses
        )

        if len(scores) >= 2 and np.std(scores) > 0 and np.std(correctness) > 0:
            spearman_correct = float(
                stats.spearmanr(scores, correctness).statistic
            )
        else:
            spearman_correct = 0.0

        if len(final_scores) >= 2 and np.std(final_scores) > 0 and np.std(correctness) > 0:
            spearman_final_correct = float(
                stats.spearmanr(final_scores, correctness).statistic
            )
        else:
            spearman_final_correct = 0.0

        return {
            "scores": scores,
            "final_scores": final_scores,
            "training_weights": final_scores,
            "mean_loss": losses,
            "mean_correctness": correctness,
            "groups": groups,
            "labels": labels,
            "attrs": attrs,
            "spearman_score_vs_loss": spearman_loss,
            "pearson_score_vs_loss": pearson_loss,
            "spearman_score_vs_correctness": spearman_correct,
            "spearman_final_score_vs_loss": spearman_final_loss,
            "pearson_final_score_vs_loss": pearson_final_loss,
            "spearman_final_score_vs_correctness": spearman_final_correct,
        }

    def _group_display_name(self, gid):
        """Readable group labels for Waterbirds; generic labels otherwise."""

        if self.config.dataset == "waterbirds":
            names = {
                0: "Group 0: landbird / land",
                1: "Group 1: landbird / water",
                2: "Group 2: waterbird / land",
                3: "Group 3: waterbird / water",
            }
            return names.get(int(gid), f"Group {int(gid)}")

        return f"Group {int(gid)}"

    def _save_rate_vs_loss_plot(
        self,
        relationship,
        output_dir,
        meta_step,
        split_name="val",
        tag_prefix="meta_step",
    ):
        """
        Save a group-colored scatter plot of raw rater score vs mean
        per-example inner-model CE loss.

        Each point is one embedding. Each Waterbirds group is drawn as a
        separate scatter series, so matplotlib automatically assigns a
        different color to each group.
        """

        scores = np.asarray(relationship["scores"])
        final_scores = np.asarray(relationship["final_scores"])
        losses = np.asarray(relationship["mean_loss"])
        groups = np.asarray(relationship["groups"])
        labels = np.asarray(relationship["labels"])
        attrs = np.asarray(relationship["attrs"])
        correctness = np.asarray(
            relationship["mean_correctness"]
        )

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "rate_vs_loss",
        )
        data_dir = os.path.join(
            output_dir,
            "plots",
            "rate_vs_loss_data",
        )

        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        for gid in sorted(np.unique(groups)):
            mask = groups == gid
            ax.scatter(
                losses[mask],
                scores[mask],
                alpha=0.50,
                s=24,
                label=self._group_display_name(gid),
            )

        # Overall least-squares line, matching the spirit of the user's
        # previous evaluation script.
        if (
            len(losses) >= 2
            and np.std(losses) > 0
            and np.std(scores) > 0
        ):
            slope, intercept, r_value, _, _ = stats.linregress(
                losses,
                scores,
            )
            x_line = np.linspace(
                float(losses.min()),
                float(losses.max()),
                200,
            )
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                linewidth=2,
                label=f"Overall linear fit (R²={r_value ** 2:.3f})",
            )

        ax.set_xlabel(
            "Mean per-example cross-entropy loss across inner models"
        )
        ax.set_ylabel("Raw rater score")
        ax.set_title(
            f"Rater score vs inner-model loss | {split_name} | "
            f"step {meta_step}\n"
            f"Spearman={relationship['spearman_score_vs_loss']:.3f}, "
            f"Pearson={relationship['pearson_score_vs_loss']:.3f}"
        )
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        plot_path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{meta_step:06d}_{split_name}.png",
        )
        fig.savefig(
            plot_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        if self.save_plot_data:
            csv_path = os.path.join(
                data_dir,
                f"{tag_prefix}_{meta_step:06d}_{split_name}.csv",
            )

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "index",
                        "raw_rater_score",
                        "effective_ea_style_weight",
                        "mean_inner_loss",
                        "mean_inner_correctness",
                        "group",
                        "label",
                        "attribute",
                        "weighting_method",
                        "temperature",
                    ]
                )

                for i in range(len(scores)):
                    writer.writerow(
                        [
                            i,
                            float(scores[i]),
                            float(final_scores[i]),
                            float(losses[i]),
                            float(correctness[i]),
                            int(groups[i]),
                            int(labels[i]),
                            int(attrs[i]),
                            self.weighting,
                            float(self.temperature),
                        ]
                    )

        self._run_log(
            f"[Rater plot] saved score-vs-loss plot: {plot_path}"
        )

        return plot_path

    def _save_group_score_histogram(
        self,
        relationship,
        output_dir,
        meta_step,
        split_name="val",
        tag_prefix="meta_step",
        score_kind="raw",
    ):
        """
        Save the distribution of Rater outputs separately for every group.

        score_kind="raw":
            histogram of unrestricted rater outputs r_eta(z).

        score_kind="final":
            histogram of the EFFECTIVE EA-STYLE WEIGHT actually used:
            correct samples have weight 1; misclassified samples use the
            selected Rater rating transform.

        Density normalization is used so strongly imbalanced Waterbirds
        groups can be compared by distribution shape rather than count.
        """

        if score_kind == "raw":
            values = np.asarray(relationship["scores"])
            folder = "raw_score_histogram"
            xlabel = "Raw rater score"
            title_name = "Raw Rater score distribution"
            filename_kind = "raw_score_hist"
        elif score_kind == "final":
            values = np.asarray(relationship["final_scores"])
            folder = f"final_score_histogram_{self.weighting}"
            xlabel = (
                "Effective ALL-SAMPLE Rater weight "
                "(every sample=Rater weight)"
            )
            title_name = (
                f"ALL-SAMPLE Rater weight distribution "
                f"(wrong uses {self.weighting})"
            )
            filename_kind = "final_score_hist"
        else:
            raise ValueError(
                f"Unknown score_kind={score_kind}; expected 'raw' or 'final'."
            )

        groups = np.asarray(relationship["groups"])

        plot_dir = os.path.join(
            output_dir,
            "plots",
            folder,
        )
        os.makedirs(plot_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        if len(values) == 0:
            plt.close(fig)
            return None

        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.05, 1e-6)
            bins = np.linspace(vmin - eps, vmax + eps, 20)
        else:
            bins = np.linspace(vmin, vmax, 41)

        for gid in sorted(np.unique(groups)):
            mask = groups == gid
            group_values = values[mask]
            if len(group_values) == 0:
                continue

            ax.hist(
                group_values,
                bins=bins,
                density=True,
                alpha=0.45,
                label=(
                    f"{self._group_display_name(gid)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(
            f"{title_name} by group | {split_name} | step {meta_step}"
        )
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        plot_path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{meta_step:06d}_{split_name}_{filename_kind}.png",
        )
        fig.savefig(
            plot_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        self._run_log(
            f"[Rater plot] saved {score_kind}-score histogram: "
            f"{plot_path}"
        )
        return plot_path

    def _save_final_score_vs_loss_plot(
        self,
        relationship,
        output_dir,
        meta_step,
        split_name="val",
        tag_prefix="meta_step",
    ):
        """
        Save group-colored scatter of FINAL SCORE vs classifier loss.

        FINAL SCORE means the effective ALL-SAMPLE Rater weight:
        every sample -> Rater weight.
        """

        final_scores = np.asarray(relationship["final_scores"])
        losses = np.asarray(relationship["mean_loss"])
        groups = np.asarray(relationship["groups"])

        plot_dir = os.path.join(
            output_dir,
            "plots",
            f"final_score_vs_loss_{self.weighting}",
        )
        os.makedirs(plot_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        for gid in sorted(np.unique(groups)):
            mask = groups == gid
            ax.scatter(
                losses[mask],
                final_scores[mask],
                alpha=0.50,
                s=24,
                label=self._group_display_name(gid),
            )

        if (
            len(losses) >= 2
            and np.std(losses) > 0
            and np.std(final_scores) > 0
        ):
            slope, intercept, r_value, _, _ = stats.linregress(
                losses,
                final_scores,
            )
            x_line = np.linspace(
                float(losses.min()),
                float(losses.max()),
                200,
            )
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                linewidth=2,
                label=f"Overall linear fit (R²={r_value ** 2:.3f})",
            )

        ax.set_xlabel(
            "Mean per-example cross-entropy loss across inner models"
        )
        ax.set_ylabel(
            "Effective EA-style weight (every sample=Rater weight)"
        )
        ax.set_title(
            f"ALL-SAMPLE Rater weight vs inner-model loss "
            f"(wrong uses {self.weighting}) | "
            f"{split_name} | "
            f"step {meta_step}\n"
            f"Spearman={relationship['spearman_final_score_vs_loss']:.3f}, "
            f"Pearson={relationship['pearson_final_score_vs_loss']:.3f}"
        )
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        plot_path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{meta_step:06d}_{split_name}.png",
        )
        fig.savefig(
            plot_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        self._run_log(
            f"[Rater plot] saved final-score-vs-loss plot: {plot_path}"
        )
        return plot_path

    def _save_all_score_diagnostics(
        self,
        relationship,
        output_dir,
        meta_step,
        split_name="val",
        tag_prefix="meta_step",
    ):
        """Save all score diagnostics for one evaluation step."""

        self._save_rate_vs_loss_plot(
            relationship=relationship,
            output_dir=output_dir,
            meta_step=meta_step,
            split_name=split_name,
            tag_prefix=tag_prefix,
        )
        self._save_group_score_histogram(
            relationship=relationship,
            output_dir=output_dir,
            meta_step=meta_step,
            split_name=split_name,
            tag_prefix=tag_prefix,
            score_kind="raw",
        )
        self._save_group_score_histogram(
            relationship=relationship,
            output_dir=output_dir,
            meta_step=meta_step,
            split_name=split_name,
            tag_prefix=tag_prefix,
            score_kind="final",
        )
        self._save_final_score_vs_loss_plot(
            relationship=relationship,
            output_dir=output_dir,
            meta_step=meta_step,
            split_name=split_name,
            tag_prefix=tag_prefix,
        )

    @torch.no_grad()
    def _classifier_score_loss_relationship(self, dataset, model):
        """Raw score and effective EA-style weight vs loss for one model."""

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.rater.eval()
        model.eval()

        all_scores = []
        all_final_scores = []
        all_losses = []
        all_correct = []
        all_groups = []
        all_labels = []
        all_attrs = []

        for z, y, g, a in loader:
            z = z.to(self.device, non_blocking=True)
            y_device = y.to(self.device, non_blocking=True)

            raw_scores = self.rater(z)
            logits = model(z)

            effective_weights, correct_mask, _ = (
                self._all_sample_training_weights(
                    z=z,
                    y=y_device,
                    logits=logits,
                    raw_scores_all=raw_scores,
                )
            )

            losses = F.cross_entropy(
                logits,
                y_device,
                reduction="none",
            )

            all_scores.append(raw_scores.cpu())
            all_final_scores.append(effective_weights.cpu())
            all_losses.append(losses.cpu())
            all_correct.append(correct_mask.float().cpu())
            all_groups.append(g.cpu())
            all_labels.append(y.cpu())
            all_attrs.append(a.cpu())

        scores = torch.cat(all_scores).numpy()
        final_scores = torch.cat(all_final_scores).numpy()
        losses = torch.cat(all_losses).numpy()
        correctness = torch.cat(all_correct).numpy()
        groups = torch.cat(all_groups).numpy()
        labels = torch.cat(all_labels).numpy()
        attrs = torch.cat(all_attrs).numpy()

        def _corr(x, y):
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                return (
                    float(stats.spearmanr(x, y).statistic),
                    float(stats.pearsonr(x, y).statistic),
                )
            return 0.0, 0.0

        spearman_loss, pearson_loss = _corr(scores, losses)
        spearman_final_loss, pearson_final_loss = _corr(final_scores, losses)

        if len(scores) >= 2 and np.std(scores) > 0 and np.std(correctness) > 0:
            spearman_correct = float(stats.spearmanr(scores, correctness).statistic)
        else:
            spearman_correct = 0.0

        if len(final_scores) >= 2 and np.std(final_scores) > 0 and np.std(correctness) > 0:
            spearman_final_correct = float(
                stats.spearmanr(final_scores, correctness).statistic
            )
        else:
            spearman_final_correct = 0.0

        return {
            "scores": scores,
            "final_scores": final_scores,
            "training_weights": final_scores,
            "mean_loss": losses,
            "mean_correctness": correctness,
            "groups": groups,
            "labels": labels,
            "attrs": attrs,
            "spearman_score_vs_loss": spearman_loss,
            "pearson_score_vs_loss": pearson_loss,
            "spearman_score_vs_correctness": spearman_correct,
            "spearman_final_score_vs_loss": spearman_final_loss,
            "pearson_final_score_vs_loss": pearson_final_loss,
            "spearman_final_score_vs_correctness": spearman_final_correct,
        }

    @torch.no_grad()
    def _compute_scores(self, dataset, model=None):
        """
        Save raw Rater scores and, when a classifier is supplied, the
        effective ALL-SAMPLE Rater weights for that classifier.
        """

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.rater.eval()
        if model is not None:
            model.eval()

        scores, final_scores = [], []
        labels, groups, attrs, correctness = [], [], [], []

        for z, y, g, a in loader:
            z = z.to(self.device, non_blocking=True)
            y_device = y.to(self.device, non_blocking=True)
            batch_scores = self.rater(z)

            if model is None:
                batch_final_scores = self._scores_to_weights(batch_scores)
                batch_correctness = torch.full(
                    (y_device.shape[0],),
                    float("nan"),
                    device=self.device,
                )
            else:
                logits = model(z)
                batch_final_scores, correct_mask, _ = (
                    self._all_sample_training_weights(
                        z=z,
                        y=y_device,
                        logits=logits,
                        raw_scores_all=batch_scores,
                    )
                )
                batch_correctness = correct_mask.float()

            scores.append(batch_scores.cpu())
            final_scores.append(batch_final_scores.cpu())
            correctness.append(batch_correctness.cpu())
            labels.append(y.cpu())
            groups.append(g.cpu())
            attrs.append(a.cpu())

        return {
            "scores": torch.cat(scores),
            "final_scores": torch.cat(final_scores),
            "training_weights": torch.cat(final_scores),
            "correctness": torch.cat(correctness),
            "labels": torch.cat(labels),
            "groups": torch.cat(groups),
            "attrs": torch.cat(attrs),
            "weights_are_all_sample_rater": model is not None,
        }

    def _rating_summary(self, payload):
        scores = payload["scores"]
        final_scores = payload.get("final_scores", None)
        labels = payload["labels"]
        groups = payload["groups"]

        lines = [
            f"raw score mean={scores.mean():.6f}, "
            f"std={scores.std():.6f}, "
            f"min={scores.min():.6f}, "
            f"max={scores.max():.6f}"
        ]

        if final_scores is not None:
            lines.append(
                f"saved final/effective weight mean="
                f"{final_scores.mean():.6f}, "
                f"std={final_scores.std():.6f}, "
                f"min={final_scores.min():.6f}, "
                f"max={final_scores.max():.6f}"
            )

        for c in torch.unique(labels):
            mask = labels == c
            msg = (
                f"class {int(c)}: n={int(mask.sum())}, "
                f"mean_raw_score={scores[mask].mean():.6f}"
            )
            if final_scores is not None:
                msg += (
                    f", mean_final_score="
                    f"{final_scores[mask].mean():.6f}"
                )
            lines.append(msg)

        for g in torch.unique(groups):
            mask = groups == g
            msg = (
                f"group {int(g)}: n={int(mask.sum())}, "
                f"mean_raw_score={scores[mask].mean():.6f}"
            )
            if final_scores is not None:
                msg += (
                    f", mean_final_score="
                    f"{final_scores[mask].mean():.6f}"
                )
            lines.append(msg)

        return lines


    def _save_rating_boxplot(
        self,
        payload,
        output_dir,
        split_name,
        score_kind="raw",
        tag_prefix="final_rater",
    ):
        if score_kind == "raw":
            values = np.asarray(
                payload["scores"]
            )
            ylabel = "Raw Rater score"
            suffix = "raw_score_boxplot"
        else:
            values = np.asarray(
                payload["final_scores"]
            )
            ylabel = (
                f"Transformed Rater weight "
                f"({self.weighting})"
            )
            suffix = "weight_boxplot"

        groups = np.asarray(
            payload["groups"]
        )

        unique_groups = sorted(
            np.unique(groups)
        )

        data = [
            values[
                groups == gid
            ]
            for gid in unique_groups
        ]

        labels = [
            self._group_display_name(
                gid
            )
            for gid in unique_groups
        ]

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        try:
            ax.boxplot(
                data,
                tick_labels=labels,
                showfliers=False,
            )
        except TypeError:
            ax.boxplot(
                data,
                labels=labels,
                showfliers=False,
            )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{ylabel} by Waterbirds group | {split_name}"
        )

        ax.tick_params(
            axis="x",
            labelrotation=20,
        )

        ax.grid(
            True,
            axis="y",
            linestyle=":",
            alpha=0.30,
        )

        fig.tight_layout()

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "rating_boxplots",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{split_name}_{suffix}.png",
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        return path

    def _save_rating_ecdf(
        self,
        payload,
        output_dir,
        split_name,
        score_kind="raw",
        tag_prefix="final_rater",
    ):
        if score_kind == "raw":
            values = np.asarray(
                payload["scores"]
            )
            xlabel = "Raw Rater score"
            suffix = "raw_score_ecdf"
        else:
            values = np.asarray(
                payload["final_scores"]
            )
            xlabel = (
                f"Transformed Rater weight "
                f"({self.weighting})"
            )
            suffix = "weight_ecdf"

        groups = np.asarray(
            payload["groups"]
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        for gid in sorted(
            np.unique(groups)
        ):
            group_values = np.sort(
                values[
                    groups == gid
                ]
            )

            if len(group_values) == 0:
                continue

            y = np.arange(
                1,
                len(group_values) + 1,
                dtype=np.float64,
            ) / len(group_values)

            ax.step(
                group_values,
                y,
                where="post",
                linewidth=2,
                label=self._group_display_name(
                    gid
                ),
            )

        ax.set_xlabel(
            xlabel
        )

        ax.set_ylabel(
            "Empirical CDF"
        )

        ax.set_title(
            f"{xlabel} ECDF by Waterbirds group | {split_name}"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        ax.legend(
            fontsize=9,
        )

        fig.tight_layout()

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "rating_ecdf",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{split_name}_{suffix}.png",
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        return path

    def _save_rating_class_histogram(
        self,
        payload,
        output_dir,
        split_name,
        score_kind="raw",
        tag_prefix="final_rater",
    ):
        if score_kind == "raw":
            values = np.asarray(
                payload["scores"]
            )
            xlabel = "Raw Rater score"
            suffix = "raw_score_by_class"
        else:
            values = np.asarray(
                payload["final_scores"]
            )
            xlabel = (
                f"Transformed Rater weight "
                f"({self.weighting})"
            )
            suffix = "weight_by_class"

        labels = np.asarray(
            payload["labels"]
        )

        if len(values) == 0:
            return None

        vmin = float(
            np.min(values)
        )

        vmax = float(
            np.max(values)
        )

        if np.isclose(
            vmin,
            vmax,
        ):
            eps = max(
                abs(vmin) * 0.05,
                1e-6,
            )

            bins = np.linspace(
                vmin - eps,
                vmax + eps,
                30,
            )
        else:
            bins = np.linspace(
                vmin,
                vmax,
                41,
            )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        for class_id in sorted(
            np.unique(labels)
        ):
            mask = (
                labels == class_id
            )

            ax.hist(
                values[
                    mask
                ],
                bins=bins,
                density=True,
                alpha=0.45,
                label=(
                    f"class {int(class_id)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax.set_xlabel(
            xlabel
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            f"{xlabel} by class | {split_name}"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        ax.legend()

        fig.tight_layout()

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "rating_class_histograms",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        path = os.path.join(
            plot_dir,
            f"{tag_prefix}_{split_name}_{suffix}.png",
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        return path

    def _save_final_rater_distribution_plots(
        self,
        payload,
        output_dir,
        split_name,
    ):
        relationship_like = {
            "scores": np.asarray(
                payload["scores"]
            ),
            "final_scores": np.asarray(
                payload["final_scores"]
            ),
            "groups": np.asarray(
                payload["groups"]
            ),
        }

        self._save_group_score_histogram(
            relationship=relationship_like,
            output_dir=output_dir,
            meta_step=self.meta_steps,
            split_name=split_name,
            tag_prefix="final_rater",
            score_kind="raw",
        )

        self._save_group_score_histogram(
            relationship=relationship_like,
            output_dir=output_dir,
            meta_step=self.meta_steps,
            split_name=split_name,
            tag_prefix="final_rater",
            score_kind="final",
        )

        for score_kind in (
            "raw",
            "final",
        ):
            self._save_rating_boxplot(
                payload=payload,
                output_dir=output_dir,
                split_name=split_name,
                score_kind=score_kind,
            )

            self._save_rating_ecdf(
                payload=payload,
                output_dir=output_dir,
                split_name=split_name,
                score_kind=score_kind,
            )

            self._save_rating_class_histogram(
                payload=payload,
                output_dir=output_dir,
                split_name=split_name,
                score_kind=score_kind,
            )

    def _save_rate_trajectory_plots(
        self,
        rate_trajectory,
        output_dir,
    ):
        if not rate_trajectory:
            return

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "meta_rate_trajectories",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        for value_key, ylabel, filename in [
            (
                "mean_raw_score",
                "Mean raw Rater score",
                "mean_raw_score_by_group_over_meta_steps.png",
            ),
            (
                "mean_weight",
                "Mean transformed Rater weight",
                "mean_weight_by_group_over_meta_steps.png",
            ),
        ]:
            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            groups = sorted(
                {
                    int(row["group"])
                    for row in rate_trajectory
                }
            )

            for gid in groups:
                rows = [
                    row
                    for row in rate_trajectory
                    if int(
                        row["group"]
                    ) == gid
                ]

                rows = sorted(
                    rows,
                    key=lambda row: int(
                        row["meta_step"]
                    ),
                )

                ax.plot(
                    [
                        row["meta_step"]
                        for row in rows
                    ],
                    [
                        row[value_key]
                        for row in rows
                    ],
                    marker="o",
                    linewidth=1.5,
                    markersize=3,
                    label=self._group_display_name(
                        gid
                    ),
                )

            ax.set_xlabel(
                "Meta step"
            )

            ax.set_ylabel(
                ylabel
            )

            ax.set_title(
                f"{ylabel} by group during Rater meta-learning"
            )

            ax.grid(
                True,
                linestyle=":",
                alpha=0.30,
            )

            ax.legend(
                fontsize=9,
            )

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    plot_dir,
                    filename,
                ),
                dpi=160,
                bbox_inches="tight",
            )

            plt.close(fig)

    def _save_meta_loss_plot(
        self,
        history,
        output_dir,
    ):
        if not history:
            return

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "training_curves",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        steps = [
            row[0]
            for row in history
        ]

        losses = [
            row[1]
            for row in history
        ]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            steps,
            losses,
            linewidth=1.8,
        )

        ax.set_xlabel(
            "Meta step"
        )

        ax.set_ylabel(
            "Outer classification CE"
        )

        ax.set_title(
            "Rater meta-learning objective"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                plot_dir,
                "meta_outer_classification_ce.png",
            ),
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # Checkpoints
    # ========================================================

    def _save_rater_checkpoint(
        self,
        path,
        meta_step,
        meta_loss,
        outer_ce=None,
        grad_mse=None,
    ):
        """
        Compact reusable Rater checkpoint.

        No optimizer state, inner classifiers, embedding tensors, or
        diagnostic arrays are stored.
        """
        torch.save(
            {
                "rater_sd": self.rater.state_dict(),
                "meta_step": int(
                    meta_step
                ),
                "meta_loss": float(
                    meta_loss
                ),
                "outer_ce": (
                    None
                    if outer_ce is None
                    else float(
                        outer_ce
                    )
                ),
                "meta_objective": (
                    "outer_classification_cross_entropy_only"
                ),
                "feature_dim": int(
                    self.feature_dim
                ),
                "num_classes": int(
                    self.n_classes
                ),
                "backbone": str(
                    self.config.backbone
                ),
                "rater_capacity": str(
                    self.rater_capacity
                ),
                "rater_weighting": str(
                    self.weighting
                ),
                "rater_temperature": float(
                    self.temperature
                ),
                "erm_model_path": str(
                    self.erm_model_path
                ),
                "erm_checkpoint_fingerprint": str(
                    self.erm_checkpoint_fingerprint
                ),
                "erm_checkpoint_epoch": (
                    self.erm_checkpoint_epoch
                ),
                "inner_initialization": (
                    "saved_erm_classifier"
                ),
                "num_inner_models": int(
                    self.num_inner_models
                ),
                "inner_init_noise_std": float(
                    self.inner_init_noise_std
                ),
            },
            path,
        )

    def _load_rater_checkpoint(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Rater checkpoint does not exist: {path}")

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if "rater_sd" in checkpoint:
            self.rater.load_state_dict(checkpoint["rater_sd"])
        else:
            # Also permit loading a raw rater state_dict.
            self.rater.load_state_dict(checkpoint)

        self._run_log(f"[Rater] Loaded checkpoint from {path}")

    def _save_population_snapshot(
        self,
        output_dir,
        meta_step,
    ):
        # Disabled in minimal-storage mode.
        return

    def _save_final_classifier(
        self,
        path,
        model,
    ):
        # Final classifier is evaluated in memory only.
        return

    def _load_final_classifier(self, path):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        model = self._new_final_classifier()
        if "model_sd" in checkpoint:
            model.load_state_dict(checkpoint["model_sd"])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    # ========================================================
    # Split handling
    # ========================================================

    def _has_ea_calibration_split(self):
        """
        Return True when the validation set has been split into
        val_subset1 / val_subset2.
        """
        return (
            float(
                getattr(
                    self.config,
                    "split_val",
                    1.0,
                )
            ) < 1.0
            and "val_subset1" in self.dataloaders
            and "val_subset2" in self.dataloaders
        )

    def _resolve_meta_splits(self, split):
        """
        Rater bilevel data protocol.

        EA-style calibration mode (--split_val < 1):
            inner/meta-train = train_no_aug
            outer/meta-val   = val_subset1

        This reserves val_subset2 for final-classifier model selection.

        Legacy mode (--split_val == 1):
            inner/meta-train = train_no_aug
            outer/meta-val   = val
        """

        if split in {"train", "train_no_aug"}:
            inner_split = "train_no_aug"

            if (
                self.use_calibration_final
                and self._has_ea_calibration_split()
            ):
                return inner_split, "val_subset1"

            return inner_split, "val"

        if split == "train_subset1":
            if "train_no_aug_subset1" in self.dataloaders:
                inner_split = "train_no_aug_subset1"
            else:
                inner_split = split

            if (
                self.use_calibration_final
                and self._has_ea_calibration_split()
            ):
                return inner_split, "val_subset1"

            return inner_split, "val"

        if split == "val_subset1":
            if "val_subset2" not in self.dataloaders:
                raise ValueError(
                    "val_subset1 requires --split_val < 1 so that "
                    "val_subset2 exists."
                )
            return "val_subset1", "val_subset2"

        return split, "val"

    def _resolve_final_classifier_splits(self):
        """
        Final classifier protocol with the trained Rater frozen.

        Meta-learning:
            train_no_aug -> differentiable inner updates
            val_subset1  -> Rater outer classification loss

        Final classifier:
            train_no_aug -> weighted classifier fitting
            val_subset2  -> checkpoint/model selection
            test         -> diagnostic/final reporting
        """
        if (
            self.use_calibration_final
            and self._has_ea_calibration_split()
        ):
            return (
                "train_no_aug",
                "val_subset2",
            )

        return (
            "train_no_aug",
            "val",
        )

    # ========================================================
    # Final-classifier epoch diagnostics
    # ========================================================

    def _save_final_epoch_weight_histograms(
        self,
        output_dir,
        epoch,
        weights,
        groups,
        correctness,
        split_name="train",
    ):
        """
        Save ACTUAL ALL-SAMPLE Rater weights used during one
        final-classifier epoch.

        Correctness is diagnostic only; it does not gate the weights.
        """
        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        groups = np.asarray(
            groups,
            dtype=np.int64,
        )

        correctness = np.asarray(
            correctness,
            dtype=np.float64,
        )

        if len(weights) == 0:
            return

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "final_classifier_epoch_all_sample_weight_histogram",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        finite = weights[
            np.isfinite(weights)
        ]

        if len(finite) == 0:
            return

        wmin = float(
            finite.min()
        )

        wmax = float(
            finite.max()
        )

        if np.isclose(
            wmin,
            wmax,
        ):
            eps = max(
                abs(wmin) * 0.05,
                1e-6,
            )

            bins = np.linspace(
                wmin - eps,
                wmax + eps,
                30,
            )
        else:
            bins = np.linspace(
                wmin,
                wmax,
                41,
            )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        for gid in sorted(
            np.unique(groups)
        ):
            mask = groups == gid

            if not np.any(mask):
                continue

            ax.hist(
                weights[mask],
                bins=bins,
                density=True,
                alpha=0.45,
                label=(
                    f"{self._group_display_name(gid)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax.set_xlabel(
            f"ALL-SAMPLE trained-Rater {self.weighting} weight"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            f"Final-classifier ALL-SAMPLE training weights by group | "
            f"epoch {epoch}\\n"
            f"correct before update="
            f"{100.0 * correctness.mean():.2f}%"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        ax.legend(
            loc="best",
            fontsize=9,
        )

        fig.tight_layout()

        plot_path = os.path.join(
            plot_dir,
            f"epoch_{epoch:03d}_{split_name}_all_sample_weight_hist.png",
        )

        fig.savefig(
            plot_path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        summary_parts = []

        for gid in sorted(
            np.unique(groups)
        ):
            mask = groups == gid

            summary_parts.append(
                f"g{int(gid)}: "
                f"mean_w={weights[mask].mean():.3f}, "
                f"std_w={weights[mask].std():.3f}, "
                f"acc_before_update="
                f"{100.0 * correctness[mask].mean():.1f}%"
            )

        self._run_log(
            f"[Final classifier epoch {epoch:03d} ALL-SAMPLE weights] "
            + "; ".join(summary_parts)
        )

        self._run_log(
            f"[Final classifier plot] saved ALL-SAMPLE weight histogram: "
            f"{plot_path}"
        )

    # ========================================================
    # Final weighted classifier
    # ========================================================

    def _selection_value(self, metrics):
        if self.final_selection == "accuracy":
            return metrics["accuracy"]
        if self.final_selection == "wga":
            return metrics["worst_group_accuracy"]
        # Lower loss is better, so negate it to retain max-selection logic.
        return -metrics["loss"]

    def _train_final_classifier(
        self,
        calibration_dataset,
        selection_dataset,
        test_dataset,
        output_dir,
        calibration_split="train_no_aug",
        selection_split="val",
        test_split="test",
    ):
        """
        Train the downstream classifier with the trained Rater frozen.

        The classifier starts from the exact saved ERM head, trains on the
        full train_no_aug embeddings, is selected on val_subset2 when
        available, and is evaluated on test.

        No classifier checkpoints or CSV histories are saved.
        """
        self._run_log(
            "[Rater] Training final classifier from the saved ERM head "
            "with the trained Rater COMPLETELY FROZEN."
        )

        self._run_log(
            f"[Final classifier] weighted train = "
            f"{calibration_split} ({len(calibration_dataset)} samples)"
        )

        self._run_log(
            f"[Final classifier] selection = "
            f"{selection_split} ({len(selection_dataset)} samples)"
        )

        self._run_log(
            f"[Final classifier] test = "
            f"{test_split} ({len(test_dataset)} samples)"
        )

        self.rater.eval()

        for p in self.rater.parameters():
            p.requires_grad_(False)

        model = self._new_final_classifier()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.final_lr,
            momentum=self.final_momentum,
            weight_decay=self.final_weight_decay,
        )

        train_loader = DataLoader(
            calibration_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        best_value = -float("inf")
        best_state = None
        best_epoch = None
        best_metrics = None

        history = []

        for epoch in range(
            1,
            self.final_epochs + 1,
        ):
            model.train()

            running_weighted_loss = 0.0
            total_correct = 0
            total_examples = 0
            num_batches = 0

            epoch_weights = []
            epoch_groups = []
            epoch_correctness = []

            for z, y, g, _ in train_loader:
                z = z.to(
                    self.device,
                    non_blocking=True,
                )

                y = y.to(
                    self.device,
                    non_blocking=True,
                )

                g = torch.as_tensor(
                    g
                ).to(
                    self.device,
                    non_blocking=True,
                )

                logits = model(
                    z
                )

                with torch.no_grad():
                    raw_scores = self.rater(
                        z
                    )

                    weights = self._scores_to_weights(
                        raw_scores
                    )

                    correct_mask = (
                        logits.argmax(
                            dim=1
                        ).eq(
                            y
                        )
                    )

                epoch_weights.append(
                    weights.detach().cpu()
                )

                epoch_groups.append(
                    g.detach().cpu()
                )

                epoch_correctness.append(
                    correct_mask.float()
                    .detach()
                    .cpu()
                )

                per_sample_loss = F.cross_entropy(
                    logits,
                    y,
                    reduction="none",
                )

                weighted_loss = (
                    per_sample_loss
                    * weights
                ).sum()

                optimizer.zero_grad()

                weighted_loss.backward()

                optimizer.step()

                running_weighted_loss += float(
                    weighted_loss.item()
                )

                total_correct += int(
                    correct_mask.sum().item()
                )

                total_examples += int(
                    y.numel()
                )

                num_batches += 1

            train_acc = (
                total_correct
                / max(
                    total_examples,
                    1,
                )
            )

            train_weighted_loss = (
                running_weighted_loss
                / max(
                    num_batches,
                    1,
                )
            )

            epoch_weights_np = torch.cat(
                epoch_weights
            ).numpy()

            epoch_groups_np = torch.cat(
                epoch_groups
            ).numpy()

            epoch_correctness_np = torch.cat(
                epoch_correctness
            ).numpy()

            should_plot_final = (
                epoch == 1
                or epoch % self.final_plot_freq == 0
                or epoch == self.final_epochs
            )

            if should_plot_final:
                self._save_final_epoch_weight_histograms(
                    output_dir=output_dir,
                    epoch=epoch,
                    weights=epoch_weights_np,
                    groups=epoch_groups_np,
                    correctness=epoch_correctness_np,
                    split_name=calibration_split,
                )

            selection_metrics = self._evaluate_classifier(
                model,
                selection_dataset,
            )

            selection_value = self._selection_value(
                selection_metrics
            )

            test_metrics = self._evaluate_classifier(
                model,
                test_dataset,
            )

            history.append(
                [
                    epoch,
                    train_weighted_loss,
                    train_acc,
                    selection_metrics[
                        "accuracy"
                    ],
                    selection_metrics[
                        "worst_group_accuracy"
                    ],
                    test_metrics[
                        "accuracy"
                    ],
                    test_metrics[
                        "worst_group_accuracy"
                    ],
                    float(
                        epoch_weights_np.mean()
                    ),
                    float(
                        epoch_weights_np.std()
                    ),
                ]
            )

            self._run_log(
                f"[Final classifier epoch {epoch:03d}] "
                f"weighted_train_loss={train_weighted_loss:.6f}, "
                f"train_acc={100.0 * train_acc:.2f}%, "
                f"mean_weight={epoch_weights_np.mean():.4f}, "
                f"selection_acc="
                f"{100.0 * selection_metrics['accuracy']:.2f}%, "
                f"selection_WGA="
                f"{100.0 * selection_metrics['worst_group_accuracy']:.2f}%, "
                f"test_acc={100.0 * test_metrics['accuracy']:.2f}%, "
                f"test_WGA="
                f"{100.0 * test_metrics['worst_group_accuracy']:.2f}%"
            )

            if selection_value > best_value:
                best_value = float(
                    selection_value
                )

                best_epoch = int(
                    epoch
                )

                best_metrics = {
                    "loss": float(
                        selection_metrics[
                            "loss"
                        ]
                    ),
                    "accuracy": float(
                        selection_metrics[
                            "accuracy"
                        ]
                    ),
                    "worst_group_accuracy": float(
                        selection_metrics[
                            "worst_group_accuracy"
                        ]
                    ),
                    "group_accuracy": {
                        int(k): float(v)
                        for k, v
                        in selection_metrics[
                            "group_accuracy"
                        ].items()
                    },
                }

                best_state = {
                    key: value.detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError(
                "Final classifier training produced no selected model."
            )

        model.load_state_dict(
            best_state
        )

        model.to(
            self.device
        )

        model.eval()

        selected_test_metrics = self._evaluate_classifier(
            model,
            test_dataset,
        )

        self.final_best_epoch = int(
            best_epoch
        )

        self.final_best_selection_metrics = (
            best_metrics
        )

        self.final_test_metrics = {
            "loss": float(
                selected_test_metrics[
                    "loss"
                ]
            ),
            "accuracy": float(
                selected_test_metrics[
                    "accuracy"
                ]
            ),
            "worst_group_accuracy": float(
                selected_test_metrics[
                    "worst_group_accuracy"
                ]
            ),
            "group_accuracy": {
                int(k): float(v)
                for k, v
                in selected_test_metrics[
                    "group_accuracy"
                ].items()
            },
        }

        plot_dir = os.path.join(
            output_dir,
            "plots",
            "training_curves",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        epochs = [
            row[0]
            for row in history
        ]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            epochs,
            [
                row[3]
                for row in history
            ],
            label="selection accuracy",
        )

        ax.plot(
            epochs,
            [
                row[4]
                for row in history
            ],
            label="selection WGA",
        )

        ax.plot(
            epochs,
            [
                row[5]
                for row in history
            ],
            label="test accuracy",
            alpha=0.75,
        )

        ax.plot(
            epochs,
            [
                row[6]
                for row in history
            ],
            label="test WGA",
            alpha=0.75,
        )

        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.5,
            label=f"selected epoch {best_epoch}",
        )

        ax.set_xlabel(
            "Final-classifier epoch"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_title(
            "Frozen-Rater final classifier performance"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                plot_dir,
                "final_classifier_accuracy_wga.png",
            ),
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            epochs,
            [
                row[1]
                for row in history
            ],
        )

        ax.set_xlabel(
            "Final-classifier epoch"
        )

        ax.set_ylabel(
            "Weighted training loss"
        )

        ax.set_title(
            "Frozen-Rater weighted classifier loss"
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.30,
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                plot_dir,
                "final_classifier_weighted_loss.png",
            ),
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        self._run_log(
            f"[Final classifier] selected epoch={best_epoch}; "
            f"selection_acc={100.0 * best_metrics['accuracy']:.2f}%, "
            f"selection_WGA="
            f"{100.0 * best_metrics['worst_group_accuracy']:.2f}%, "
            f"test_acc="
            f"{100.0 * self.final_test_metrics['accuracy']:.2f}%, "
            f"test_WGA="
            f"{100.0 * self.final_test_metrics['worst_group_accuracy']:.2f}%"
        )

        self.rater.eval()

        return model


    def _payload_group_summary_lines(
        self,
        split_name,
        payload,
    ):
        scores = torch.as_tensor(
            payload["scores"]
        ).float()

        weights = torch.as_tensor(
            payload["final_scores"]
        ).float()

        groups = torch.as_tensor(
            payload["groups"]
        ).long()

        lines = [
            (
                f"{split_name}: "
                f"n={len(scores)}, "
                f"raw_mean={scores.mean().item():.6f}, "
                f"raw_std={scores.std(unbiased=False).item():.6f}, "
                f"weight_mean={weights.mean().item():.6f}, "
                f"weight_std={weights.std(unbiased=False).item():.6f}"
            )
        ]

        for gid in torch.unique(
            groups
        ):
            mask = (
                groups == gid
            )

            lines.append(
                (
                    f"  g{int(gid)} "
                    f"({self._group_display_name(int(gid))}): "
                    f"n={int(mask.sum())}, "
                    f"raw_mean={scores[mask].mean().item():.6f}, "
                    f"raw_std="
                    f"{scores[mask].std(unbiased=False).item():.6f}, "
                    f"weight_mean={weights[mask].mean().item():.6f}, "
                    f"weight_std="
                    f"{weights[mask].std(unbiased=False).item():.6f}"
                )
            )

        return lines

    def _write_summary(
        self,
        output_dir,
        best_meta_loss,
        best_meta_step,
        split_payloads,
    ):
        summary_path = os.path.join(
            output_dir,
            "summary.txt",
        )

        lines = [
            "RATER EXPERIMENT SUMMARY",
            "=" * 80,
            f"ERM checkpoint: {self.erm_model_path}",
            f"ERM fingerprint: {self.erm_checkpoint_fingerprint}",
            f"ERM checkpoint epoch: {self.erm_checkpoint_epoch}",
            (
                f"representation: frozen Waterbirds-ERM "
                f"{self.config.backbone} embeddings"
            ),
            f"feature_dim: {self.feature_dim}",
            f"Rater capacity: {self.rater_capacity}",
            f"Rater transform: {self.weighting}",
            f"temperature: {self.temperature}",
            (
                "meta objective: outer classification "
                "cross-entropy ONLY"
            ),
            f"meta_steps: {self.meta_steps}",
            f"inner_steps: {self.inner_steps}",
            f"inner_models: {self.num_inner_models}",
            f"inner_init_noise_std: {self.inner_init_noise_std}",
            f"best meta step: {best_meta_step}",
            f"best outer classification CE: {best_meta_loss:.8f}",
            "",
            "FINAL FROZEN-RATER CLASSIFIER",
            "-" * 80,
            "initialization: exact saved ERM classifier head",
            "weighted train split: train_no_aug",
            "selection split: val_subset2 when split_val < 1",
            "test split: test",
            f"epochs: {self.final_epochs}",
            f"lr: {self.final_lr}",
            f"momentum: {self.final_momentum}",
            f"weight_decay: {self.final_weight_decay}",
        ]

        if hasattr(
            self,
            "final_best_epoch",
        ):
            selection = (
                self.final_best_selection_metrics
            )

            test_metrics = (
                self.final_test_metrics
            )

            lines.extend(
                [
                    f"selected epoch: {self.final_best_epoch}",
                    (
                        f"selection accuracy: "
                        f"{selection['accuracy']:.8f}"
                    ),
                    (
                        f"selection WGA: "
                        f"{selection['worst_group_accuracy']:.8f}"
                    ),
                    (
                        f"selection group accuracy: "
                        f"{selection['group_accuracy']}"
                    ),
                    (
                        f"test accuracy: "
                        f"{test_metrics['accuracy']:.8f}"
                    ),
                    (
                        f"test WGA: "
                        f"{test_metrics['worst_group_accuracy']:.8f}"
                    ),
                    (
                        f"test group accuracy: "
                        f"{test_metrics['group_accuracy']}"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "FINAL FROZEN-RATER RATE / WEIGHT SUMMARIES",
                "-" * 80,
            ]
        )

        for split_name, payload in split_payloads.items():
            lines.extend(
                self._payload_group_summary_lines(
                    split_name,
                    payload,
                )
            )

        lines.extend(
            [
                "",
                "SAVED OUTPUTS",
                "-" * 80,
                "plots/",
                "log.txt",
                "summary.txt",
                "final_rater.pt",
            ]
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(
                "\n".join(
                    lines
                )
                + "\n"
            )

        self.summary_path = (
            summary_path
        )

        return summary_path

    def _cleanup_minimal_output(
        self,
        output_dir,
    ):
        keep = {
            "plots",
            "log.txt",
            "summary.txt",
            "final_rater.pt",
            "embedding_cache",
        }

        for name in os.listdir(
            output_dir
        ):
            if name in keep:
                continue

            path = os.path.join(
                output_dir,
                name,
            )

            try:
                if os.path.isdir(
                    path
                ):
                    import shutil
                    shutil.rmtree(
                        path
                    )
                else:
                    os.remove(
                        path
                    )
            except FileNotFoundError:
                pass

    # ========================================================
    # Main training -- FROZEN RATER / FINAL CLASSIFIER ONLY
    # ========================================================

    def train(
        self,
        output_dir,
        split="train",
    ):
        """
        Classification-only Rater meta-training with minimal persistent output.

        Experiment directory keeps only:
            plots/
            log.txt
            summary.txt
            final_rater.pt

        ERM embeddings are cached separately in RATER_EMBEDDING_CACHE_ROOT.
        """
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        self.run_log_path = os.path.join(
            output_dir,
            "log.txt",
        )

        with open(
            self.run_log_path,
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(
                "Rater experiment log\n"
            )

        self._run_log(
            f"[Rater] Output directory: {output_dir}"
        )

        self._run_log(
            "[Rater] Minimal-output mode: plots/, log.txt, "
            "summary.txt, final_rater.pt only."
        )

        if self.precompute_embeddings_only:
            self._precompute_shared_embeddings(
                output_dir
            )

            with open(
                os.path.join(
                    output_dir,
                    "summary.txt",
                ),
                "w",
                encoding="utf-8",
            ) as fout:
                fout.write(
                    "Persistent ERM embedding cache prepared successfully.\n"
                )
                fout.write(
                    f"cache_root={self.embedding_cache_root}\n"
                )
                fout.write(
                    f"erm_checkpoint={self.erm_model_path}\n"
                )
                fout.write(
                    f"erm_fingerprint="
                    f"{self.erm_checkpoint_fingerprint}\n"
                )

            self._cleanup_minimal_output(
                output_dir
            )

            return

        cache_dir = self._resolve_embedding_cache_dir(
            output_dir
        )

        (
            inner_split,
            outer_split,
        ) = self._resolve_meta_splits(
            split
        )

        self._run_log(
            f"[Rater] Inner/meta-train split: {inner_split}"
        )

        self._run_log(
            f"[Rater] Outer/meta-loss split: {outer_split}"
        )

        self._run_log(
            f"[Rater embeddings] Shared cache: {cache_dir}"
        )

        train_dataset = self._extract_embedding_dataset(
            inner_split,
            cache_dir,
        )

        val_dataset = self._extract_embedding_dataset(
            outer_split,
            cache_dir,
        )

        test_dataset = self._extract_embedding_dataset(
            "test",
            cache_dir,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        train_iterator = iter(
            train_loader
        )

        val_iterator = iter(
            val_loader
        )

        self._initialize_inner_population()

        refresh_period = max(
            1,
            self.refresh_steps,
        )

        refresh_offsets = [
            (
                i * refresh_period
            )
            // self.num_inner_models
            for i in range(
                self.num_inner_models
            )
        ]

        best_meta_loss = float(
            "inf"
        )

        best_meta_step = None
        best_rater_state = None

        history = []
        rate_trajectory = []

        self._run_log(
            "[Rater] Starting classification-only bilevel optimization "
            "from the saved ERM classifier."
        )

        for meta_step in tqdm(
            range(
                1,
                self.meta_steps + 1,
            ),
            desc="Rater meta-training",
        ):
            if meta_step > 1:
                position = (
                    meta_step - 1
                ) % refresh_period

                for i, offset in enumerate(
                    refresh_offsets
                ):
                    if position == offset:
                        self.inner_models[
                            i
                        ] = self._new_inner_model(
                            model_index=i
                        )

            (
                total_meta_loss,
                outer_ce,
                _,
                train_iterator,
                val_iterator,
            ) = self._meta_step(
                train_iterator,
                val_iterator,
                train_loader,
                val_loader,
            )

            history.append(
                [
                    int(
                        meta_step
                    ),
                    float(
                        total_meta_loss
                    ),
                ]
            )

            if total_meta_loss < best_meta_loss:
                best_meta_loss = float(
                    total_meta_loss
                )

                best_meta_step = int(
                    meta_step
                )

                best_rater_state = {
                    key: value.detach()
                    .cpu()
                    .clone()
                    for key, value
                    in self.rater.state_dict().items()
                }

            should_eval = (
                meta_step == 1
                or meta_step % self.config.eval_freq == 0
                or meta_step == self.meta_steps
            )

            should_plot = (
                meta_step == 1
                or meta_step % self.plot_freq == 0
                or meta_step == self.meta_steps
            )

            if should_eval or should_plot:
                population_metrics = (
                    self._evaluate_inner_population(
                        val_dataset,
                        verbose=should_eval,
                    )
                )

                relationship = (
                    self._score_loss_relationship(
                        val_dataset,
                        models=self.inner_models,
                    )
                )

                scores_np = np.asarray(
                    relationship[
                        "scores"
                    ]
                )

                weights_np = np.asarray(
                    relationship[
                        "final_scores"
                    ]
                )

                groups_np = np.asarray(
                    relationship[
                        "groups"
                    ]
                )

                for gid in sorted(
                    np.unique(
                        groups_np
                    )
                ):
                    mask = (
                        groups_np == gid
                    )

                    rate_trajectory.append(
                        {
                            "meta_step": int(
                                meta_step
                            ),
                            "group": int(
                                gid
                            ),
                            "mean_raw_score": float(
                                scores_np[
                                    mask
                                ].mean()
                            ),
                            "mean_weight": float(
                                weights_np[
                                    mask
                                ].mean()
                            ),
                        }
                    )

                self._run_log(
                    f"[Rater step {meta_step:04d}] "
                    f"outer_ce={outer_ce:.6f}, "
                    f"mean_inner_acc="
                    f"{100.0 * population_metrics['mean_accuracy']:.2f}%, "
                    f"mean_inner_WGA="
                    f"{100.0 * population_metrics['mean_wga']:.2f}%, "
                    f"Spearman(score, loss)="
                    f"{relationship['spearman_score_vs_loss']:.4f}, "
                    f"Spearman(weight, loss)="
                    f"{relationship['spearman_final_score_vs_loss']:.4f}"
                )

                if should_plot:
                    self._save_all_score_diagnostics(
                        relationship=relationship,
                        output_dir=output_dir,
                        meta_step=meta_step,
                        split_name=outer_split,
                        tag_prefix="meta_step",
                    )

        if best_rater_state is None:
            raise RuntimeError(
                "Rater meta-training did not produce a valid state."
            )

        self.rater.load_state_dict(
            best_rater_state
        )

        self.rater.to(
            self.device
        )

        self.rater.eval()

        self._run_log(
            f"[Rater] Best outer classification CE="
            f"{best_meta_loss:.8f} at step {best_meta_step}."
        )

        final_rater_path = os.path.join(
            output_dir,
            "final_rater.pt",
        )

        self._save_rater_checkpoint(
            final_rater_path,
            best_meta_step,
            best_meta_loss,
            outer_ce=best_meta_loss,
            grad_mse=0.0,
        )

        self._save_meta_loss_plot(
            history,
            output_dir,
        )

        self._save_rate_trajectory_plots(
            rate_trajectory,
            output_dir,
        )

        split_datasets = {
            inner_split: train_dataset,
            outer_split: val_dataset,
            "test": test_dataset,
        }

        if self._has_ea_calibration_split():
            for extra_split in (
                "val_subset1",
                "val_subset2",
            ):
                if extra_split not in split_datasets:
                    split_datasets[
                        extra_split
                    ] = self._extract_embedding_dataset(
                        extra_split,
                        cache_dir,
                    )

        final_rater_payloads = {}

        for split_name, dataset in split_datasets.items():
            payload = self._compute_scores(
                dataset,
                model=None,
            )

            final_rater_payloads[
                split_name
            ] = payload

            self._save_final_rater_distribution_plots(
                payload=payload,
                output_dir=output_dir,
                split_name=split_name,
            )

            for line in self._rating_summary(
                payload
            ):
                self._run_log(
                    f"[Final Rater/{split_name}] {line}"
                )

        (
            calibration_split,
            selection_split,
        ) = self._resolve_final_classifier_splits()

        calibration_dataset = self._extract_embedding_dataset(
            calibration_split,
            cache_dir,
        )

        selection_dataset = self._extract_embedding_dataset(
            selection_split,
            cache_dir,
        )

        self.final_classifier = self._train_final_classifier(
            calibration_dataset,
            selection_dataset,
            test_dataset,
            output_dir,
            calibration_split=calibration_split,
            selection_split=selection_split,
            test_split="test",
        )

        for split_name, dataset in {
            calibration_split: calibration_dataset,
            selection_split: selection_dataset,
            "test": test_dataset,
        }.items():
            relationship = (
                self._classifier_score_loss_relationship(
                    dataset,
                    self.final_classifier,
                )
            )

            self._save_all_score_diagnostics(
                relationship=relationship,
                output_dir=output_dir,
                meta_step=self.meta_steps,
                split_name=(
                    f"{split_name}_final_classifier"
                ),
                tag_prefix="final",
            )

        self._write_summary(
            output_dir=output_dir,
            best_meta_loss=best_meta_loss,
            best_meta_step=best_meta_step,
            split_payloads=final_rater_payloads,
        )

        self._cleanup_minimal_output(
            output_dir
        )

        self._run_log(
            "[Rater] Minimal outputs finalized."
        )

    # ========================================================
    # Test / final evaluation
    # ========================================================

    def test(
        self,
        output_dir,
        split=("test",),
        result_path="",
    ):
        """
        Minimal final evaluation.

        Saves no tensor payloads, metric .pt files, CSVs, or final-classifier
        checkpoints. It only adds log messages and PNG diagnostics.
        """
        if self.precompute_embeddings_only:
            self._run_log(
                "[Rater embeddings] Precompute-only run finished."
            )

            self._cleanup_minimal_output(
                output_dir
            )

            return

        cache_dir = self._resolve_embedding_cache_dir(
            output_dir
        )

        if isinstance(
            split,
            str,
        ):
            split = [
                split
            ]

        for sp in split:
            dataset = self._extract_embedding_dataset(
                sp,
                cache_dir,
            )

            payload = self._compute_scores(
                dataset,
                model=self.final_classifier,
            )

            for line in self._rating_summary(
                payload
            ):
                self._run_log(
                    f"[Rater/{sp}] {line}"
                )

            if self.final_classifier is not None:
                final_metrics = (
                    self._evaluate_classifier(
                        self.final_classifier,
                        dataset,
                    )
                )

                group_str = ", ".join(
                    f"g{gid}="
                    f"{100.0 * acc:.2f}%"
                    for gid, acc
                    in final_metrics[
                        "group_accuracy"
                    ].items()
                )

                self._run_log(
                    f"[Rater FINAL {sp.upper()}] "
                    f"loss={final_metrics['loss']:.6f}, "
                    f"acc="
                    f"{100.0 * final_metrics['accuracy']:.2f}%, "
                    f"WGA="
                    f"{100.0 * final_metrics['worst_group_accuracy']:.2f}% "
                    f"({group_str})"
                )

                final_relationship = (
                    self._classifier_score_loss_relationship(
                        dataset,
                        self.final_classifier,
                    )
                )

                self._save_all_score_diagnostics(
                    relationship=final_relationship,
                    output_dir=output_dir,
                    meta_step=self.meta_steps,
                    split_name=(
                        f"{sp}_final_classifier"
                    ),
                    tag_prefix="test",
                )

        self._cleanup_minimal_output(
            output_dir
        )
