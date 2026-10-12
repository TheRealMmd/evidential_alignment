"""Meta-train a rater and export final_rater.pt for independent downstream use.

The existing per-group/overlaid histogram layout, scatter plots and box/ECDF
plots are preserved. Inner-model accuracy, WGA, mean CE and worst-group mean CE
are recorded on train/outer-validation/test (plus held-out validation if present).
Existing eval_freq, RATER_EVAL_FREQ, RATER_PLOT_FREQ and save_freq control cadence;
no new training settings or automatic sensitivity sweeps are introduced.

Metric/source data live in plots/data/, Grad-CAM images/arrays in plots/gradcam/,
and full population checkpoints in plots/checkpoints/. The existing notebook's
copy of plots/ preserves them. final_rater.pt remains a compact rater checkpoint.
Grad-CAM uses one fixed image per observed group for the selected rater and inner
models. The downstream classifier and its diagnostics belong to its notebook.

Offline plotting (no training or configuration changes):
  python algorithms/rater.py --runs RUN_SEED0 RUN_SEED1 --output MULTISEED_DIR
  python algorithms/rater.py --runs RUN_A RUN_B --output COMPARISON_DIR --compare calibration_size
  python algorithms/rater.py --runs RUN_A RUN_B --output COMPARISON_DIR --compare initial_wga
Comparisons describe observed inner-model outcomes across existing runs; they do
not create controlled sensitivity experiments or evaluate a downstream model.
"""

import os
import csv
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from contextlib import contextmanager

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

if __name__ == "__main__":
    # The plotting CLI needs no dataset registry, pretrained downloads, or timm.
    Algorithm = nn.Module

    def register_algorithm(name):
        return lambda cls: cls
else:
    from .algorithm import Algorithm
    from .register import register_algorithm
    from models.classifier import Classifier
    from utils import log


METRIC_FIELDS = ("accuracy", "worst_group_accuracy", "loss", "worst_group_loss")
METRIC_LABELS = ("Accuracy", "Worst-group accuracy", "Mean CE loss", "Worst-group mean CE loss")


def _json_value(value):
    """Strict JSON, including numpy/tensor scalars and explicit missing values."""
    if torch.is_tensor(value):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = f"{path}.tmp.{os.getpid()}"
    with open(temp, "w", encoding="utf-8") as fout:
        json.dump(_json_value(payload), fout, indent=2, sort_keys=True, allow_nan=False)
        fout.write("\n")
    os.replace(temp, path)


def _atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temp)
    os.replace(temp, path)


def _append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = f"{path}.tmp.{os.getpid()}"
    with open(temp, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


@contextmanager
def _preserve_rng():
    """Diagnostics and extra experiments must not alter training randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _diagnostic_path(output_dir, *parts):
    return os.path.join(output_dir, "plots", "data", *parts)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _metric_figure(rows, title, xlabel="Step"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, field, label in zip(axes.flat, METRIC_FIELDS, METRIC_LABELS):
        for split in sorted({r["split"] for r in rows}):
            values = sorted((r for r in rows if r["split"] == split),
                            key=lambda r: float(r["step"]))
            ax.plot([float(r["step"]) for r in values],
                    [float(r[field]) for r in values], marker=".", label=split)
        ax.set(xlabel=xlabel, ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ============================================================
# Embedding dataset
# ============================================================

class EmbeddingTensorDataset(Dataset):
    """Dataset of frozen backbone embeddings and Waterbirds metadata."""

    def __init__(self, embeddings, labels, groups, attrs, sample_indices=None):
        self.embeddings = embeddings.float()
        self.y_array = labels.long()
        self.group_array = groups.long()
        self.confounder_array = attrs.long()
        self.n_classes = int(torch.unique(self.y_array).numel())
        # Indices in the original, deterministic split/cache order.
        self.sample_indices = (torch.arange(len(labels)) if sample_indices is None
                               else torch.as_tensor(sample_indices).long())

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
# Inner classifier
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
    """Learn per-example weights through a group-blind bilevel objective.

    A frozen ERM backbone supplies embeddings. Persistent randomly initialized
    linear inner models optimize SUM_i weight_i * CE_i for a full training epoch
    before each rater update. Groups are used only for diagnostics. Training ends
    by exporting the selected rater; no downstream classifier is trained here.
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
            log(
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
        # parameters are retained for checkpoint compatibility. The persistent
        # inner classifiers are initialized independently from scratch.
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

        # Saved ERM head parameters; not used to initialize the inner population.
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

        log(
            f"[Rater] Loaded frozen Waterbirds ERM model from: "
            f"{self.erm_model_path}"
        )

        log(
            f"[Rater] ERM checkpoint epoch="
            f"{self.erm_checkpoint_epoch}, "
            f"sel_metric={self.erm_checkpoint_sel_metric}, "
            f"fingerprint={self.erm_checkpoint_fingerprint}"
        )

        if incompatible.missing_keys:
            log(
                f"[Rater] ERM load missing keys: "
                f"{incompatible.missing_keys}"
            )

        if incompatible.unexpected_keys:
            log(
                f"[Rater] ERM load unexpected keys: "
                f"{incompatible.unexpected_keys}"
            )

        log(
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
            os.environ.get(
                "RATER_META_STEPS",
                str(
                    getattr(
                        self.config,
                        "rater_meta_steps",
                        self.config.epoch,
                    )
                ),
            )
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
            os.environ.get(
                "RATER_INNER_LR",
                str(
                    getattr(
                        self.config,
                        "rater_inner_lr",
                        1e-2,
                    )
                ),
            )
        )
        self.outer_lr = float(
            os.environ.get(
                "RATER_OUTER_LR",
                str(
                    getattr(
                        self.config,
                        "rater_outer_lr",
                        3e-4,
                    )
                ),
            )
        )

        self.outer_loss_type = str(
            os.environ.get(
                "RATER_OUTER_LOSS",
                getattr(
                    self.config,
                    "rater_outer_loss",
                    "classwise_top_loss",
                ),
            )
        ).lower()
        valid_outer_losses = {
            "standard_ce",
            "worst_group_loss",
            "group_balanced_loss",
            "classwise_top_loss",
            "class_balanced",
            "worst_class_loss",
            "logsumexp",
        }
        if self.outer_loss_type not in valid_outer_losses:
            raise ValueError(
                "RATER_OUTER_LOSS must be one of: "
                + ", ".join(sorted(valid_outer_losses))
            )

        self.outer_top_loss_percent = float(
            os.environ.get(
                "RATER_OUTER_TOP_LOSS_PERCENT",
                str(
                    getattr(
                        self.config,
                        "rater_outer_top_loss_percent",
                        30.0,
                    )
                ),
            )
        )

        if not (0.0 < self.outer_top_loss_percent <= 100.0):
            raise ValueError(
                "RATER_OUTER_TOP_LOSS_PERCENT must be in (0, 100]."
            )

        self.outer_top_loss_fraction = (
            self.outer_top_loss_percent / 100.0
        )
        self.outer_lse_beta = float(
            os.environ.get(
                "RATER_OUTER_LSE_BETA",
                str(
                    getattr(
                        self.config,
                        "rater_outer_lse_beta",
                        1.0,
                    )
                ),
            )
        )
        if self.outer_lse_beta <= 0.0:
            raise ValueError("RATER_OUTER_LSE_BETA must be positive.")
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
        # Rater-rating transform for ALL samples in each batch.
        # Every sample receives a learned rating regardless of correctness.
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
            os.environ.get(
                "RATER_GRAD_CLIP",
                str(
                    getattr(
                        self.config,
                        "rater_grad_clip",
                        5.0,
                    )
                ),
            )
        )
        self.inner_reg_weight = float(
            getattr(self.config, "rater_inner_reg_weight", 0.0)
        )
        self.outer_reg_weight = 0.0
        self.score_reg_weight = 0.0

        # ----------------------------------------------------
        # Current Rater experiment
        # ----------------------------------------------------
        #
        # Inner models:
        #   two persistent, independently random Linear(feature_dim, C)
        #   classifiers. They are NOT initialized from the saved ERM head.
        #
        # Inner loss:
        #   every training sample receives a Rater-derived weight, and
        #   L_inner = SUM_i w_i * CE_i.
        #
        # Outer/Rater loss is configurable. Group labels are consumed only
        # when worst_group_loss or group_balanced_loss is selected.
        #
        # There is NO gradient-magnitude auxiliary loss and no additional
        # Rater regularizer in the meta objective.
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

        # Every training sample receives a Rater-derived rating.
        self.rate_all_samples = True

        self.rater_capacity = getattr(
            self.config, "rater_capacity", "medium"
        )

        # ----------------------------------------------------
        # Diagnostic plotting. By default, plots are saved at
        # the same frequency as --eval_freq. If config.py later
        # defines --rater_plot_freq, it will override eval_freq.
        # ----------------------------------------------------
        self.eval_freq = int(
            os.environ.get(
                "RATER_EVAL_FREQ",
                str(
                    max(
                        1,
                        int(
                            getattr(
                                self.config,
                                "eval_freq",
                                10,
                            )
                        ),
                    )
                ),
            )
        )

        self.plot_freq = int(
            os.environ.get(
                "RATER_PLOT_FREQ",
                str(
                    max(
                        1,
                        self.eval_freq,
                    )
                ),
            )
        )

        # Per-sample arrays are saved once in compressed NPZ snapshots below.
        # Leave the legacy scatter-only CSV exporter disabled to avoid duplication.
        self.save_plot_data = False

        self.save_freq = int(getattr(self.config, "save_freq", 0))
        if min(self.meta_steps, self.eval_freq, self.plot_freq) < 1:
            raise ValueError("Training length and evaluation/plot frequencies must be positive.")

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

        # Preserve the existing calibration-split behavior and legacy option.
        self.use_calibration_split = bool(
            getattr(self.config, "rater_use_calibration_final", True)
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

        log(
            f"[Rater] All-sample Rater transform = "
            f"{self.weighting}; temperature = {self.temperature}"
        )
        log(
            "[Rater] Every sample uses the Rater rating, "
            "regardless of classifier correctness."
        )
        log(
            "[Rater] Inner objective: L_inner = sum_i w_i * CE_i "
            "WITHOUT dividing by sum(w)."
        )
        uses_groups = self.outer_loss_type in {
            "worst_group_loss",
            "group_balanced_loss",
        }
        log(
            f"[Rater] Meta objective = {self.outer_loss_type}; "
            f"top-loss percent={self.outer_top_loss_percent:g}%; "
            f"LSE beta={self.outer_lse_beta:g}; "
            f"group labels used={uses_groups}."
        )
        log(
            f"[Rater] Inner population = {self.num_inner_models}; "
            "both inner classifiers are initialized independently from scratch."
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

        log(
            f"[Rater embeddings] Found shared cache: {cache_file}"
        )

        try:
            payload = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            log(
                f"[Rater embeddings] Cache load failed "
                f"({type(exc).__name__}: {exc}). "
                f"Regenerating split '{split}'."
            )
            return None

        if not self._cache_metadata_matches(payload, split):
            log(
                f"[Rater embeddings] Cache metadata mismatch for "
                f"split '{split}'. Regenerating it once."
            )
            return None

        log(
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

    @_preserve_rng()
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

        log(
            f"[Rater embeddings] Shared cache MISS for '{split}'. "
            f"Extracting it ONCE with frozen "
            f"{self.config.backbone}..."
        )

        # Cache order must be stable for per-sample trajectories and Grad-CAM IDs.
        loader = DataLoader(self.datasets[split], batch_size=self.config.batch_size,
                            shuffle=False, num_workers=self.config.num_workers)
        self.feature_model.to(self.device)
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

        log(
            f"[Rater embeddings] SAVED persistent shared "
            f"'{split}' embeddings ({len(labels)} samples):"
        )
        log(f"[Rater embeddings]   {cache_file}")
        log(
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

        log("[Rater embeddings] PRECOMPUTE-ONLY mode.")
        log(
            f"[Rater embeddings] Shared cache directory: {cache_dir}"
        )
        log(
            f"[Rater embeddings] Preparing splits: {available}"
        )

        for split in available:
            dataset = self._extract_embedding_dataset(
                split,
                cache_dir,
            )
            log(
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

        log(
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
        Fresh random Linear(feature_dim, n_classes) classifier.

        IMPORTANT: the saved ERM classifier head is NOT used here.
        The ERM checkpoint is used only to define the frozen backbone
        representation / cached embeddings.
        """
        del model_index

        return InnerLinearClassifier(
            input_dim=self.feature_dim,
            num_classes=self.n_classes,
        ).to(self.device)


    def _initialize_inner_population(self):
        if self.num_inner_models != 2:
            raise ValueError(
                "This experiment is defined with exactly two inner models."
            )

        self.inner_models = [
            self._new_inner_model(model_index=i)
            for i in range(2)
        ]

        with torch.no_grad():
            dw = (
                self.inner_models[0].linear.weight
                - self.inner_models[1].linear.weight
            ).norm().item()

            db = (
                self.inner_models[0].linear.bias
                - self.inner_models[1].linear.bias
            ).norm().item()

        if dw == 0.0 and db == 0.0:
            raise RuntimeError(
                "The two independently-created inner classifiers are identical."
            )

        log(
            f"[Rater inner init] two independent RANDOM classifiers; "
            f"||dW||={dw:.6f}, ||db||={db:.6f}. "
            "ERM classifier initialization is NOT used."
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

        The transform is applied over every sample in the current batch.
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
        Differentiable FULL-EPOCH Rater-weighted classifier optimization.

        On each batch, every sample receives a Rater rating.

        Inner objective:
            L_inner = SUM_i weight_i * CE_i

        There is NO division by sum(weights).
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

        for _ in range(self.inner_steps):
            batch, train_iterator = self._next_batch(
                train_iterator,
                train_loader,
            )

            z, y, _, _ = batch
            z = z.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

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

            weights, _, raw_scores = (
                self._all_sample_training_weights(
                    z=z,
                    y=y,
                    logits=logits,
                    raw_scores_all=None,
                )
            )

            inner_loss = (
                per_sample_loss * weights
            ).sum()

            if self.inner_reg_weight > 0:
                inner_loss = (
                    inner_loss
                    + self.inner_reg_weight
                    * (
                        fast_weight.pow(2).sum()
                        + fast_bias.pow(2).sum()
                    )
                )

            grad_w, grad_b = torch.autograd.grad(
                inner_loss,
                [fast_weight, fast_bias],
                create_graph=True,
            )

            fast_weight = (
                fast_weight
                - self.inner_lr * grad_w
            )
            fast_bias = (
                fast_bias
                - self.inner_lr * grad_b
            )

            last_raw_scores = raw_scores

        return (
            {
                "weight": fast_weight,
                "bias": fast_bias,
            },
            train_iterator,
            last_raw_scores,
            torch.zeros((), device=self.device),
        )

    # ========================================================
    # Class-wise top-loss outer objective + one bilevel/meta step
    # ========================================================

    def _classwise_top_loss_outer_objective(
        self,
        per_sample_losses,
        labels,
    ):
        """
        Class-balanced hard-tail outer objective using CLASS labels only.

        For each class c:
            1. collect outer CE losses with y == c;
            2. keep k_c = ceil(p * n_c) highest-loss examples;
            3. compute the mean of those selected losses.

        The final Rater loss is the UNWEIGHTED mean of the per-class tail
        means, so each class contributes equally even when class counts differ.

        No group/background annotation is read or used here.
        """
        if per_sample_losses.ndim != 1:
            raise ValueError(
                "per_sample_losses must be a 1-D tensor."
            )

        if labels.ndim != 1:
            labels = labels.reshape(-1)

        if per_sample_losses.shape[0] != labels.shape[0]:
            raise ValueError(
                "Loss and label tensors must have the same length."
            )

        class_tail_means = []
        stats = {}

        for class_id_tensor in torch.unique(labels):
            class_id = int(class_id_tensor.item())
            class_mask = labels.eq(class_id_tensor)
            class_losses = per_sample_losses[class_mask]
            n_class = int(class_losses.numel())

            if n_class == 0:
                continue

            k = max(
                1,
                int(
                    math.ceil(
                        self.outer_top_loss_fraction * n_class
                    )
                ),
            )
            k = min(k, n_class)

            top_losses = torch.topk(
                class_losses,
                k=k,
                largest=True,
                sorted=False,
            ).values

            class_tail_mean = top_losses.mean()
            class_tail_means.append(class_tail_mean)

            stats[class_id] = {
                "n_class": n_class,
                "k_selected": k,
                "tail_mean": float(
                    class_tail_mean.detach().item()
                ),
            }

        if not class_tail_means:
            raise RuntimeError(
                "Outer batch/split contained no labeled examples."
            )

        outer_loss = torch.stack(
            class_tail_means
        ).mean()

        return outer_loss, stats

    def _compute_outer_objective(
        self,
        per_sample_losses,
        labels,
        groups=None,
    ):
        """Compute the selected Rater meta objective and group/class stats."""
        if per_sample_losses.ndim != 1:
            raise ValueError("per_sample_losses must be a 1-D tensor.")
        labels = labels.reshape(-1)
        if per_sample_losses.shape[0] != labels.shape[0]:
            raise ValueError("Loss and label tensors must have the same length.")
        if per_sample_losses.numel() == 0:
            raise RuntimeError("The outer split contains no examples.")

        if self.outer_loss_type == "standard_ce":
            return per_sample_losses.mean(), {
                "sample_count": int(per_sample_losses.numel()),
                "mean_loss": float(per_sample_losses.detach().mean().item()),
            }

        if self.outer_loss_type in {
            "worst_group_loss",
            "group_balanced_loss",
        }:
            if groups is None:
                raise ValueError(
                    f"groups are required for outer_loss={self.outer_loss_type}."
                )
            groups = groups.reshape(-1)
            if groups.shape[0] != per_sample_losses.shape[0]:
                raise ValueError("Loss and group tensors must have the same length.")

            group_means = []
            stats = {}
            for group_id_tensor in torch.unique(groups):
                group_id = int(group_id_tensor.item())
                group_losses = per_sample_losses[groups.eq(group_id_tensor)]
                group_mean = group_losses.mean()
                group_means.append(group_mean)
                stats[group_id] = {
                    "n_group": int(group_losses.numel()),
                    "mean_loss": float(group_mean.detach().item()),
                }

            if not group_means:
                raise RuntimeError("The outer split contains no observed groups.")
            group_means_tensor = torch.stack(group_means)
            if self.outer_loss_type == "worst_group_loss":
                return group_means_tensor.max(), stats
            return group_means_tensor.mean(), stats

        if self.outer_loss_type == "classwise_top_loss":
            return self._classwise_top_loss_outer_objective(
                per_sample_losses,
                labels,
            )

        class_means = []
        stats = {}
        for class_id_tensor in torch.unique(labels):
            class_id = int(class_id_tensor.item())
            class_losses = per_sample_losses[labels.eq(class_id_tensor)]
            class_mean = class_losses.mean()
            class_means.append(class_mean)
            stats[class_id] = {
                "n_class": int(class_losses.numel()),
                "mean_loss": float(class_mean.detach().item()),
            }

        if self.outer_loss_type == "class_balanced":
            # Give each observed target class equal contribution.
            outer_loss = torch.stack(class_means).mean()
        elif self.outer_loss_type == "worst_class_loss":
            # Worst class is defined by its mean per-example CE.
            outer_loss = torch.stack(class_means).max()
        elif self.outer_loss_type == "logsumexp":
            # Normalized log-mean-exp over examples, implemented stably.
            beta = self.outer_lse_beta
            outer_loss = (
                torch.logsumexp(beta * per_sample_losses, dim=0)
                - math.log(per_sample_losses.numel())
            ) / beta
        else:
            # The type is validated during initialization.
            raise RuntimeError(
                f"Unhandled outer loss type: {self.outer_loss_type}"
            )

        return outer_loss, stats

    def _meta_step(
        self,
        train_iterator,
        val_iterator,
        train_loader,
        val_loader,
    ):
        """
        One Rater update with the configured outer objective.

        BOTH persistent inner models first complete self.inner_steps batches.
        train() sets self.inner_steps = len(train_loader), so each inner model
        sees the entire train_no_aug set once before the Rater update.

        After an inner model finishes its full training pass, its outer
        per-example CE losses and group ids are collected over the ENTIRE
        outer split, then aggregated using `self.outer_loss_type`.

        For memory efficiency, each inner model's outer loss / num_models is
        backpropagated immediately. The Rater optimizer step happens only after
        both inner models have completed their full inner epoch and outer loss.
        """
        del val_iterator

        self.rater.train()
        self.outer_optimizer.zero_grad()

        outer_values = []
        committed_fast_params = []
        per_model_outer_stats = []

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

            # Collect outer per-example losses and group ids over the entire
            # outer split. Group ids affect the objective only in group modes.
            all_outer_losses = []
            all_outer_labels = []
            all_outer_groups = []

            for z_outer, y_outer, group_outer, _ in val_loader:
                z_outer = z_outer.to(
                    self.device,
                    non_blocking=True,
                )
                y_outer = y_outer.to(
                    self.device,
                    non_blocking=True,
                )

                outer_logits = F.linear(
                    z_outer,
                    fast_params["weight"],
                    fast_params["bias"],
                )

                per_outer_loss = F.cross_entropy(
                    outer_logits,
                    y_outer,
                    reduction="none",
                )

                all_outer_losses.append(per_outer_loss)
                all_outer_labels.append(y_outer)
                all_outer_groups.append(
                    group_outer.to(self.device, non_blocking=True).reshape(-1)
                )

            per_outer_loss = torch.cat(
                all_outer_losses,
                dim=0,
            )
            y_outer_all = torch.cat(
                all_outer_labels,
                dim=0,
            )
            group_outer_all = torch.cat(all_outer_groups, dim=0)

            outer_loss, outer_stats = (
                self._compute_outer_objective(
                    per_sample_losses=per_outer_loss,
                    labels=y_outer_all,
                    groups=group_outer_all,
                )
            )

            (
                outer_loss / self.num_inner_models
            ).backward()

            outer_values.append(
                float(outer_loss.detach().item())
            )
            per_model_outer_stats.append(outer_stats)

            committed_fast_params.append(
                {
                    "weight": fast_params["weight"].detach(),
                    "bias": fast_params["bias"].detach(),
                }
            )

        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.rater.parameters()):
            raise RuntimeError("Nonfinite rater gradient; checkpoint update aborted.")
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.rater.parameters(),
                self.grad_clip,
            )

        self.outer_optimizer.step()

        with torch.no_grad():
            for inner_model, fast_params in zip(
                self.inner_models,
                committed_fast_params,
            ):
                inner_model.linear.weight.copy_(
                    fast_params["weight"]
                )
                inner_model.linear.bias.copy_(
                    fast_params["bias"]
                )

        outer_objective = float(
            np.mean(outer_values)
        )

        self._last_outer_stats = per_model_outer_stats

        return (
            outer_objective,
            outer_objective,
            0.0,
            train_iterator,
            iter(val_loader),
        )

    # ========================================================
    # Evaluation helpers
    # ========================================================

    @_preserve_rng()
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
        group_loss_sum = {}

        for z, y, g, _ in loader:
            z = z.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)

            logits = model(z)
            loss = F.cross_entropy(
                logits,
                y,
                reduction="none",
            )
            pred = logits.argmax(dim=1)

            total_loss += loss.sum().item()
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
                group_loss_sum[gid] = group_loss_sum.get(gid, 0.0) + loss[mask].sum().item()

        if total_examples == 0:
            raise RuntimeError("Cannot evaluate an empty dataset.")

        group_accuracy = {
            gid: group_correct[gid] / group_total[gid]
            for gid in sorted(group_total)
            if group_total[gid] > 0
        }
        group_loss = {gid: group_loss_sum[gid] / group_total[gid]
                      for gid in sorted(group_total)}

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
            "group_accuracy": group_accuracy,
            "group_loss": group_loss,
            "group_count": group_total,
            "n_samples": total_examples,
            "worst_group_loss": max(group_loss.values()),
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
                log(
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
            log(
                f"[Inner population] mean_loss={mean_loss:.6f}, "
                f"mean_acc={100.0 * mean_acc:.2f}%, "
                f"mean_WGA={100.0 * mean_wga:.2f}%, "
                f"best_acc={100.0 * best_acc:.2f}%, "
                f"best_WGA={100.0 * best_wga:.2f}%"
            )

        return {
            "models": metrics,
            "mean_loss": mean_loss,
            "mean_worst_group_loss": float(np.mean([m["worst_group_loss"] for m in metrics])),
            "mean_accuracy": mean_acc,
            "mean_wga": mean_wga,
            "best_accuracy": best_acc,
            "best_wga": best_wga,
        }

    @_preserve_rng()
    @torch.no_grad()
    def _score_loss_relationship(self, dataset, models=None):
        """
        Diagnostic relationship between raw Rater scores, effective
        all-sample Rater weights, classifier loss, and correctness.

        Every sample receives a Rater rating for each inner model.

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

        log(
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
        Save ONE 5-panel histogram figure.

        Layout for Waterbirds:
            left column, top -> bottom:
                Group 0 histogram
                Group 1 histogram
                Group 2 histogram
                Group 3 histogram

            right column (spans all 4 rows):
                all four groups overlaid in one larger histogram.

        Group annotations are used ONLY for this diagnostic visualization.
        They are NOT used by the Rater outer/meta loss.

        A common set of bins is shared by all five panels. Density
        normalization is used so group shapes remain comparable despite
        Waterbirds group-size imbalance.
        """
        if score_kind == "raw":
            values = np.asarray(relationship["scores"])
            folder = "raw_score_histogram"
            xlabel = "Raw Rater score"
            title_name = "Raw Rater score distribution"
            filename_kind = "raw_score_hist"
        elif score_kind == "final":
            values = np.asarray(relationship["final_scores"])
            folder = f"final_score_histogram_{self.weighting}"
            xlabel = f"All-sample {self.weighting} Rater weight"
            title_name = (
                f"All-sample {self.weighting} Rater-weight distribution"
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

        if len(values) == 0:
            return None

        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return None

        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.05, 1e-6)
            bins = np.linspace(vmin - eps, vmax + eps, 20)
        else:
            bins = np.linspace(vmin, vmax, 41)

        # Waterbirds has exactly four group IDs 0,1,2,3. Keep the requested
        # four fixed left panels even if a diagnostic split happens to have
        # no sample from one group.
        if self.config.dataset == "waterbirds":
            group_ids = [0, 1, 2, 3]
        else:
            group_ids = list(sorted(np.unique(groups)))[:4]
            while len(group_ids) < 4:
                group_ids.append(None)

        fig = plt.figure(
            figsize=(16, 12)
        )
        grid = fig.add_gridspec(
            nrows=4,
            ncols=2,
            width_ratios=[1.0, 2.15],
            hspace=0.50,
            wspace=0.28,
        )

        # Four separate histograms on the left.
        for row, gid in enumerate(group_ids):
            ax = fig.add_subplot(
                grid[row, 0]
            )

            if gid is None:
                ax.axis("off")
                continue

            mask = groups == gid
            group_values = values[mask]

            if len(group_values) > 0:
                ax.hist(
                    group_values,
                    bins=bins,
                    density=True,
                    alpha=0.80,
                )

            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylabel("Density")
            ax.set_title(
                f"{self._group_display_name(gid)} (n={int(mask.sum())})",
                fontsize=10,
            )
            ax.grid(
                True,
                linestyle=":",
                alpha=0.25,
            )

            if row == 3:
                ax.set_xlabel(xlabel)

        # Large all-group overlay on the right.
        ax_all = fig.add_subplot(
            grid[:, 1]
        )

        for gid in [g for g in group_ids if g is not None]:
            mask = groups == gid
            group_values = values[mask]

            if len(group_values) == 0:
                continue

            ax_all.hist(
                group_values,
                bins=bins,
                density=True,
                alpha=0.38,
                label=(
                    f"{self._group_display_name(gid)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax_all.set_xlim(bins[0], bins[-1])
        ax_all.set_xlabel(xlabel)
        ax_all.set_ylabel("Density")
        ax_all.set_title(
            "All groups overlaid",
            fontsize=13,
        )
        ax_all.grid(
            True,
            linestyle=":",
            alpha=0.25,
        )
        ax_all.legend(
            loc="best",
            fontsize=9,
        )

        fig.suptitle(
            f"{title_name} | {split_name} | step {meta_step}",
            fontsize=15,
        )
        fig.subplots_adjust(
            top=0.94
        )

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

        log(
            f"[Rater plot] saved 5-panel {score_kind}-score histogram: "
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

        FINAL SCORE means the effective all-sample Rater weight:
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
            "All-sample Rater weight"
        )
        ax.set_title(
            f"All-sample Rater weight vs inner-model loss "
            f"({self.weighting} on every sample) | "
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

        log(
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

    def _start_artifacts(self, output_dir, split_names, datasets):
        """Keep diagnostic data inside the directory copied by the training notebook."""
        data_dir = _diagnostic_path(output_dir)
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(os.path.join(data_dir, "metrics.csv")):
            raise ValueError(f"Use a fresh output directory; metrics already exist in {data_dir}")
        self._run_started = time.monotonic()
        self._metric_rows = []
        self._group_rows = []
        self._run_metadata = {
            "schema_version": 2, "seed": int(self.config.seed),
            "config": vars(self.config),
            "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                         "matplotlib": matplotlib.__version__},
            "splits": split_names,
            "split_sizes": {k: len(v) for k, v in datasets.items()},
            "settings": {
                "dataset": self.config.dataset, "backbone": self.config.backbone,
                "resolution": getattr(self.config, "resolution", 224),
                "split_train": getattr(self.config, "split_train", 1.0),
                "split_val": getattr(self.config, "split_val", 1.0),
                "num_classes": self.n_classes,
                "erm_fingerprint": self.erm_checkpoint_fingerprint,
                "feature_dim": self.feature_dim, "batch_size": self.config.batch_size,
                "meta_steps": self.meta_steps, "num_inner_models": self.num_inner_models,
                "inner_lr": self.inner_lr, "outer_lr": self.outer_lr,
                "inner_reg_weight": self.inner_reg_weight, "grad_clip": self.grad_clip,
                "capacity": self.rater_capacity, "weighting": self.weighting,
                "temperature": self.temperature, "outer_loss": self.outer_loss_type,
                "outer_top_loss_percent": self.outer_top_loss_percent,
                "outer_lse_beta": self.outer_lse_beta,
                "calibration_size": len(datasets["val"]),
                "inner_initialization": "independent_random_linear",
            },
            "metric_definition": {
                "accuracy": "fraction correct",
                "worst_group_accuracy": "minimum accuracy over observed groups",
                "loss": "unweighted sample-mean cross entropy",
                "worst_group_loss": "maximum mean cross entropy over observed groups",
                "diagnostic_weights": "eval-mode rater, fixed ordered batches; batch-relative",
                "sample_index": "row in the original deterministic split/cache",
                "val": "outer calibration split used for the meta objective",
                "heldout_val": "reserved validation subset, diagnostic-only when available",
                "test_usage": "diagnostics only; never checkpoint selection",
            },
        }
        _write_json(os.path.join(data_dir, "run_config.json"), self._run_metadata)
        self._gradcam_sample_indices = {
            split_names[role]: dataset.sample_indices.numpy() for role, dataset in datasets.items()
        }
        self.meta_log_path = os.path.join(output_dir, "log.txt")
        self._meta_log("Rater meta-learning diagnostics")
        for role, dataset in datasets.items():
            _write_json(os.path.join(data_dir, "sample_indices", f"{role}.json"),
                        {"split": split_names[role], "indices": dataset.sample_indices})

    def _record_metrics(self, output_dir, step, model_name, split_name,
                        role, metrics, phase="trajectory"):
        row = {
            "seed": int(self.config.seed), "stage": "meta", "phase": phase,
            "step": int(step), "elapsed_seconds": time.monotonic() - self._run_started,
            "model": model_name, "split": split_name, "role": role,
            "n_samples": metrics["n_samples"],
            **{key: metrics[key] for key in METRIC_FIELDS},
        }
        self._metric_rows.append(row)
        _append_csv(_diagnostic_path(output_dir, "metrics.csv"), [row])
        group_rows = [
            {**{k: row[k] for k in ("seed", "stage", "phase", "step", "model", "split", "role")},
             "group": gid, "n_samples": count,
             "accuracy": metrics["group_accuracy"][gid],
             "loss": metrics["group_loss"][gid]}
            for gid, count in sorted(metrics["group_count"].items())
        ]
        _append_csv(_diagnostic_path(output_dir, "group_metrics.csv"), group_rows)
        self._meta_log(
            f"[meta {phase} {step:04d}] {model_name} {role}={split_name}: "
            + ", ".join(f"{k}={metrics[k]:.6f}" for k in METRIC_FIELDS)
        )

    def _evaluate_inner_diagnostics(self, output_dir, step, datasets, split_names,
                                    phase="trajectory"):
        results = {}
        with _preserve_rng():
            for i, model in enumerate(self.inner_models):
                name = f"inner_{i}"
                results[name] = {}
                for role, dataset in datasets.items():
                    result = self._evaluate_classifier(model, dataset)
                    results[name][role] = result
                    self._record_metrics(output_dir, step, name, split_names[role], role, result, phase)
        return results

    def _save_metric_curves(self, output_dir):
        plot_dir = os.path.join(output_dir, "plots", "meta_metrics")
        os.makedirs(plot_dir, exist_ok=True)
        rows = [r for r in self._metric_rows if r["phase"] == "trajectory"]
        for name in sorted({r["model"] for r in rows}):
            fig = _metric_figure([r for r in rows if r["model"] == name],
                                 f"Meta training: {name}", "Meta step")
            fig.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=160)
            plt.close(fig)

    def _score_snapshot(self, output_dir, step, role, split_name, dataset, phase="trajectory"):
        """Reuse the existing histogram/scatter/box/ECDF plots; save their inputs once."""
        relationship = self._score_loss_relationship(dataset)
        prefix = "meta_step" if phase == "trajectory" else "FINAL_BEST_RATER"
        self._save_all_score_diagnostics(relationship, output_dir, step, split_name, prefix)
        self._save_meta_rate_box_ecdf(relationship, output_dir,
                                     f"{prefix}_{step:06d}", split_name)
        data_dir = _diagnostic_path(output_dir, "score_snapshots")
        os.makedirs(data_dir, exist_ok=True)
        arrays = {k: v for k, v in relationship.items() if isinstance(v, np.ndarray)}
        arrays.update(sample_indices=dataset.sample_indices.numpy(),
                      batch_indices=np.arange(len(dataset)) // self.config.batch_size)
        np.savez_compressed(os.path.join(data_dir, f"{phase}_{step:06d}_{split_name}.npz"), **arrays)
        correlations = {k: v for k, v in relationship.items()
                        if k.startswith(("spearman", "pearson"))}
        _append_csv(_diagnostic_path(output_dir, "score_correlations.csv"), [
            {"seed": self.config.seed, "stage": "meta", "phase": phase, "step": step,
             "split": split_name, "role": role, **correlations}
        ])
        groups, weights = relationship["groups"], relationship["final_scores"]
        rows = []
        for gid in sorted(np.unique(groups)):
            mask = groups == gid
            w, raw = weights[mask], relationship["scores"][mask]
            rows.append({
                "seed": self.config.seed, "stage": "meta", "phase": phase,
                "step": step, "split": split_name, "role": role, "group": int(gid),
                "n_samples": int(mask.sum()), "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std()), "weight_mean": float(w.mean()),
                "weight_std": float(w.std()), "weight_q25": float(np.quantile(w, .25)),
                "weight_median": float(np.median(w)), "weight_q75": float(np.quantile(w, .75)),
                "weight_mass_share": float(w.sum() / max(float(weights.sum()), 1e-12)),
            })
        self._group_rows.extend(rows)
        _append_csv(_diagnostic_path(output_dir, "group_statistics.csv"), rows)
        return relationship

    def _checkpoint_payload(self, step):
        return {
            "schema_version": 2, "stage": "meta", "step": step,
            "rater_sd": _cpu_state(self.rater),
            "inner_model_sds": [_cpu_state(m) for m in self.inner_models],
            "outer_optimizer_sd": copy.deepcopy(self.outer_optimizer.state_dict()),
            "rng_state": _rng_state(), "run_metadata": self._run_metadata,
            "feature_dim": self.feature_dim, "num_classes": self.n_classes,
            "erm_model_path": self.erm_model_path,
            "erm_checkpoint_fingerprint": self.erm_checkpoint_fingerprint,
            "rater_capacity": self.rater_capacity, "rater_weighting": self.weighting,
            "rater_temperature": self.temperature,
        }

    @_preserve_rng()
    @torch.no_grad()
    def _compute_scores(self, dataset, model=None):
        """
        Save raw Rater scores and, when a classifier is supplied, the
        effective all-sample Rater weights for that classifier.
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

    # ========================================================
    # Checkpoints
    # ========================================================

    def _load_rater_checkpoint(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Rater checkpoint does not exist: {path}")

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_checkpoint_representation(checkpoint)

        if "rater_sd" in checkpoint:
            self.rater.load_state_dict(checkpoint["rater_sd"])
        else:
            # Also permit loading a raw rater state_dict.
            self.rater.load_state_dict(checkpoint)

        log(f"[Rater] Loaded rater weights from {path} (warm start, not a resumed optimizer run)")

    def _validate_checkpoint_representation(self, checkpoint):
        fingerprint = checkpoint.get("erm_checkpoint_fingerprint", checkpoint.get("erm_fingerprint"))
        if fingerprint is not None and fingerprint != self.erm_checkpoint_fingerprint:
            raise ValueError("Checkpoint was trained on a different frozen ERM representation.")
        if checkpoint.get("feature_dim", self.feature_dim) != self.feature_dim:
            raise ValueError("Checkpoint feature dimension does not match the backbone.")

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

        The reserved val_subset2 remains diagnostic-only during meta-training.

        Legacy mode (--split_val == 1):
            inner/meta-train = train_no_aug
            outer/meta-val   = val
        """

        if split in {"train", "train_no_aug"}:
            inner_split = "train_no_aug"

            if (
                self.use_calibration_split
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
                self.use_calibration_split
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


    # ========================================================
    # Image-space diagnostics for the rater and inner models
    # ========================================================


    def _save_gradcam(self, output_dir, tag, splits, heads, samples_per_group=1):
        """Image-space Grad-CAM through the frozen backbone and embedding heads.

        Rater target: raw scalar score. Classifier target: predicted-class logit.
        CPU execution keeps the backbone out of GPU memory during meta unrolling.
        """
        if samples_per_group == 0:
            return
        plot_dir = os.path.join(output_dir, "plots", "gradcam", tag)
        os.makedirs(plot_dir, exist_ok=True)
        if self.feature_model is None:
            _write_json(os.path.join(plot_dir, "status.json"),
                        {"status": "skipped", "reason": "Image backbone unavailable."})
            return
        backbone = self.feature_model.backbone
        layer = getattr(backbone, "layer4", None)
        if layer is None:
            _write_json(os.path.join(plot_dir, "status.json"),
                        {"status": "skipped", "reason": "Backbone has no spatial layer4."})
            return
        previous_device = next(backbone.parameters()).device
        previous_mode = backbone.training
        backbone.cpu().eval()
        cpu_heads = {name: copy.deepcopy(head).cpu().eval() for name, head in heads.items()}
        activations = []
        hook = layer.register_forward_hook(lambda module, args, output: activations.append(output))
        rows = []
        try:
            with _preserve_rng(), torch.enable_grad():
                for split_name in dict.fromkeys(splits):
                    dataset = self.datasets.get(split_name)
                    if dataset is None or not hasattr(dataset, "group_array"):
                        rows.append({"split": split_name, "status": "skipped: no image/group metadata"})
                        continue
                    groups = np.asarray(dataset.group_array)
                    eligible = getattr(self, "_gradcam_sample_indices", {}).get(
                        split_name, np.arange(len(dataset)))
                    indexes = [int(i) for gid in sorted(np.unique(groups[eligible]))
                               for i in eligible[groups[eligible] == gid][:samples_per_group]]
                    for index in indexes:
                        x, y, group, _ = dataset[index]
                        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[0] != 3:
                            rows.append({"split": split_name, "sample_index": index,
                                         "status": "skipped: expected RGB image tensor"})
                            continue
                        x = x.detach().cpu().unsqueeze(0).requires_grad_(True)
                        activations.clear()
                        embedding = backbone(x)
                        activation = activations[-1]
                        # Undo the actual dataset's Normalize transform for faithful overlays.
                        display = x.detach()[0].clone()
                        transforms = list(getattr(getattr(dataset, "transform", None), "transforms", []))
                        for transform in reversed(transforms):
                            if hasattr(transform, "mean") and hasattr(transform, "std"):
                                display = display * torch.as_tensor(transform.std)[:, None, None]
                                display = display + torch.as_tensor(transform.mean)[:, None, None]
                                break
                        display = display.clamp(0, 1).permute(1, 2, 0).numpy()
                        fig, axes = plt.subplots(1, len(cpu_heads) + 1,
                                                 figsize=(4 * (len(cpu_heads) + 1), 4), squeeze=False)
                        axes[0, 0].imshow(display)
                        axes[0, 0].set_title(f"{split_name}[{index}]\ny={int(y)}, group={int(group)}")
                        saved = {"image": display}
                        targets = {}
                        for column, (name, head) in enumerate(cpu_heads.items(), 1):
                            value = head(embedding)
                            target_class = None if name == "rater" else int(value.argmax(dim=1).item())
                            target = value.reshape(-1)[0] if target_class is None else value[0, target_class]
                            gradients = torch.autograd.grad(target, activation, retain_graph=True)[0]
                            cam = (gradients.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1)
                            cam = F.relu(cam)
                            cam = F.interpolate(cam[:, None], size=x.shape[-2:],
                                                mode="bilinear", align_corners=False)[0, 0]
                            cam = cam.detach()
                            cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-12)
                            cam = cam.numpy()
                            saved[name] = cam
                            targets[name] = {"class": target_class, "value": float(target.detach()),
                                             "target": "raw_score" if name == "rater" else "predicted_logit"}
                            axes[0, column].imshow(display)
                            axes[0, column].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                            axes[0, column].set_title(name + (" (raw score)" if name == "rater"
                                                           else f" (class {target_class})"))
                        for ax in axes.flat:
                            ax.axis("off")
                        fig.tight_layout()
                        stem = f"{split_name}_g{int(group)}_sample{index:06d}"
                        fig.savefig(os.path.join(plot_dir, stem + ".png"), dpi=140)
                        plt.close(fig)
                        np.savez_compressed(os.path.join(plot_dir, stem + ".npz"), **saved)
                        filenames = getattr(dataset, "filename_array", None)
                        rows.append({"split": split_name, "sample_index": index,
                                     "filename": str(filenames[index]) if filenames is not None else None,
                                     "label": int(y), "group": int(group), "targets": targets,
                                     "status": "saved", "artifact": stem})
        finally:
            hook.remove()
            activations.clear()
            backbone.to(previous_device).train(previous_mode)
        _write_json(os.path.join(plot_dir, "samples.json"), rows)

    def _meta_log(self, message):
        path = getattr(self, "meta_log_path", None)
        before = os.path.getsize(path) if path and os.path.exists(path) else 0
        log(message)
        # utils.log may already write to this run's log.txt (main.py does this).
        # Direct library use still gets a log without duplicating those entries.
        after = os.path.getsize(path) if path and os.path.exists(path) else 0
        if path and after == before:
            with open(path, "a", encoding="utf-8") as fout:
                fout.write(str(message) + "\n")

    def _group_name_meta(self, gid):
        names = {
            0: "g0: landbird / land",
            1: "g1: landbird / water",
            2: "g2: waterbird / land",
            3: "g3: waterbird / water",
        }

        return names.get(
            int(gid),
            f"g{int(gid)}",
        )

    def _save_meta_rate_box_ecdf(
        self,
        relationship,
        output_dir,
        tag,
        split_name,
    ):
        plot_dir = os.path.join(
            output_dir,
            "plots",
            "meta_rate_distributions",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        groups = np.asarray(
            relationship["groups"]
        )

        for key, ylabel, suffix in [
            (
                "scores",
                "Raw Rater score",
                "raw",
            ),
            (
                "final_scores",
                f"Effective {self.weighting} weight",
                "weight",
            ),
        ]:
            values = np.asarray(
                relationship[key]
            )

            gids = sorted(
                np.unique(groups)
            )

            data = [
                values[groups == gid]
                for gid in gids
            ]

            labels = [
                self._group_name_meta(gid)
                for gid in gids
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

            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{ylabel} by group | {split_name} | {tag}"
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
            fig.savefig(
                os.path.join(
                    plot_dir,
                    f"{tag}_{split_name}_{suffix}_box.png",
                ),
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            for gid in gids:
                vals = np.sort(
                    values[groups == gid]
                )

                if len(vals) == 0:
                    continue

                yy = np.arange(
                    1,
                    len(vals) + 1,
                    dtype=np.float64,
                ) / len(vals)

                ax.step(
                    vals,
                    yy,
                    where="post",
                    linewidth=2,
                    label=self._group_name_meta(gid),
                )

            ax.set_xlabel(ylabel)
            ax.set_ylabel("Empirical CDF")
            ax.set_title(
                f"{ylabel} ECDF by group | {split_name} | {tag}"
            )
            ax.grid(
                True,
                linestyle=":",
                alpha=0.30,
            )
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    plot_dir,
                    f"{tag}_{split_name}_{suffix}_ecdf.png",
                ),
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _save_meta_training_curves(
        self,
        history,
        group_trajectory,
        output_dir,
    ):
        plot_dir = os.path.join(
            output_dir,
            "plots",
            "meta_training_curves",
        )

        os.makedirs(
            plot_dir,
            exist_ok=True,
        )

        if history:
            steps = np.asarray(
                [row["step"] for row in history]
            )
            outer_loss = np.asarray(
                [row["outer_loss"] for row in history]
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )
            ax.plot(
                steps,
                outer_loss,
                linewidth=1.8,
            )
            ax.set_xlabel("Meta step")
            ax.set_ylabel(f"{self.outer_loss_type} outer CE")
            ax.set_title(
                f"Rater meta objective: {self.outer_loss_type}"
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
                    "outer_meta_loss.png",
                ),
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

        trajectory = [r for r in group_trajectory
                      if r.get("phase", "trajectory") == "trajectory"]
        if not trajectory:
            return
        splits = sorted({r.get("split", "outer") for r in trajectory})
        for field, ylabel, filename in [
            ("raw_mean", "Mean raw Rater score", "group_mean_raw_score.png"),
            ("weight_mean", f"Mean {self.weighting} weight", "group_mean_weight.png"),
        ]:
            columns = min(2, len(splits))
            fig, axes = plt.subplots(math.ceil(len(splits) / columns), columns,
                                     figsize=(7 * columns, 4 * math.ceil(len(splits) / columns)),
                                     squeeze=False)
            for ax, split_name in zip(axes.flat, splits):
                split_rows = [r for r in trajectory if r.get("split", "outer") == split_name]
                for gid in sorted({int(r["group"]) for r in split_rows}):
                    rows = sorted((r for r in split_rows if int(r["group"]) == gid),
                                  key=lambda r: int(r["step"]))
                    ax.plot([r["step"] for r in rows], [r[field] for r in rows],
                            marker="o", markersize=3, label=self._group_name_meta(gid))
                ax.set(xlabel="Meta step", ylabel=ylabel, title=split_name)
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(splits):]:
                ax.set_visible(False)
            fig.suptitle(f"{ylabel} by group during meta training")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, filename), dpi=160, bbox_inches="tight")
            plt.close(fig)

    def train(self, output_dir, split="train"):
        """Meta-train only the rater; keep the notebook's training protocol intact."""
        os.makedirs(output_dir, exist_ok=True)
        if self.precompute_embeddings_only:
            self._precompute_shared_embeddings(output_dir)
            return
        inner_split, outer_split = self._resolve_meta_splits(split)
        split_names = {"train": inner_split, "val": outer_split, "test": "test"}
        if self._has_ea_calibration_split() and "val_subset2" not in split_names.values():
            split_names["heldout_val"] = "val_subset2"
        cache_dir = self._resolve_embedding_cache_dir(output_dir)
        by_split = {name: self._extract_embedding_dataset(name, cache_dir)
                    for name in dict.fromkeys(split_names.values())}
        datasets = {role: by_split[name] for role, name in split_names.items()}
        if any(len(dataset) == 0 for dataset in datasets.values()):
            raise ValueError("Cannot train or evaluate the rater on an empty split.")
        # Keep image-space diagnostics available without occupying meta-unroll GPU RAM.
        if self.feature_model is not None:
            self.feature_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._start_artifacts(output_dir, split_names, datasets)
        train_loader = DataLoader(datasets["train"], batch_size=self.config.batch_size,
                                  shuffle=True, num_workers=self.config.num_workers)
        outer_loader = DataLoader(datasets["val"], batch_size=self.config.batch_size,
                                  shuffle=False, num_workers=self.config.num_workers)
        self.inner_steps = len(train_loader)
        # Keep the original iterator/population initialization order.
        train_iterator, outer_iterator = iter(train_loader), iter(outer_loader)
        self._initialize_inner_population()
        self._meta_log(f"[Rater] inner={inner_split}; outer={outer_split}; full_inner_steps={self.inner_steps}")
        uses_groups = self.outer_loss_type in {
            "worst_group_loss", "group_balanced_loss"
        }
        self._meta_log(
            f"[Rater] meta objective={self.outer_loss_type}; "
            f"group annotations used={uses_groups}."
        )
        initial = self._evaluate_inner_diagnostics(output_dir, 0, datasets, split_names)
        for role, dataset in datasets.items():
            self._score_snapshot(output_dir, 0, role, split_names[role], dataset)
        history = []
        best_outer_loss = float("inf")
        best_step = None
        checkpoint_dir = os.path.join(output_dir, "plots", "checkpoints")
        best_path = os.path.join(checkpoint_dir, "best_meta.pt")
        for step in tqdm(range(1, self.meta_steps + 1), desc="Rater full-epoch meta-training"):
            # Associate each measured outer loss with the rater that produced it.
            candidate_rater = _cpu_state(self.rater)
            candidate_optimizer = copy.deepcopy(self.outer_optimizer.state_dict())
            _, outer_loss, _, train_iterator, outer_iterator = self._meta_step(
                train_iterator, outer_iterator, train_loader, outer_loader)
            if not math.isfinite(outer_loss):
                raise RuntimeError(f"Nonfinite meta objective at step {step}.")
            row = {"step": step, "outer_loss": outer_loss,
                   "elapsed_seconds": time.monotonic() - self._run_started}
            history.append(row)
            _append_csv(_diagnostic_path(output_dir, "meta_history.csv"), [row])
            with open(_diagnostic_path(output_dir, "outer_objective_details.jsonl"), "a", encoding="utf-8") as fout:
                fout.write(json.dumps(_json_value({"step": step, "models": self._last_outer_stats})) + "\n")
            self._meta_log(f"[META {step:04d}] {self.outer_loss_type}={outer_loss:.6f}")
            latest = self._checkpoint_payload(step)
            latest.update(meta_step=step, outer_loss=outer_loss,
                          loss_timing="before_rater_optimizer_update")
            if outer_loss < best_outer_loss:
                best_outer_loss, best_step = outer_loss, step
                selected = {**latest, "rater_sd": candidate_rater,
                            "outer_optimizer_sd": candidate_optimizer,
                            "best_meta_step": step, "best_outer_loss": outer_loss,
                            "checkpoint_kind": "selected_pre_update_rater_with_committed_inner_models"}
                _atomic_torch_save(selected, best_path)
            latest.update(best_meta_step=best_step, best_outer_loss=best_outer_loss,
                          checkpoint_kind="latest_after_optimizer_update")
            _atomic_torch_save(latest, os.path.join(checkpoint_dir, "latest_meta.pt"))
            if self.save_freq > 0 and step % self.save_freq == 0:
                _atomic_torch_save(latest, os.path.join(checkpoint_dir, f"meta_{step:06d}.pt"))
            should_eval = step == 1 or step % self.eval_freq == 0 or step == self.meta_steps
            should_plot = step == 1 or step % self.plot_freq == 0 or step == self.meta_steps
            if should_eval or should_plot:
                self._evaluate_inner_diagnostics(output_dir, step, datasets, split_names)
            if should_plot:
                for role, dataset in datasets.items():
                    self._score_snapshot(output_dir, step, role, split_names[role], dataset)
                self._save_metric_curves(output_dir)
                self._save_meta_training_curves(history, self._group_rows, output_dir)
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        self.rater.load_state_dict(selected["rater_sd"])
        self.outer_optimizer.load_state_dict(selected["outer_optimizer_sd"])
        for model, state in zip(self.inner_models, selected["inner_model_sds"]):
            model.load_state_dict(state)
        self.rater.eval()
        # Preserve the standalone notebook's checkpoint contract and keep it compact.
        export = {
            "rater_sd": _cpu_state(self.rater), "meta_steps": self.meta_steps,
            "best_meta_step": best_step, "best_outer_loss": best_outer_loss,
            "outer_loss_type": self.outer_loss_type, "outer_lse_beta": self.outer_lse_beta,
            "outer_top_loss_percent_per_class": self.outer_top_loss_percent,
            "outer_loss_uses_group_annotations": self.outer_loss_type in {
                "worst_group_loss", "group_balanced_loss"
            },
            "outer_loss_class_aggregation": {
                "standard_ce": "mean_over_all_outer_examples",
                "worst_group_loss": "maximum_observed_group_mean_ce",
                "group_balanced_loss": "equal_mean_of_observed_group_mean_ce",
                "classwise_top_loss": "equal_mean_of_per_class_top_tail_means",
                "class_balanced": "equal_mean_of_per_class_mean_losses",
                "worst_class_loss": "maximum_per_class_mean_loss",
                "logsumexp": "normalized_logsumexp_over_all_examples",
            }[self.outer_loss_type],
            "feature_dim": self.feature_dim, "num_classes": self.n_classes,
            "backbone": self.config.backbone, "rater_capacity": self.rater_capacity,
            "weighting": self.weighting, "temperature": self.temperature,
            "ea_style_gate": "all=RaterRating", "rate_all_samples": True,
            "rater_applied_only_to_misclassified": False,
            "inner_models": self.num_inner_models, "inner_initialization": "independent_random_linear",
            "full_train_epoch_before_each_meta_update": True, "full_inner_steps": self.inner_steps,
            "inner_lr": self.inner_lr, "outer_lr": self.outer_lr,
            "erm_model_path": self.erm_model_path, "erm_fingerprint": self.erm_checkpoint_fingerprint,
            "erm_epoch": self.erm_checkpoint_epoch,
        }
        _atomic_torch_save(export, os.path.join(output_dir, "final_rater.pt"))
        self._selected_inner_metrics = self._evaluate_inner_diagnostics(
            output_dir, best_step, datasets, split_names, phase="selected")
        for role, dataset in datasets.items():
            self._score_snapshot(output_dir, best_step, role, split_names[role], dataset, phase="selected")
        self._save_gradcam(output_dir, "selected", split_names.values(),
                           {"rater": self.rater, **{f"inner_{i}": m for i, m in enumerate(self.inner_models)}})
        summary = {
            "seed": self.config.seed, "status": "complete", "stage": "meta",
            "best_meta_step": best_step, "best_outer_loss": best_outer_loss,
            "calibration_size": len(datasets["val"]),
            "initial_inner_metrics": initial, "selected_inner_metrics": self._selected_inner_metrics,
            "elapsed_seconds": time.monotonic() - self._run_started, "split_names": split_names,
            "notes": ["Only the rater is trained/exported; downstream training is a separate notebook.",
                      "Each inner model sees a full training epoch before each rater update.",
                      "Losses are unweighted mean CE; worst_group_loss is the largest observed group-mean CE.",
                      "Diagnostic weights use fixed ordered batches and are batch-relative.",
                      "Group counts are logged; absent groups are excluded from WGA and worst-group loss.",
                      "Held-out validation and test metrics do not select the rater.",
                      "Selected inner heads match the selected rater's meta step."],
        }
        _write_json(_diagnostic_path(output_dir, "summary.json"), summary)
        lines = ["RATER META-LEARNING SUMMARY", f"Seed: {self.config.seed}",
                 f"Splits: {split_names}", f"Calibration samples: {summary['calibration_size']}",
                 f"Selected meta step: {best_step}; outer loss: {best_outer_loss:.6f}", ""]
        for name, by_split in self._selected_inner_metrics.items():
            for role, metrics in by_split.items():
                lines.append(f"{name}/{role}: " + ", ".join(
                    f"{field}={metrics[field]:.6f}" for field in METRIC_FIELDS))
        lines.extend(["", *summary["notes"], "", "Artifacts:",
                      "final_rater.pt: compact rater export compatible with the standalone notebook",
                      "plots/: existing distributions, per-inner-model curves and selected-model Grad-CAM",
                      "plots/data/: metric/group CSVs, source NPZs, metadata, indices and summary.json",
                      "plots/checkpoints/: best/latest population + optimizer/RNG states and optional snapshots",
                      "log.txt, summary.txt"])
        with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as fout:
            fout.write("\n".join(lines) + "\n")
        self._meta_log(f"[Rater] Meta-training complete; exported {os.path.join(output_dir, 'final_rater.pt')}")

    def test(self, output_dir, split=("test",), result_path=""):
        """Compatibility with main.py: classification diagnostics are part of train()."""
        del output_dir, split, result_path
        if self.precompute_embeddings_only:
            return
        log("[Rater] Meta-only run. Inner-model metrics are in plots/data/; "
            "final_rater.pt is ready for the standalone downstream notebook.")
        return getattr(self, "_selected_inner_metrics", {})


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _read_diagnostic_run(run_dir):
    data_dir = _diagnostic_path(run_dir)
    with open(os.path.join(data_dir, "run_config.json"), encoding="utf-8") as fin:
        metadata = json.load(fin)
    rows = {}
    for filename in ("metrics.csv", "group_statistics.csv"):
        with open(os.path.join(data_dir, filename), newline="", encoding="utf-8") as fin:
            values = list(csv.DictReader(fin))
        for row in values:
            if int(row["seed"]) != int(metadata["seed"]) or row["stage"] != "meta":
                raise ValueError(f"Expected one meta-training seed in {data_dir}/{filename}.")
        rows[filename] = values
    return metadata, rows


def plot_run_comparison(run_dirs, output_dir, comparison):
    """Observed calibration-size/initial-WGA comparisons from existing meta runs only."""
    if comparison not in {"calibration_size", "initial_wga"}:
        raise ValueError("comparison must be calibration_size or initial_wga")
    points, references = [], None
    for run_dir in run_dirs:
        info, data = _read_diagnostic_run(run_dir)
        settings = dict(info["settings"])
        if comparison == "calibration_size":
            settings.pop("calibration_size", None)
            settings.pop("split_val", None)
        if references is not None and settings != references:
            raise ValueError("Comparison runs differ in settings beyond the requested variable and seed.")
        references = settings
        initial_wga = {r["model"]: float(r["worst_group_accuracy"])
                       for r in data["metrics.csv"] if r["phase"] == "trajectory"
                       and int(r["step"]) == 0 and r["role"] == "val"}
        selected = [r for r in data["metrics.csv"] if r["phase"] == "selected"]
        if not selected:
            raise ValueError(f"No selected-rater metrics in {run_dir}; comparisons need completed runs.")
        for row in selected:
            x = info["settings"]["calibration_size"] if comparison == "calibration_size" else initial_wga[row["model"]]
            points.append({"seed": int(info["seed"]), "model": row["model"], "split": row["split"],
                           "role": row["role"], "x": x, "selected_meta_step": int(row["step"]),
                           "run_dir": os.path.abspath(run_dir),
                           **{field: float(row[field]) for field in METRIC_FIELDS}})
    if not points:
        raise ValueError("No comparison data found.")
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, f"{comparison}_observations.csv"), points)
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["model"], point["split"], point["x"])].append(point)
    rows = []
    for (model, split, x), values in sorted(grouped.items()):
        if len({v["seed"] for v in values}) != len(values):
            raise ValueError(f"Repeated seed/condition for {model}/{split}/{x}.")
        row = {"model": model, "split": split, "x": x, "n_seeds": len(values)}
        for field in METRIC_FIELDS:
            row[field + "_mean"], row[field + "_std"] = _mean_std([v[field] for v in values])
        rows.append(row)
    _write_csv(os.path.join(output_dir, f"{comparison}_mean_std.csv"), rows)
    for model in sorted({r["model"] for r in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, field, label in zip(axes.flat, METRIC_FIELDS, METRIC_LABELS):
            for split in sorted({r["split"] for r in rows if r["model"] == model}):
                values = sorted((r for r in rows if r["model"] == model and r["split"] == split),
                                key=lambda r: r["x"])
                ax.errorbar([r["x"] for r in values], [r[field + "_mean"] for r in values],
                            yerr=[r[field + "_std"] for r in values], marker="o", capsize=3,
                            linestyle="-" if comparison == "calibration_size" else "none", label=split)
            ax.set(xlabel="Outer calibration samples" if comparison == "calibration_size" else "Initial outer-validation WGA",
                   ylabel=label)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{model} at selected-rater step: observed {comparison} comparison")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{model}_{comparison}.png"), dpi=160)
        plt.close(fig)
    _write_json(os.path.join(output_dir, "comparison.json"), {
        "comparison": comparison,
        "interpretation": "Observed meta-training outcomes, not a controlled causal sensitivity experiment.",
        "initial_wga_split": "outer calibration split",
        "note": "Changing split_val may also change held-out validation membership; use recorded split IDs.",
        "runs": [os.path.abspath(p) for p in run_dirs],
    })
    return rows


def plot_multi_seed_runs(run_dirs, output_dir):
    """Aggregate observed meta steps, without interpolating missing evaluations."""
    all_rows, group_rows = [], []
    seeds, metadata = set(), []
    reference = None
    for run_dir in run_dirs:
        info, data = _read_diagnostic_run(run_dir)
        seed = int(info["seed"])
        if seed in seeds:
            raise ValueError(f"Seed {seed} occurs twice; pass one run per seed and condition.")
        seeds.add(seed)
        comparable = {"settings": info["settings"], "splits": info["splits"], "split_sizes": info["split_sizes"]}
        if reference is not None and comparable != reference:
            raise ValueError("Runs have different settings or splits; aggregate one condition at a time, or use --compare.")
        reference = comparable
        metadata.append({"seed": seed, "run_dir": os.path.abspath(run_dir)})
        all_rows.extend(data["metrics.csv"])
        group_rows.extend(data["group_statistics.csv"])
    if not all_rows:
        raise ValueError("No metric rows found.")
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "runs.json"), metadata)
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["phase"], row["model"], row["split"], row["role"],
               int(row["step"]) if row["phase"] == "trajectory" else -1)
        grouped[key].append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        if len({int(v["seed"]) for v in values}) != len(values):
            raise ValueError(f"Duplicate metric observation for {key}.")
        result = dict(zip(("phase", "model", "split", "role", "step"), key))
        result["n_seeds"] = len(values)
        for field in METRIC_FIELDS:
            result[field + "_mean"], result[field + "_std"] = _mean_std([v[field] for v in values])
        aggregate.append(result)
    _write_csv(os.path.join(output_dir, "metrics_mean_std.csv"), aggregate)
    for model, phase in sorted({(r["model"], r["phase"]) for r in aggregate}):
        rows = [r for r in aggregate if (r["model"], r["phase"]) == (model, phase)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, field, label in zip(axes.flat, METRIC_FIELDS, METRIC_LABELS):
            if phase == "selected":
                ax.bar([r["split"] for r in rows], [r[field + "_mean"] for r in rows],
                       yerr=[r[field + "_std"] for r in rows], capsize=3)
                ax.tick_params(axis="x", labelrotation=15)
            else:
                for split in sorted({r["split"] for r in rows}):
                    values = sorted((r for r in rows if r["split"] == split), key=lambda r: r["step"])
                    x = np.array([r["step"] for r in values])
                    mean = np.array([r[field + "_mean"] for r in values])
                    std = np.array([r[field + "_std"] for r in values])
                    counts = [r["n_seeds"] for r in values]
                    ax.plot(x, mean, label=f"{split} (n={min(counts)}–{max(counts)})")
                    ax.fill_between(x, mean - std, mean + std, alpha=.15)
                ax.set_xlabel("Meta step")
                ax.legend(fontsize=8)
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        fig.suptitle(f"{model}/{phase}: mean ± seed SD")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{model}_{phase}.png"), dpi=160)
        plt.close(fig)
    grouped = defaultdict(list)
    for row in group_rows:
        if row["phase"] == "trajectory":
            grouped[(row["split"], int(row["group"]), int(row["step"]))].append(row)
    group_aggregate = []
    for key, values in sorted(grouped.items()):
        if len({int(v["seed"]) for v in values}) != len(values):
            raise ValueError(f"Duplicate group-weight observation for {key}.")
        result = dict(zip(("split", "group", "step"), key))
        result["n_seeds"] = len(values)
        for field in ("raw_mean", "weight_mean", "weight_mass_share"):
            result[field + "_mean"], result[field + "_std"] = _mean_std([v[field] for v in values])
        group_aggregate.append(result)
    _write_csv(os.path.join(output_dir, "group_weights_mean_std.csv"), group_aggregate)
    for split in sorted({r["split"] for r in group_aggregate}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, field in zip(axes, ("raw_mean", "weight_mean", "weight_mass_share")):
            for group in sorted({r["group"] for r in group_aggregate if r["split"] == split}):
                values = sorted((r for r in group_aggregate if (r["split"], r["group"]) == (split, group)),
                                key=lambda r: r["step"])
                x = [r["step"] for r in values]
                mean = np.array([r[field + "_mean"] for r in values])
                std = np.array([r[field + "_std"] for r in values])
                ax.plot(x, mean, label=f"Group {group}")
                ax.fill_between(x, mean - std, mean + std, alpha=.15)
            ax.set(xlabel="Meta step", ylabel=field)
            ax.legend(fontsize=8)
        fig.suptitle(f"{split}: group weights, mean ± seed SD")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{split}_group_weights.png"), dpi=160)
        plt.close(fig)
    return aggregate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot existing rater meta-training runs; never launch training.")
    parser.add_argument("--runs", nargs="+", required=True, help="Existing rater output directories.")
    parser.add_argument("--output", required=True, help="Directory for aggregate CSVs and figures.")
    parser.add_argument("--compare", choices=("calibration_size", "initial_wga"),
                        help="Compare existing runs instead of averaging a single condition across seeds.")
    args = parser.parse_args()
    if args.compare:
        plot_run_comparison(args.runs, args.output, args.compare)
    else:
        plot_multi_seed_runs(args.runs, args.output)
