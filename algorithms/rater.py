import os
import csv

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
    """
    Bilevel feature rater.

    Image x -> frozen ImageNet ResNet-50 -> z -> r_eta(z) -> scalar score.

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
      * uses a persistent shared embedding cache so frozen ResNet-50
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
            log(
                f"[Rater] Warning: requested backbone is "
                f"{self.config.backbone}, not resnet50."
            )

        if not self.config.pretrained:
            raise ValueError(
                "Rater is designed for an ImageNet-pretrained backbone. "
                "Run with --pretrained True."
            )

        # ----------------------------------------------------
        # Frozen ImageNet feature extractor
        # ----------------------------------------------------
        self.feature_model = Classifier(
            backbone=self.config.backbone,
            num_classes=self.n_classes,
            pretrained=True,
        ).to(self.device)

        self.feature_model.eval()
        for p in self.feature_model.parameters():
            p.requires_grad_(False)

        self.feature_dim = self.feature_model.backbone.num_features

        log(
            f"[Rater] Frozen {self.config.backbone} feature extractor. "
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
            getattr(self.config, "rater_num_inner_models", 4)
        )
        self.inner_lr = float(
            getattr(self.config, "rater_inner_lr", 1e-2)
        )
        self.outer_lr = float(
            getattr(self.config, "rater_outer_lr", 3e-4)
        )
        self.temperature = float(
            getattr(self.config, "rater_temperature", 2.0)
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
            getattr(self.config, "rater_weighting", "softmax")
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
        self.outer_reg_weight = float(
            getattr(self.config, "rater_outer_reg_weight", 0.0)
        )
        self.score_reg_weight = float(
            getattr(self.config, "rater_score_reg_weight", 0.0)
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
        self.save_plot_data = bool(
            getattr(self.config, "rater_save_plot_data", True)
        )

        # ----------------------------------------------------
        # Final weighted classifier hyperparameters.
        # These have defaults, so config.py does not HAVE to
        # define them unless you want command-line control.
        # ----------------------------------------------------
        self.final_epochs = int(
            getattr(self.config, "rater_final_epochs", 50)
        )
        self.final_lr = float(
            getattr(self.config, "rater_final_lr", 1e-2)
        )
        self.final_momentum = float(
            getattr(self.config, "rater_final_momentum", 0.9)
        )
        self.final_weight_decay = float(
            getattr(self.config, "rater_final_weight_decay", 1e-4)
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
            getattr(self.config, "rater_final_plot_freq", 1)
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

        log(
            f"[Rater] Misclassified-sample Rater transform = "
            f"{self.weighting}; temperature = {self.temperature}"
        )
        log(
            "[Rater] EA-style gating enabled: correct samples get "
            "weight=1; only misclassified samples use the Rater."
        )
        log(
            "[Rater] Inner objective remains L_inner = sum_i w_i * CE_i "
            "WITHOUT dividing by sum(w)."
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
        Namespace shared embeddings by representation.
        """
        dataset_name = self._safe_cache_token(
            getattr(self.config, "dataset", "dataset")
        )
        backbone_name = self._safe_cache_token(
            self.config.backbone
        )
        resolution = self._safe_cache_token(
            getattr(self.config, "resolution", 224)
        )

        pretrained_tag = (
            "imagenet_pretrained"
            if bool(self.config.pretrained)
            else "not_pretrained"
        )

        return os.path.join(
            dataset_name,
            f"{backbone_name}_{pretrained_tag}",
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
    # ========================================================
    # Inner models
    # ========================================================

    def _new_inner_model(self):
        return InnerLinearClassifier(
            input_dim=self.feature_dim,
            num_classes=self.n_classes,
        ).to(self.device)

    def _initialize_inner_population(self):
        self.inner_models = [
            self._new_inner_model()
            for _ in range(self.num_inner_models)
        ]

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

    def _ea_style_training_weights(
        self,
        z,
        y,
        logits,
        raw_scores_all=None,
    ):
        """
        Build effective sample weights:

            weight_i = 1
                if argmax(logits_i) == y_i

            weight_i = RaterRating(z_i)
                otherwise

        During actual training `raw_scores_all` is None, so the Rater is
        called ONLY for misclassified samples. For diagnostics, optional
        precomputed raw scores can be supplied.
        """

        pred = logits.argmax(dim=1)
        correct_mask = pred.eq(y)
        misclassified_mask = ~correct_mask

        weights = torch.ones(
            y.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        )

        if not bool(misclassified_mask.any()):
            empty_scores = torch.empty(
                0,
                device=logits.device,
                dtype=logits.dtype,
            )
            return weights, correct_mask, empty_scores

        misclassified_indices = torch.nonzero(
            misclassified_mask,
            as_tuple=False,
        ).squeeze(1)

        if raw_scores_all is None:
            misclassified_raw_scores = self.rater(
                z[misclassified_mask]
            )
        else:
            misclassified_raw_scores = raw_scores_all[
                misclassified_mask
            ]

        misclassified_ratings = self._scores_to_weights(
            misclassified_raw_scores
        )

        weights = weights.index_copy(
            0,
            misclassified_indices,
            misclassified_ratings,
        )

        return weights, correct_mask, misclassified_raw_scores

    # ========================================================
    # Differentiable inner optimization
    # ========================================================

    def _inner_unroll(self, inner_model, train_iterator, train_loader):
        """Perform differentiable EA-style weighted inner updates."""

        fast_weight = (
            inner_model.linear.weight.detach().clone().requires_grad_(True)
        )
        fast_bias = (
            inner_model.linear.bias.detach().clone().requires_grad_(True)
        )

        last_raw_scores = None

        for _ in range(self.inner_steps):
            batch, train_iterator = self._next_batch(
                train_iterator, train_loader
            )
            z, y, _, _ = batch
            z = z.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Predict first. The current prediction determines whether the
            # example gets weight 1 or a Rater-derived rating.
            logits = F.linear(z, fast_weight, fast_bias)

            per_sample_loss = F.cross_entropy(
                logits,
                y,
                reduction="none",
            )

            weights, _, misclassified_raw_scores = (
                self._ea_style_training_weights(
                    z=z,
                    y=y,
                    logits=logits,
                    raw_scores_all=None,
                )
            )

            # Correct -> 1; wrong -> Rater rating.
            # No division by sum(weights).
            inner_loss = (
                per_sample_loss * weights
            ).sum()

            if self.inner_reg_weight > 0:
                inner_reg = (
                    fast_weight.pow(2).sum()
                    + fast_bias.pow(2).sum()
                )
                inner_loss = (
                    inner_loss
                    + self.inner_reg_weight * inner_reg
                )

            grad_w, grad_b = torch.autograd.grad(
                inner_loss,
                [fast_weight, fast_bias],
                create_graph=True,
            )

            fast_weight = fast_weight - self.inner_lr * grad_w
            fast_bias = fast_bias - self.inner_lr * grad_b
            last_raw_scores = misclassified_raw_scores

        fast_params = {
            "weight": fast_weight,
            "bias": fast_bias,
        }

        return fast_params, train_iterator, last_raw_scores

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
        self.rater.train()

        outer_batch, val_iterator = self._next_batch(
            val_iterator, val_loader
        )
        z_outer, y_outer, _, _ = outer_batch
        z_outer = z_outer.to(self.device, non_blocking=True)
        y_outer = y_outer.to(self.device, non_blocking=True)

        outer_losses = []
        fast_parameter_sets = []

        for inner_model in self.inner_models:
            fast_params, train_iterator, _ = self._inner_unroll(
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

            outer_losses.append(outer_loss)
            fast_parameter_sets.append(fast_params)

        meta_loss = torch.stack(outer_losses).mean()

        if self.outer_reg_weight > 0:
            outer_reg = sum(
                p.pow(2).sum()
                for p in self.rater.parameters()
            )
            meta_loss = (
                meta_loss
                + self.outer_reg_weight * outer_reg
            )

        if self.score_reg_weight > 0:
            outer_scores = self.rater(z_outer)
            score_reg = -torch.var(outer_scores)
            meta_loss = (
                meta_loss
                + self.score_reg_weight * score_reg
            )

        self.outer_optimizer.zero_grad()
        meta_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.rater.parameters(),
                self.grad_clip,
            )

        self.outer_optimizer.step()

        # Advance persistent inner models using detached fast parameters.
        with torch.no_grad():
            for inner_model, fast_params in zip(
                self.inner_models,
                fast_parameter_sets,
            ):
                inner_model.linear.weight.copy_(
                    fast_params["weight"].detach()
                )
                inner_model.linear.bias.copy_(
                    fast_params["bias"].detach()
                )

        return (
            float(meta_loss.detach().item()),
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
                    self._ea_style_training_weights(
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
                "Effective EA-style training weight "
                "(correct=1, wrong=Rater rating)"
            )
            title_name = (
                f"EA-style effective weight distribution "
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

        log(
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

        FINAL SCORE means the effective EA-style training weight:
        correct -> 1; wrong -> Rater rating.
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
            "Effective EA-style weight (correct=1, wrong=Rater rating)"
        )
        ax.set_title(
            f"EA-style effective weight vs inner-model loss "
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
                self._ea_style_training_weights(
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
        effective EA-style weights for that classifier.
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
                    self._ea_style_training_weights(
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
            "weights_are_ea_style_gated": model is not None,
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

    def _save_rater_checkpoint(self, path, meta_step, outer_loss):
        torch.save(
            {
                "rater_sd": self.rater.state_dict(),
                "outer_optimizer_sd": self.outer_optimizer.state_dict(),
                "meta_step": meta_step,
                "outer_loss": outer_loss,
                "feature_dim": self.feature_dim,
                "backbone": self.config.backbone,
                "pretrained": True,
                "rater_capacity": self.rater_capacity,
                "rater_weighting": self.weighting,
                "rater_temperature": self.temperature,
                "embedding_cache_root": self.embedding_cache_root,
                "embedding_cache_namespace": (
                    self._embedding_cache_namespace()
                ),
                "ea_style_correct_weight": 1.0,
                "ea_style_rater_only_on_misclassified": True,
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

        log(f"[Rater] Loaded checkpoint from {path}")

    def _save_population_snapshot(self, output_dir, meta_step):
        """Save raw state_dict snapshots for post-hoc evaluation."""

        rater_dir = os.path.join(output_dir, "rater_checkpoints")
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(rater_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(
            self.rater.state_dict(),
            os.path.join(
                rater_dir,
                f"data_rater_{meta_step:06d}.pt",
            ),
        )

        for i, model in enumerate(self.inner_models):
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_dir,
                    f"inner_model{i}_{meta_step:06d}.pt",
                ),
            )

    def _save_final_classifier(self, path, model):
        torch.save(
            {
                "model_sd": model.state_dict(),
                "feature_dim": self.feature_dim,
                "num_classes": self.n_classes,
            },
            path,
        )

    def _load_final_classifier(self, path):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        model = self._new_inner_model()
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
        Final classifier protocol.

        EA-style:
            calibration / classifier training = val_subset1
            checkpoint selection              = val_subset2
            final reporting                   = test
        """
        if self.use_calibration_final:
            if not self._has_ea_calibration_split():
                raise ValueError(
                    "EA-style final-classifier calibration is enabled, "
                    "but val_subset1/val_subset2 do not exist. "
                    "Run with --split_val 0.4 (the EA repository example) "
                    "or another value < 1."
                )

            return "val_subset1", "val_subset2"

        return "train_no_aug", "val"

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
        Save histograms of the ACTUAL EA-style weights used during one
        final-classifier training epoch.

        Effective weight:
            correct sample -> 1
            wrong sample   -> trained-Rater rating

        Two plots are saved:
          1) all effective weights;
          2) only misclassified samples, where effective_weight is exactly
             the trained Rater rating.

        Values are recorded before each SGD update, so these are the weights
        the final classifier actually saw.
        """

        weights = np.asarray(weights, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64)
        correctness = np.asarray(correctness, dtype=np.float64)

        if len(weights) == 0:
            return

        correct_mask = correctness >= 0.5
        wrong_mask = ~correct_mask

        # Save numeric data
        data_dir = os.path.join(
            output_dir,
            "plots",
            "final_classifier_epoch_weight_data",
        )
        os.makedirs(data_dir, exist_ok=True)

        csv_path = os.path.join(
            data_dir,
            f"epoch_{epoch:03d}_{split_name}_weights.csv",
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "effective_weight",
                    "group",
                    "correct_before_update",
                    "misclassified_before_update",
                ]
            )
            for i in range(len(weights)):
                writer.writerow(
                    [
                        i,
                        float(weights[i]),
                        int(groups[i]),
                        int(correct_mask[i]),
                        int(wrong_mask[i]),
                    ]
                )

        # Plot 1: all effective weights
        plot_dir = os.path.join(
            output_dir,
            "plots",
            "final_classifier_epoch_effective_weight_histogram",
        )
        os.makedirs(plot_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        bins = np.linspace(0.0, 1.000001, 41)

        for gid in sorted(np.unique(groups)):
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
            "Effective EA-style training weight "
            "(correct=1, wrong=trained-Rater rating)"
        )
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.02)
        ax.set_title(
            f"Final-classifier ACTUAL training weights by group | "
            f"epoch {epoch}\\n"
            f"correct={100.0 * correct_mask.mean():.2f}% | "
            f"misclassified={100.0 * wrong_mask.mean():.2f}%"
        )
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        effective_plot_path = os.path.join(
            plot_dir,
            f"epoch_{epoch:03d}_{split_name}_effective_weight_hist.png",
        )
        fig.savefig(
            effective_plot_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot 2: misclassified samples only
        wrong_plot_path = None

        if np.any(wrong_mask):
            wrong_plot_dir = os.path.join(
                output_dir,
                "plots",
                "final_classifier_epoch_misclassified_rater_histogram",
            )
            os.makedirs(wrong_plot_dir, exist_ok=True)

            wrong_weights = weights[wrong_mask]
            wrong_groups = groups[wrong_mask]

            fig, ax = plt.subplots(figsize=(10, 7))
            bins = np.linspace(0.0, 1.000001, 41)

            for gid in sorted(np.unique(wrong_groups)):
                mask = wrong_groups == gid
                if not np.any(mask):
                    continue

                ax.hist(
                    wrong_weights[mask],
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    label=(
                        f"{self._group_display_name(gid)} "
                        f"(wrong n={int(mask.sum())})"
                    ),
                )

            ax.set_xlabel(
                f"Trained-Rater {self.weighting} rating "
                "(misclassified samples only)"
            )
            ax.set_ylabel("Density")
            ax.set_xlim(0.0, 1.02)
            ax.set_title(
                f"Rater ratings of misclassified training samples by group "
                f"| final-classifier epoch {epoch}"
            )
            ax.grid(True, linestyle=":", alpha=0.30)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()

            wrong_plot_path = os.path.join(
                wrong_plot_dir,
                f"epoch_{epoch:03d}_{split_name}_misclassified_rating_hist.png",
            )
            fig.savefig(
                wrong_plot_path,
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Group summaries
        summary_parts = []

        for gid in sorted(np.unique(groups)):
            group_mask = groups == gid
            group_wrong = group_mask & wrong_mask

            mean_weight = float(weights[group_mask].mean())
            wrong_rate = float(
                group_wrong.sum() / max(group_mask.sum(), 1)
            )

            if np.any(group_wrong):
                mean_wrong_rating = float(
                    weights[group_wrong].mean()
                )
            else:
                mean_wrong_rating = float("nan")

            summary_parts.append(
                f"g{int(gid)}: mean_w={mean_weight:.3f}, "
                f"wrong={100.0 * wrong_rate:.1f}%, "
                f"mean_wrong_rating={mean_wrong_rating:.3f}"
            )

        log(
            f"[Final classifier epoch {epoch:03d} weights] "
            + "; ".join(summary_parts)
        )
        log(
            f"[Final classifier plot] saved actual-weight histogram: "
            f"{effective_plot_path}"
        )

        if wrong_plot_path is not None:
            log(
                f"[Final classifier plot] saved misclassified-rating "
                f"histogram: {wrong_plot_path}"
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
        calibration_split="val_subset1",
        selection_split="val_subset2",
        test_split="test",
    ):
        """
        Train a FRESH downstream linear classifier on the CALIBRATION set
        with the already-trained Rater frozen as the weighting network.

        EA-style data roles:
            calibration_dataset -> classifier optimization
            selection_dataset   -> checkpoint/model selection
            test_dataset        -> DIAGNOSTIC evaluation every epoch

        IMPORTANT:
            test metrics are NEVER used for checkpoint selection.
            They are logged only so we can inspect generalization dynamics.

        Per minibatch:
            current classifier predicts
                -> correct: weight = 1
                -> wrong:   weight = trained-Rater rating

        The classifier is then updated with:

            sum_i weight_i * CE_i

        with NO division by sum(weights).

        The Rater is never updated during this stage.

        For every final-classifier epoch, this function saves:
          * histogram of the ACTUAL effective training weights by group;
          * histogram of Rater ratings for misclassified samples only;
          * CSV of all actual weights/group/correctness values;
          * validation effective-weight histogram.

        By default, the best final classifier is selected by held-out selection WGA.
        """

        log(
            "[Rater] Training fresh final classifier with "
            "FROZEN trained Rater..."
        )
        log(
            f"[Final classifier] CALIBRATION/TRAIN split = "
            f"{calibration_split} ({len(calibration_dataset)} samples)"
        )
        log(
            f"[Final classifier] SELECTION split = "
            f"{selection_split} ({len(selection_dataset)} samples)"
        )
        log(
            f"[Final classifier] TEST diagnostic split = "
            f"{test_split} ({len(test_dataset)} samples)"
        )
        log(
            "[Final classifier] TEST is diagnostic only and is NOT used "
            "for checkpoint selection."
        )
        log(
            f"[Final classifier] selection metric = "
            f"{self.final_selection}"
        )
        log(
            "[Final classifier] weighting rule: "
            "correct -> 1.0; "
            f"misclassified -> trained-Rater {self.weighting} rating."
        )

        # Freeze the trained Rater completely.
        self.rater.eval()
        for p in self.rater.parameters():
            p.requires_grad_(False)

        model = self._new_inner_model()

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

        for epoch in range(1, self.final_epochs + 1):
            model.train()

            running_weighted_loss = 0.0
            total_correct = 0
            total_examples = 0
            num_batches = 0

            # Exact weights used during this epoch.
            epoch_weights = []
            epoch_groups = []
            epoch_correctness = []

            for z, y, g, _ in train_loader:
                z = z.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                g = torch.as_tensor(g).to(
                    self.device,
                    non_blocking=True,
                )

                # Current classifier predicts first.
                logits = model(z)

                # EA-style gate with the frozen trained Rater.
                with torch.no_grad():
                    (
                        weights,
                        correct_mask,
                        _,
                    ) = self._ea_style_training_weights(
                        z=z,
                        y=y,
                        logits=logits,
                        raw_scores_all=None,
                    )

                # Store EXACT pre-update values used by this SGD step.
                epoch_weights.append(
                    weights.detach().cpu()
                )
                epoch_groups.append(
                    g.detach().cpu()
                )
                epoch_correctness.append(
                    correct_mask.float().detach().cpu()
                )

                per_sample_loss = F.cross_entropy(
                    logits,
                    y,
                    reduction="none",
                )

                weighted_loss = (
                    per_sample_loss * weights
                ).sum()

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

                running_weighted_loss += weighted_loss.item()
                total_correct += correct_mask.sum().item()
                total_examples += y.numel()
                num_batches += 1

            train_acc = total_correct / max(
                total_examples,
                1,
            )

            train_weighted_loss = (
                running_weighted_loss
                / max(num_batches, 1)
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

            # Held-out selection evaluation.
            selection_metrics = self._evaluate_classifier(
                model,
                selection_dataset,
            )

            selection_value = self._selection_value(
                selection_metrics
            )

            # ------------------------------------------------
            # TEST evaluation after EVERY epoch.
            #
            # This is DIAGNOSTIC ONLY. It must not influence
            # checkpoint/model selection.
            # ------------------------------------------------
            test_metrics = self._evaluate_classifier(
                model,
                test_dataset,
            )

            history.append(
                [
                    epoch,
                    train_weighted_loss,
                    train_acc,
                    selection_metrics["loss"],
                    selection_metrics["accuracy"],
                    selection_metrics["worst_group_accuracy"],
                    test_metrics["loss"],
                    test_metrics["accuracy"],
                    test_metrics["worst_group_accuracy"],
                    float(epoch_weights_np.mean()),
                    float(epoch_weights_np.std()),
                    float(epoch_correctness_np.mean()),
                ]
            )

            log(
                f"[Final classifier epoch {epoch:03d}] "
                f"weighted_calibration_loss={train_weighted_loss:.6f}, "
                f"calibration_acc={100.0 * train_acc:.2f}%, "
                f"mean_effective_weight={epoch_weights_np.mean():.4f}, "
                f"selection_loss={selection_metrics['loss']:.6f}, "
                f"selection_acc={100.0 * selection_metrics['accuracy']:.2f}%, "
                f"selection_WGA="
                f"{100.0 * selection_metrics['worst_group_accuracy']:.2f}%, "
                f"test_loss={test_metrics['loss']:.6f}, "
                f"test_acc={100.0 * test_metrics['accuracy']:.2f}%, "
                f"test_WGA="
                f"{100.0 * test_metrics['worst_group_accuracy']:.2f}%"
            )

            test_group_str = ", ".join(
                f"g{gid}={100.0 * acc:.2f}%"
                for gid, acc in test_metrics["group_accuracy"].items()
            )

            log(
                f"[Final classifier epoch {epoch:03d} TEST groups] "
                f"{test_group_str}"
            )

            # Post-epoch selection-set weight histogram.
            if should_plot_final:
                selection_relationship = (
                    self._classifier_score_loss_relationship(
                        selection_dataset,
                        model,
                    )
                )

                self._save_group_score_histogram(
                    relationship=selection_relationship,
                    output_dir=output_dir,
                    meta_step=epoch,
                    split_name=f"{selection_split}_final_classifier_epoch",
                    tag_prefix="final_classifier",
                    score_kind="final",
                )

            # Save best downstream classifier according to selected metric.
            if selection_value > best_value:
                best_value = selection_value
                best_epoch = epoch

                best_metrics = {
                    "loss": float(selection_metrics["loss"]),
                    "accuracy": float(selection_metrics["accuracy"]),
                    "worst_group_accuracy": float(
                        selection_metrics["worst_group_accuracy"]
                    ),
                    "group_accuracy": {
                        int(k): float(v)
                        for k, v in selection_metrics[
                            "group_accuracy"
                        ].items()
                    },
                }

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                torch.save(
                    {
                        "model_sd": best_state,
                        "epoch": best_epoch,
                        "selection_metric": self.final_selection,
                        "selection_value": float(best_value),
                        "selection_metrics": best_metrics,
                    },
                    os.path.join(
                        output_dir,
                        "best_final_weighted_classifier.pt",
                    ),
                )

                log(
                    f"[Final classifier] NEW BEST at epoch "
                    f"{best_epoch:03d}: "
                    f"selection_acc={100.0 * best_metrics['accuracy']:.2f}%, "
                    f"selection_WGA="
                    f"{100.0 * best_metrics['worst_group_accuracy']:.2f}% "
                    f"(selected by {self.final_selection})"
                )

        if best_state is None:
            raise RuntimeError(
                "Final classifier training produced no model."
            )

        # Restore selected best downstream classifier.
        model.load_state_dict(best_state)
        model.to(self.device)
        model.eval()

        log(
            f"[Final classifier] Restored selected epoch "
            f"{best_epoch:03d}. "
            f"selection_acc={100.0 * best_metrics['accuracy']:.2f}%, "
            f"selection_WGA="
            f"{100.0 * best_metrics['worst_group_accuracy']:.2f}%."
        )

        history_path = os.path.join(
            output_dir,
            "final_classifier_history.csv",
        )

        with open(
            history_path,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "weighted_calibration_loss",
                    "calibration_accuracy",
                    "selection_loss",
                    "selection_accuracy",
                    "selection_wga",
                    "test_loss",
                    "test_accuracy",
                    "test_wga",
                    "mean_effective_calibration_weight",
                    "std_effective_train_weight",
                    "fraction_correct_before_update",
                ]
            )
            writer.writerows(history)

        # Keep trained Rater frozen from here onward.
        self.rater.eval()

        return model

    # ========================================================
    # Main training
    # ========================================================

    def train(self, output_dir, split="train"):
        os.makedirs(output_dir, exist_ok=True)

        if self.precompute_embeddings_only:
            self._precompute_shared_embeddings(output_dir)
            return

        cache_dir = self._resolve_embedding_cache_dir(output_dir)
        inner_split, outer_split = self._resolve_meta_splits(split)

        log(f"[Rater] Inner/meta-train split: {inner_split}")
        log(f"[Rater] Outer/held-out split: {outer_split}")
        log(
            f"[Rater embeddings] Shared cache directory: {cache_dir}"
        )

        train_dataset = self._extract_embedding_dataset(
            inner_split,
            cache_dir,
        )
        val_dataset = self._extract_embedding_dataset(
            outer_split,
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

        train_iterator = iter(train_loader)
        val_iterator = iter(val_loader)
        self._initialize_inner_population()

        refresh_period = max(1, self.refresh_steps)
        refresh_offsets = [
            (i * refresh_period) // self.num_inner_models
            for i in range(self.num_inner_models)
        ]

        best_outer_loss = float("inf")
        best_path = os.path.join(output_dir, "best_rater.pt")
        latest_path = os.path.join(output_dir, "latest_rater.pt")
        history_path = os.path.join(output_dir, "rater_history.csv")
        eval_history_path = os.path.join(output_dir, "rater_eval.csv")

        history = []
        eval_history = []

        log("[Rater] Starting feature-rater bilevel optimization")
        log(
            f"[Rater] meta_steps={self.meta_steps}, "
            f"inner_steps={self.inner_steps}, "
            f"inner_models={self.num_inner_models}, "
            f"inner_lr={self.inner_lr}, "
            f"outer_lr={self.outer_lr}, "
            f"temperature={self.temperature}"
        )
        log(
            "[Rater] Effective weighting rule: correct -> 1.0; "
            f"misclassified -> {self.weighting} Rater rating."
        )

        for meta_step in tqdm(
            range(1, self.meta_steps + 1),
            desc="Rater meta-training",
        ):
            # Staggered reset of persistent inner models.
            if meta_step > 1:
                position = (meta_step - 1) % refresh_period
                for i, offset in enumerate(refresh_offsets):
                    if position == offset:
                        self.inner_models[i] = self._new_inner_model()

            outer_loss, train_iterator, val_iterator = self._meta_step(
                train_iterator,
                val_iterator,
                train_loader,
                val_loader,
            )

            history.append([meta_step, outer_loss])

            if outer_loss < best_outer_loss:
                best_outer_loss = outer_loss
                self._save_rater_checkpoint(
                    best_path,
                    meta_step,
                    outer_loss,
                )

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
                population_metrics = self._evaluate_inner_population(
                    val_dataset,
                    verbose=should_eval,
                )
                relationship = self._score_loss_relationship(
                    val_dataset,
                    models=self.inner_models,
                )

                with torch.no_grad():
                    sample_z = val_dataset.embeddings[
                        : min(1024, len(val_dataset))
                    ].to(self.device)
                    self.rater.eval()
                    sample_scores = self.rater(sample_z)
                    score_mean = sample_scores.mean().item()
                    score_std = sample_scores.std().item()

                log(
                    f"[Rater step {meta_step}] "
                    f"weighting={self.weighting}, "
                    f"outer_loss={outer_loss:.6f}, "
                    f"score_mean={score_mean:.6f}, "
                    f"score_std={score_std:.6f}, "
                    f"Spearman(score, loss)="
                    f"{relationship['spearman_score_vs_loss']:.4f}, "
                    f"Pearson(score, loss)="
                    f"{relationship['pearson_score_vs_loss']:.4f}, "
                    f"Spearman(score, correctness)="
                    f"{relationship['spearman_score_vs_correctness']:.4f}, "
                    f"Spearman(final_score, loss)="
                    f"{relationship['spearman_final_score_vs_loss']:.4f}, "
                    f"Pearson(final_score, loss)="
                    f"{relationship['pearson_final_score_vs_loss']:.4f}"
                )

                # Group-colored score-vs-loss diagnostic. By default
                # plot frequency equals --eval_freq. If config.py defines
                # rater_plot_freq, that can be controlled independently.
                if should_plot:
                    self._save_all_score_diagnostics(
                        relationship=relationship,
                        output_dir=output_dir,
                        meta_step=meta_step,
                        split_name=outer_split,
                        tag_prefix="meta_step",
                    )

                if should_eval:
                    eval_history.append(
                        [
                        meta_step,
                        outer_loss,
                        score_mean,
                        score_std,
                        population_metrics["mean_loss"],
                        population_metrics["mean_accuracy"],
                        population_metrics["mean_wga"],
                        population_metrics["best_accuracy"],
                        population_metrics["best_wga"],
                        relationship["spearman_score_vs_loss"],
                        relationship["pearson_score_vs_loss"],
                        relationship["spearman_score_vs_correctness"],
                        relationship["spearman_final_score_vs_loss"],
                        relationship["pearson_final_score_vs_loss"],
                        relationship["spearman_final_score_vs_correctness"],
                    ]
                )

            if (
                getattr(self.config, "save_freq", 0) > 0
                and meta_step % self.config.save_freq == 0
            ):
                self._save_population_snapshot(output_dir, meta_step)

        # ----------------------------------------------------
        # Save meta histories
        # ----------------------------------------------------
        self._save_rater_checkpoint(
            latest_path,
            self.meta_steps,
            history[-1][1],
        )

        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["meta_step", "outer_loss"])
            writer.writerows(history)

        with open(eval_history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "meta_step",
                    "outer_loss",
                    "score_mean",
                    "score_std",
                    "mean_inner_loss",
                    "mean_inner_accuracy",
                    "mean_inner_wga",
                    "best_inner_accuracy",
                    "best_inner_wga",
                    "spearman_score_vs_loss",
                    "pearson_score_vs_loss",
                    "spearman_score_vs_correctness",
                    "spearman_final_score_vs_loss",
                    "pearson_final_score_vs_loss",
                    "spearman_final_score_vs_correctness",
                ]
            )
            writer.writerows(eval_history)

        # ----------------------------------------------------
        # Restore best rater
        # ----------------------------------------------------
        checkpoint = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )
        self.rater.load_state_dict(checkpoint["rater_sd"])
        self.rater.to(self.device)
        self.rater.eval()

        log(
            f"[Rater] Meta-training complete. "
            f"Best outer loss = {best_outer_loss:.6f}"
        )
        log(
            f"[Rater] Restored TRAINED best Rater checkpoint "
            f"from meta step {checkpoint.get('meta_step', 'unknown')}. "
            f"This Rater is now frozen and used to train the "
            f"fresh final classifier."
        )

        # Save learned scores on inner/outer datasets.
        train_scores = self._compute_scores(train_dataset)
        torch.save(
            train_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{inner_split}.pt",
            ),
        )

        val_scores = self._compute_scores(val_dataset)
        torch.save(
            val_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{outer_split}.pt",
            ),
        )

        # ----------------------------------------------------
        # EA-style CALIBRATION stage for the final classifier.
        #
        # val_subset1 = calibration/training
        # val_subset2 = held-out model selection
        # test        = final reporting
        # ----------------------------------------------------
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

        test_split = "test"
        test_dataset = self._extract_embedding_dataset(
            test_split,
            cache_dir,
        )

        log(
            "[Rater] Final-classifier EA-style calibration protocol:"
        )
        log(
            f"[Rater]   calibration/training = {calibration_split} "
            f"({len(calibration_dataset)} samples)"
        )
        log(
            f"[Rater]   model selection      = {selection_split} "
            f"({len(selection_dataset)} samples)"
        )
        log(
            "[Rater]   test diagnostics      = test after EVERY epoch"
        )
        log(
            "[Rater]   checkpoint selection  = selection split ONLY"
        )

        self.final_classifier = self._train_final_classifier(
            calibration_dataset,
            selection_dataset,
            test_dataset,
            output_dir,
            calibration_split=calibration_split,
            selection_split=selection_split,
            test_split=test_split,
        )

        final_model_path = os.path.join(
            output_dir,
            "final_weighted_classifier.pt",
        )
        self._save_final_classifier(
            final_model_path,
            self.final_classifier,
        )

        final_calibration_scores = self._compute_scores(
            calibration_dataset,
            model=self.final_classifier,
        )
        torch.save(
            final_calibration_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{calibration_split}_final_classifier.pt",
            ),
        )

        final_selection_scores = self._compute_scores(
            selection_dataset,
            model=self.final_classifier,
        )
        torch.save(
            final_selection_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{selection_split}_final_classifier.pt",
            ),
        )

        final_calibration_metrics = self._evaluate_classifier(
            self.final_classifier,
            calibration_dataset,
        )

        final_selection_metrics = self._evaluate_classifier(
            self.final_classifier,
            selection_dataset,
        )

        log(
            f"[Rater FINAL CALIBRATION] "
            f"loss={final_calibration_metrics['loss']:.6f}, "
            f"acc={100.0 * final_calibration_metrics['accuracy']:.2f}%, "
            f"WGA="
            f"{100.0 * final_calibration_metrics['worst_group_accuracy']:.2f}%"
        )

        log(
            f"[Rater FINAL SELECTION] "
            f"loss={final_selection_metrics['loss']:.6f}, "
            f"acc={100.0 * final_selection_metrics['accuracy']:.2f}%, "
            f"WGA="
            f"{100.0 * final_selection_metrics['worst_group_accuracy']:.2f}%"
        )

        for gid, acc in final_selection_metrics["group_accuracy"].items():
            log(
                f"[Rater FINAL SELECTION] group {gid}: "
                f"{100.0 * acc:.2f}%"
            )

        final_selection_relationship = (
            self._classifier_score_loss_relationship(
                selection_dataset,
                self.final_classifier,
            )
        )
        self._save_all_score_diagnostics(
            relationship=final_selection_relationship,
            output_dir=output_dir,
            meta_step=self.meta_steps,
            split_name=f"{selection_split}_final_classifier",
            tag_prefix="final",
        )

    # ========================================================
    # Test / final evaluation
    # ========================================================

    def test(self, output_dir, split=("test",), result_path=""):
        """
        For each requested split:
          1. load its PERSISTENT shared embeddings;
          2. compute and save rater scores;
          3. if a final weighted classifier is available, evaluate its
             loss, overall accuracy, per-group accuracy and WGA.
        """

        if self.precompute_embeddings_only:
            log(
                "[Rater embeddings] Precompute-only run finished; "
                "skipping normal test/evaluation stage."
            )
            return

        cache_dir = self._resolve_embedding_cache_dir(output_dir)

        if isinstance(split, str):
            split = [split]

        # main.py calls train() and then test() on the same object, so the
        # classifier normally already exists. This also supports a test-only
        # run if the saved final classifier is present in output_dir.
        if self.final_classifier is None:
            final_model_path = os.path.join(
                output_dir,
                "final_weighted_classifier.pt",
            )
            if os.path.exists(final_model_path):
                self.final_classifier = self._load_final_classifier(
                    final_model_path
                )

        for sp in split:
            dataset = self._extract_embedding_dataset(sp, cache_dir)

            # ----------------------------------------------
            # Rater-score evaluation
            # ----------------------------------------------
            payload = self._compute_scores(
                dataset,
                model=self.final_classifier,
            )
            score_path = os.path.join(
                output_dir,
                f"rater_scores_{sp}.pt",
            )
            torch.save(payload, score_path)

            summary = self._rating_summary(payload)
            log(f"[Rater] {sp} ratings saved to {score_path}")
            for line in summary:
                log(f"[Rater/{sp}] {line}")

            # ----------------------------------------------
            # Final classifier evaluation
            # ----------------------------------------------
            final_metrics = None
            if self.final_classifier is not None:
                final_metrics = self._evaluate_classifier(
                    self.final_classifier,
                    dataset,
                )

                group_str = ", ".join(
                    f"g{gid}={100.0 * acc:.2f}%"
                    for gid, acc in final_metrics["group_accuracy"].items()
                )

                log(
                    f"[Rater FINAL {sp.upper()}] "
                    f"loss={final_metrics['loss']:.6f}, "
                    f"acc={100.0 * final_metrics['accuracy']:.2f}%, "
                    f"WGA={100.0 * final_metrics['worst_group_accuracy']:.2f}% "
                    f"({group_str})"
                )

                metrics_path = os.path.join(
                    output_dir,
                    f"final_metrics_{sp}.pt",
                )
                torch.save(final_metrics, metrics_path)

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
                    split_name=f"{sp}_final_classifier",
                    tag_prefix="final",
                )

            if result_path:
                with open(result_path, "a") as fout:
                    fout.write(f"Rater {sp}\n")
                    for line in summary:
                        fout.write(line + "\n")

                    if final_metrics is not None:
                        fout.write(
                            f"Final classifier {sp}: "
                            f"loss={final_metrics['loss']:.6f}, "
                            f"accuracy={final_metrics['accuracy']:.6f}, "
                            f"wga={final_metrics['worst_group_accuracy']:.6f}\n"
                        )
                        for gid, acc in final_metrics[
                            "group_accuracy"
                        ].items():
                            fout.write(
                                f"group {gid} accuracy={acc:.6f}\n"
                            )
                        fout.write("\n")
