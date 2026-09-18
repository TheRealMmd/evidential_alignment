"""
EXAMINE: frozen-embedding linear-classifier gradient diagnostics.

This algorithm intentionally contains NO Rater and NO meta-learning.

It does exactly this:

    cached frozen ResNet-50 embeddings
                |
                v
        Linear(2048 -> K)
                |
                v
      ordinary cross-entropy training

After epoch 0 and after every training epoch, it computes exact
PER-SAMPLE gradients for the linear classifier and saves/plots their
distributions by Waterbirds group.

For one sample i:

    z_i        : frozen embedding
    p_i        : softmax probabilities
    y_i        : target
    delta_i    = p_i - onehot(y_i)

For a linear classifier logits = W z + b:

    dL_i/dW = delta_i outer z_i
    dL_i/db = delta_i

Therefore the exact gradient magnitudes are available without doing
one backward pass per sample:

    logit_grad_norm
        = ||delta_i||_2

    weight_grad_norm
        = ||dL_i/dW||_F
        = ||delta_i||_2 * ||z_i||_2

    bias_grad_norm
        = ||delta_i||_2

    param_grad_norm
        = sqrt(
            ||dL_i/dW||_F^2
            + ||dL_i/db||_2^2
          )

    feature_grad_norm
        = ||dL_i/dz_i||_2
        = ||W^T delta_i||_2

Additional potentially useful signals are also saved:

    loss
    true-class probability
    logit margin
    correctness
    cosine(sample gradient, split mean gradient)
        g_bar_split = (1/N) sum_j g_j

    cosine(sample gradient, same-class mean gradient)
        g_bar_class(i) = (1/N_y) sum_{j: y_j = y_i} g_j
    within-class z-score of parameter-gradient magnitude
    within-class percentile of parameter-gradient magnitude

Why the extra metrics?
----------------------
Raw gradient magnitude can still be class-dependent. The within-class
z-score/percentile removes much of that scale/class effect and may be
a better signal to later inject into a Rater loss. Gradient cosine
measures whether a sample pushes the classifier in the same direction
as the average training signal, which captures something different
from just "how large" its gradient is.

Group/background labels are NEVER used for optimization. They are only
used for diagnostics and plotting.

Persistent embedding cache
--------------------------
This code reuses the same cache as the existing Rater experiments:

    $RATER_EMBEDDING_CACHE_ROOT/
        waterbirds/
        resnet50_imagenet_pretrained/
        resolution_224/
            train_no_aug.pt
            val.pt
            test.pt

The code requires these cached files and does not recompute ResNet
features.
"""

import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .algorithm import Algorithm
from .register import register_algorithm
from utils import log


# ============================================================
# Cached embedding dataset
# ============================================================

class IndexedEmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings,
        labels,
        groups,
        attrs,
        split_name,
    ):
        self.embeddings = embeddings.float()
        self.y_array = labels.long()
        self.group_array = groups.long()
        self.confounder_array = attrs.long()
        self.split_name = str(split_name)
        self.n_classes = int(
            torch.unique(self.y_array).numel()
        )

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        return (
            idx,
            self.embeddings[idx],
            self.y_array[idx],
            self.group_array[idx],
            self.confounder_array[idx],
        )


class LinearEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
    ):
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, z):
        return self.linear(z)


# ============================================================
# Small helpers
# ============================================================

def _env_bool(name, default):
    raw = os.environ.get(
        name,
        "1" if default else "0",
    ).strip().lower()

    return raw in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_csv(name, default):
    raw = os.environ.get(
        name,
        default,
    )

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


def _safe_float(x):
    if torch.is_tensor(x):
        return float(
            x.detach().cpu().item()
        )
    return float(x)


# ============================================================
# Algorithm
# ============================================================

@register_algorithm("examine")
class Examine(Algorithm):
    """
    Ordinary linear-classifier training on frozen embeddings plus
    exact per-sample gradient diagnostics.
    """

    def __init__(self, config):
        # Keep repository behavior compatible with last-layer methods.
        config.last_layer = True
        super().__init__(config)

        if torch.cuda.is_available():
            self.device = (
                f"cuda:{self.config.gpu}"
            )
        else:
            self.device = "cpu"

        # ----------------------------------------------------
        # Shared embedding cache
        # ----------------------------------------------------
        cache_root = (
            os.environ.get(
                "EXAMINE_EMBEDDING_CACHE_ROOT",
                "",
            ).strip()
            or os.environ.get(
                "RATER_EMBEDDING_CACHE_ROOT",
                "",
            ).strip()
        )

        if not cache_root:
            raise ValueError(
                "Set EXAMINE_EMBEDDING_CACHE_ROOT or "
                "RATER_EMBEDDING_CACHE_ROOT to the shared "
                "embedding root, e.g. "
                "/content/drive/MyDrive/"
                "evidential_alignment_colab/embeddings"
            )

        self.embedding_cache_root = Path(
            cache_root
        )

        self.cache_dir = (
            self.embedding_cache_root
            / "waterbirds"
            / "resnet50_imagenet_pretrained"
            / "resolution_224"
        )

        # ----------------------------------------------------
        # Experiment controls
        # ----------------------------------------------------
        self.epochs = int(
            self.config.epoch
        )

        self.batch_size = int(
            self.config.batch_size
        )

        self.num_workers = int(
            self.config.num_workers
        )

        self.include_epoch0 = _env_bool(
            "EXAMINE_INCLUDE_EPOCH0",
            True,
        )

        self.analyze_splits = _env_csv(
            "EXAMINE_ANALYZE_SPLITS",
            "train_no_aug,val",
        )

        self.test_each_epoch = _env_bool(
            "EXAMINE_TEST_EACH_EPOCH",
            False,
        )

        if (
            self.test_each_epoch
            and "test" not in self.analyze_splits
        ):
            self.analyze_splits.append(
                "test"
            )

        self.save_per_sample = _env_bool(
            "EXAMINE_SAVE_PER_SAMPLE",
            True,
        )

        self.plot_metrics = _env_csv(
            "EXAMINE_PLOT_METRICS",
            (
                "param_grad_norm,"
                "log10_param_grad_norm,"
                "feature_grad_norm,"
                "logit_grad_norm,"
                "grad_cos_split_mean,"
                "grad_cos_class_mean,"
                "param_grad_norm_within_class_z,"
                "loss"
            ),
        )

        # ----------------------------------------------------
        # Standard optimizer hyperparameters
        # ----------------------------------------------------
        optimizer_kwargs = dict(
            getattr(
                self.config,
                "optimizer_kwargs",
                {},
            )
        )

        self.lr = float(
            optimizer_kwargs.get(
                "lr",
                1e-2,
            )
        )

        self.momentum = float(
            optimizer_kwargs.get(
                "momentum",
                0.9,
            )
        )

        self.weight_decay = float(
            optimizer_kwargs.get(
                "weight_decay",
                1e-4,
            )
        )

        self.classifier = None
        self.optimizer = None

        self.feature_dim = None
        self.n_classes = None

        # group mean history used to make trajectory plots
        self.group_epoch_rows = []

        # train-sample trajectory state
        self.train_trajectory = None

        log("=" * 96)
        log("[EXAMINE] Gradient-diagnostics mode")
        log("[EXAMINE] NO Rater, NO meta-learning, NO inner population")
        log(
            f"[EXAMINE] shared cache: {self.cache_dir}"
        )
        log(
            f"[EXAMINE] analyze each epoch: "
            f"{self.analyze_splits}"
        )
        log(
            f"[EXAMINE] optimizer: SGD "
            f"lr={self.lr}, "
            f"momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}"
        )
        log("=" * 96)

    # ========================================================
    # Cache
    # ========================================================

    def _cache_file(self, split_name):
        mapping = {
            "train": "train_no_aug.pt",
            "train_no_aug": "train_no_aug.pt",
            "val": "val.pt",
            "test": "test.pt",
        }

        if split_name not in mapping:
            raise ValueError(
                f"Unsupported EXAMINE split: {split_name}. "
                "Use train_no_aug, val, or test."
            )

        return (
            self.cache_dir
            / mapping[split_name]
        )

    def _load_cached_dataset(self, split_name):
        path = self._cache_file(
            split_name
        )

        if not path.exists():
            raise FileNotFoundError(
                "Required frozen embedding cache is missing:\n"
                f"{path}\n\n"
                "Run the shared-embedding preparation notebook first."
            )

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        required = {
            "embeddings",
            "labels",
            "groups",
            "attrs",
        }

        if not (
            isinstance(payload, dict)
            and required.issubset(
                payload.keys()
            )
        ):
            raise ValueError(
                f"Unexpected cache format in {path}."
            )

        ds = IndexedEmbeddingDataset(
            embeddings=payload[
                "embeddings"
            ],
            labels=payload[
                "labels"
            ],
            groups=payload[
                "groups"
            ],
            attrs=payload[
                "attrs"
            ],
            split_name=split_name,
        )

        log(
            f"[EXAMINE cache] {split_name}: "
            f"n={len(ds)}, "
            f"dim={ds.embeddings.shape[1]} "
            f"<- {path}"
        )

        return ds

    # ========================================================
    # Classification metrics
    # ========================================================

    @staticmethod
    def _group_name(group_id):
        waterbirds_names = {
            0: "landbird_land",
            1: "landbird_water",
            2: "waterbird_land",
            3: "waterbird_water",
        }

        return waterbirds_names.get(
            int(group_id),
            f"group_{int(group_id)}",
        )

    @staticmethod
    def _classification_summary(
        pred,
        y,
        groups,
        losses,
    ):
        pred = np.asarray(pred)
        y = np.asarray(y)
        groups = np.asarray(groups)
        losses = np.asarray(losses)

        result = {
            "loss": float(
                losses.mean()
            ),
            "accuracy": float(
                (pred == y).mean()
            ),
            "group_accuracy": {},
        }

        for gid in sorted(
            np.unique(groups)
        ):
            mask = groups == gid

            result[
                "group_accuracy"
            ][int(gid)] = float(
                (
                    pred[mask]
                    == y[mask]
                ).mean()
            )

        result[
            "worst_group_accuracy"
        ] = float(
            min(
                result[
                    "group_accuracy"
                ].values()
            )
        )

        return result

    @staticmethod
    def _format_groups(summary):
        return ", ".join(
            f"g{gid}="
            f"{100.0 * value:.2f}%"
            for gid, value
            in summary[
                "group_accuracy"
            ].items()
        )

    # ========================================================
    # Exact per-sample gradient diagnostics
    # ========================================================

    @torch.no_grad()
    def _collect_basic_gradient_quantities(
        self,
        dataset,
    ):
        """
        First pass:
          - predictions / losses
          - exact gradient norms
          - accumulate split-mean and class-mean parameter gradients
        """

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        n = len(dataset)
        k = self.n_classes
        d = self.feature_dim

        # per-sample arrays
        labels = np.empty(
            n,
            dtype=np.int64,
        )
        groups = np.empty(
            n,
            dtype=np.int64,
        )
        attrs = np.empty(
            n,
            dtype=np.int64,
        )

        pred = np.empty(
            n,
            dtype=np.int64,
        )
        correct = np.empty(
            n,
            dtype=np.int64,
        )

        loss_arr = np.empty(
            n,
            dtype=np.float64,
        )
        true_prob = np.empty(
            n,
            dtype=np.float64,
        )
        margin = np.empty(
            n,
            dtype=np.float64,
        )
        embedding_norm = np.empty(
            n,
            dtype=np.float64,
        )

        logit_grad_norm = np.empty(
            n,
            dtype=np.float64,
        )
        weight_grad_norm = np.empty(
            n,
            dtype=np.float64,
        )
        bias_grad_norm = np.empty(
            n,
            dtype=np.float64,
        )
        param_grad_norm = np.empty(
            n,
            dtype=np.float64,
        )
        feature_grad_norm = np.empty(
            n,
            dtype=np.float64,
        )

        # Gradient sums over the whole split.
        grad_w_sum = torch.zeros(
            k,
            d,
            dtype=torch.float64,
            device=self.device,
        )

        grad_b_sum = torch.zeros(
            k,
            dtype=torch.float64,
            device=self.device,
        )

        # Same-class mean-gradient accumulators.
        class_grad_w_sum = {
            c: torch.zeros(
                k,
                d,
                dtype=torch.float64,
                device=self.device,
            )
            for c in range(k)
        }

        class_grad_b_sum = {
            c: torch.zeros(
                k,
                dtype=torch.float64,
                device=self.device,
            )
            for c in range(k)
        }

        class_count = {
            c: 0
            for c in range(k)
        }

        self.classifier.eval()

        W = (
            self.classifier
            .linear
            .weight
            .detach()
        )

        for (
            indices,
            z,
            y,
            g,
            a,
        ) in loader:

            indices_np = (
                indices.numpy()
            )

            z = z.to(
                self.device,
                non_blocking=True,
            )

            y_device = y.to(
                self.device,
                non_blocking=True,
            )

            logits = self.classifier(
                z
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            onehot = F.one_hot(
                y_device,
                num_classes=k,
            ).to(
                probs.dtype
            )

            # Exact d CE / d logits for each sample.
            delta = probs - onehot

            losses = F.cross_entropy(
                logits,
                y_device,
                reduction="none",
            )

            predictions = logits.argmax(
                dim=1
            )

            # -----------------------------------------------
            # Confidence / margin
            # -----------------------------------------------
            row = torch.arange(
                y_device.shape[0],
                device=self.device,
            )

            p_true = probs[
                row,
                y_device,
            ]

            true_logits = logits[
                row,
                y_device,
            ]

            masked_logits = logits.clone()
            masked_logits[
                row,
                y_device,
            ] = -torch.inf

            max_other_logit = (
                masked_logits.max(
                    dim=1
                ).values
            )

            logit_margin = (
                true_logits
                - max_other_logit
            )

            # -----------------------------------------------
            # Gradient magnitude formulas
            # -----------------------------------------------
            delta_norm = torch.linalg.vector_norm(
                delta,
                ord=2,
                dim=1,
            )

            z_norm = torch.linalg.vector_norm(
                z,
                ord=2,
                dim=1,
            )

            # || delta outer z ||_F
            grad_w_norm = (
                delta_norm
                * z_norm
            )

            grad_b_norm = delta_norm

            full_param_norm = torch.sqrt(
                grad_w_norm.pow(2)
                + grad_b_norm.pow(2)
            )

            # dL/dz = W^T delta
            grad_z = (
                delta
                @ W
            )

            grad_z_norm = (
                torch.linalg.vector_norm(
                    grad_z,
                    ord=2,
                    dim=1,
                )
            )

            # -----------------------------------------------
            # Accumulate exact parameter gradients
            # -----------------------------------------------
            delta64 = delta.to(
                torch.float64
            )

            z64 = z.to(
                torch.float64
            )

            grad_w_sum += (
                delta64.t()
                @ z64
            )

            grad_b_sum += (
                delta64.sum(
                    dim=0
                )
            )

            for c in range(k):
                class_mask = (
                    y_device == c
                )

                count_c = int(
                    class_mask.sum().item()
                )

                if count_c == 0:
                    continue

                d_c = delta64[
                    class_mask
                ]

                z_c = z64[
                    class_mask
                ]

                class_grad_w_sum[c] += (
                    d_c.t()
                    @ z_c
                )

                class_grad_b_sum[c] += (
                    d_c.sum(
                        dim=0
                    )
                )

                class_count[c] += (
                    count_c
                )

            # -----------------------------------------------
            # Save batch arrays by original sample index
            # -----------------------------------------------
            labels[
                indices_np
            ] = y.numpy()

            groups[
                indices_np
            ] = g.numpy()

            attrs[
                indices_np
            ] = a.numpy()

            pred[
                indices_np
            ] = (
                predictions.cpu().numpy()
            )

            correct[
                indices_np
            ] = (
                predictions
                .eq(y_device)
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            loss_arr[
                indices_np
            ] = (
                losses.cpu().numpy()
            )

            true_prob[
                indices_np
            ] = (
                p_true.cpu().numpy()
            )

            margin[
                indices_np
            ] = (
                logit_margin.cpu().numpy()
            )

            embedding_norm[
                indices_np
            ] = (
                z_norm.cpu().numpy()
            )

            logit_grad_norm[
                indices_np
            ] = (
                delta_norm.cpu().numpy()
            )

            weight_grad_norm[
                indices_np
            ] = (
                grad_w_norm.cpu().numpy()
            )

            bias_grad_norm[
                indices_np
            ] = (
                grad_b_norm.cpu().numpy()
            )

            param_grad_norm[
                indices_np
            ] = (
                full_param_norm.cpu().numpy()
            )

            feature_grad_norm[
                indices_np
            ] = (
                grad_z_norm.cpu().numpy()
            )

        split_grad_w_mean = (
            grad_w_sum
            / float(n)
        )

        split_grad_b_mean = (
            grad_b_sum
            / float(n)
        )

        class_grad_means = {}

        for c in range(k):
            count_c = max(
                class_count[c],
                1,
            )

            class_grad_means[c] = (
                class_grad_w_sum[c]
                / float(count_c),
                class_grad_b_sum[c]
                / float(count_c),
            )

        return {
            "labels": labels,
            "groups": groups,
            "attrs": attrs,
            "pred": pred,
            "correct": correct,
            "loss": loss_arr,
            "true_prob": true_prob,
            "margin": margin,
            "embedding_norm": embedding_norm,
            "logit_grad_norm": logit_grad_norm,
            "weight_grad_norm": weight_grad_norm,
            "bias_grad_norm": bias_grad_norm,
            "param_grad_norm": param_grad_norm,
            "feature_grad_norm": feature_grad_norm,
            "split_grad_w_mean": split_grad_w_mean,
            "split_grad_b_mean": split_grad_b_mean,
            "class_grad_means": class_grad_means,
        }

    @torch.no_grad()
    def _add_gradient_alignment_metrics(
        self,
        dataset,
        metrics,
    ):
        """
        Second pass:

          A) SPLIT-MEAN alignment
             g_bar_split = (1/N) sum_j g_j

             grad_cos_split_mean[i]
                 = cos(g_i, g_bar_split)

          B) SAME-CLASS-MEAN alignment
             g_bar_class(i)
                 = (1/N_y) sum_{j: y_j = y_i} g_j

             grad_cos_class_mean[i]
                 = cos(g_i, g_bar_class(i))

        Both are diagnostics only. Neither one changes training.
        """

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        n = len(dataset)
        k = self.n_classes

        dot_split = np.empty(
            n,
            dtype=np.float64,
        )

        cos_split = np.empty(
            n,
            dtype=np.float64,
        )

        dot_class = np.empty(
            n,
            dtype=np.float64,
        )

        cos_class = np.empty(
            n,
            dtype=np.float64,
        )

        mean_w = metrics[
            "split_grad_w_mean"
        ]

        mean_b = metrics[
            "split_grad_b_mean"
        ]

        mean_grad_norm = torch.sqrt(
            mean_w.pow(2).sum()
            + mean_b.pow(2).sum()
            + 1e-24
        )

        self.classifier.eval()

        for (
            indices,
            z,
            y,
            g,
            a,
        ) in loader:

            indices_np = (
                indices.numpy()
            )

            z = z.to(
                self.device,
                non_blocking=True,
            )

            y_device = y.to(
                self.device,
                non_blocking=True,
            )

            logits = self.classifier(
                z
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            onehot = F.one_hot(
                y_device,
                num_classes=k,
            ).to(
                probs.dtype
            )

            delta = (
                probs - onehot
            ).to(
                torch.float64
            )

            z64 = z.to(
                torch.float64
            )

            # For sample gradient:
            #
            # <g_i, g_mean>
            # = sum_k delta_ik *
            #   ( <mean_W[k], z_i> + mean_b[k] )
            split_projection = (
                z64
                @ mean_w.t()
                + mean_b.unsqueeze(0)
            )

            sample_dot_split = (
                delta
                * split_projection
            ).sum(
                dim=1
            )

            sample_norm = torch.as_tensor(
                metrics[
                    "param_grad_norm"
                ][
                    indices_np
                ],
                dtype=torch.float64,
                device=self.device,
            )

            sample_cos_split = (
                sample_dot_split
                / (
                    sample_norm
                    * mean_grad_norm
                    + 1e-24
                )
            )

            dot_split[
                indices_np
            ] = (
                sample_dot_split
                .cpu()
                .numpy()
            )

            cos_split[
                indices_np
            ] = (
                sample_cos_split
                .cpu()
                .numpy()
            )

            # Same-class gradient alignment.
            batch_dot_class = torch.empty(
                y_device.shape[0],
                dtype=torch.float64,
                device=self.device,
            )

            batch_cos_class = torch.empty(
                y_device.shape[0],
                dtype=torch.float64,
                device=self.device,
            )

            for c in range(k):
                mask = (
                    y_device == c
                )

                if not bool(
                    mask.any()
                ):
                    continue

                class_w, class_b = (
                    metrics[
                        "class_grad_means"
                    ][c]
                )

                class_norm = torch.sqrt(
                    class_w.pow(2).sum()
                    + class_b.pow(2).sum()
                    + 1e-24
                )

                z_c = z64[
                    mask
                ]

                delta_c = delta[
                    mask
                ]

                projection_c = (
                    z_c
                    @ class_w.t()
                    + class_b.unsqueeze(0)
                )

                dot_c = (
                    delta_c
                    * projection_c
                ).sum(
                    dim=1
                )

                norm_c = sample_norm[
                    mask
                ]

                cos_c = (
                    dot_c
                    / (
                        norm_c
                        * class_norm
                        + 1e-24
                    )
                )

                batch_dot_class[
                    mask
                ] = dot_c

                batch_cos_class[
                    mask
                ] = cos_c

            dot_class[
                indices_np
            ] = (
                batch_dot_class
                .cpu()
                .numpy()
            )

            cos_class[
                indices_np
            ] = (
                batch_cos_class
                .cpu()
                .numpy()
            )

        metrics[
            "grad_dot_split_mean"
        ] = dot_split

        metrics[
            "grad_cos_split_mean"
        ] = cos_split

        metrics[
            "grad_dot_class_mean"
        ] = dot_class

        metrics[
            "grad_cos_class_mean"
        ] = cos_class

        return metrics

    @staticmethod
    def _within_class_standardize(
        values,
        labels,
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        zscore = np.zeros_like(
            values,
            dtype=np.float64,
        )

        percentile = np.zeros_like(
            values,
            dtype=np.float64,
        )

        for c in np.unique(
            labels
        ):
            mask = (
                labels == c
            )

            vals = values[
                mask
            ]

            mean = vals.mean()
            std = vals.std()

            zscore[
                mask
            ] = (
                vals - mean
            ) / (
                std + 1e-12
            )

            # Percentile rank in [0, 1].
            order = np.argsort(
                vals,
                kind="mergesort",
            )

            ranks = np.empty(
                len(vals),
                dtype=np.float64,
            )

            if len(vals) == 1:
                ranks[0] = 0.5
            else:
                ranks[
                    order
                ] = np.linspace(
                    0.0,
                    1.0,
                    len(vals),
                )

            percentile[
                mask
            ] = ranks

        return zscore, percentile

    @torch.no_grad()
    def _compute_per_sample_metrics(
        self,
        dataset,
    ):
        metrics = (
            self._collect_basic_gradient_quantities(
                dataset
            )
        )

        metrics = (
            self._add_gradient_alignment_metrics(
                dataset,
                metrics,
            )
        )

        (
            grad_zscore,
            grad_percentile,
        ) = self._within_class_standardize(
            metrics[
                "param_grad_norm"
            ],
            metrics[
                "labels"
            ],
        )

        metrics[
            "param_grad_norm_within_class_z"
        ] = grad_zscore

        metrics[
            "param_grad_norm_within_class_percentile"
        ] = grad_percentile

        (
            feature_zscore,
            feature_percentile,
        ) = self._within_class_standardize(
            metrics[
                "feature_grad_norm"
            ],
            metrics[
                "labels"
            ],
        )

        metrics[
            "feature_grad_norm_within_class_z"
        ] = feature_zscore

        metrics[
            "feature_grad_norm_within_class_percentile"
        ] = feature_percentile

        metrics[
            "log10_param_grad_norm"
        ] = np.log10(
            metrics[
                "param_grad_norm"
            ]
            + 1e-12
        )

        # Remove large matrix objects before returning/saving.
        metrics.pop(
            "split_grad_w_mean",
            None,
        )

        metrics.pop(
            "split_grad_b_mean",
            None,
        )

        metrics.pop(
            "class_grad_means",
            None,
        )

        return metrics

    # ========================================================
    # Output tables / plots
    # ========================================================

    @staticmethod
    def _metric_columns():
        return [
            "loss",
            "true_prob",
            "margin",
            "embedding_norm",
            "logit_grad_norm",
            "weight_grad_norm",
            "bias_grad_norm",
            "param_grad_norm",
            "log10_param_grad_norm",
            "feature_grad_norm",
            "grad_dot_split_mean",
            "grad_cos_split_mean",
            "grad_dot_class_mean",
            "grad_cos_class_mean",
            "param_grad_norm_within_class_z",
            "param_grad_norm_within_class_percentile",
            "feature_grad_norm_within_class_z",
            "feature_grad_norm_within_class_percentile",
        ]

    def _save_per_sample_csv(
        self,
        output_dir,
        split_name,
        epoch,
        metrics,
    ):
        if not self.save_per_sample:
            return

        table_dir = (
            Path(output_dir)
            / "gradient_diagnostics"
            / f"epoch_{epoch:03d}"
            / "tables"
        )

        table_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            table_dir
            / f"{split_name}_per_sample.csv"
        )

        columns = (
            [
                "sample_index",
                "label",
                "group",
                "attr",
                "prediction",
                "correct",
            ]
            + self._metric_columns()
        )

        with path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.writer(
                f
            )

            writer.writerow(
                columns
            )

            n = len(
                metrics[
                    "labels"
                ]
            )

            for i in range(n):
                row = [
                    i,
                    int(
                        metrics[
                            "labels"
                        ][i]
                    ),
                    int(
                        metrics[
                            "groups"
                        ][i]
                    ),
                    int(
                        metrics[
                            "attrs"
                        ][i]
                    ),
                    int(
                        metrics[
                            "pred"
                        ][i]
                    ),
                    int(
                        metrics[
                            "correct"
                        ][i]
                    ),
                ]

                for name in self._metric_columns():
                    row.append(
                        float(
                            metrics[
                                name
                            ][i]
                        )
                    )

                writer.writerow(
                    row
                )

    def _append_group_epoch_summary(
        self,
        split_name,
        epoch,
        metrics,
    ):
        groups = metrics[
            "groups"
        ]

        for metric_name in self._metric_columns():

            values = np.asarray(
                metrics[
                    metric_name
                ],
                dtype=np.float64,
            )

            for gid in sorted(
                np.unique(
                    groups
                )
            ):
                mask = (
                    groups == gid
                )

                vals = values[
                    mask
                ]

                self.group_epoch_rows.append(
                    {
                        "epoch": int(
                            epoch
                        ),
                        "split": str(
                            split_name
                        ),
                        "group": int(
                            gid
                        ),
                        "group_name": (
                            self._group_name(
                                gid
                            )
                        ),
                        "metric": str(
                            metric_name
                        ),
                        "n": int(
                            mask.sum()
                        ),
                        "mean": float(
                            vals.mean()
                        ),
                        "std": float(
                            vals.std()
                        ),
                        "median": float(
                            np.median(
                                vals
                            )
                        ),
                        "q25": float(
                            np.quantile(
                                vals,
                                0.25,
                            )
                        ),
                        "q75": float(
                            np.quantile(
                                vals,
                                0.75,
                            )
                        ),
                    }
                )

    def _write_group_epoch_summary(
        self,
        output_dir,
    ):
        path = (
            Path(output_dir)
            / "gradient_diagnostics"
            / "group_epoch_summary.csv"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.group_epoch_rows:
            return

        with path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    self.group_epoch_rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                self.group_epoch_rows
            )

    def _save_histogram(
        self,
        output_dir,
        split_name,
        epoch,
        metric_name,
        metrics,
    ):
        if metric_name not in metrics:
            return

        values = np.asarray(
            metrics[
                metric_name
            ],
            dtype=np.float64,
        )

        groups = np.asarray(
            metrics[
                "groups"
            ],
            dtype=np.int64,
        )

        finite = np.isfinite(
            values
        )

        if not bool(
            finite.any()
        ):
            return

        values_f = values[
            finite
        ]

        # Robust bounds so a single outlier does not flatten the plot.
        low = float(
            np.quantile(
                values_f,
                0.005,
            )
        )

        high = float(
            np.quantile(
                values_f,
                0.995,
            )
        )

        if not math.isfinite(
            low
        ):
            low = float(
                np.min(
                    values_f
                )
            )

        if not math.isfinite(
            high
        ):
            high = float(
                np.max(
                    values_f
                )
            )

        if abs(
            high - low
        ) < 1e-12:
            low -= 0.5
            high += 0.5

        bins = np.linspace(
            low,
            high,
            45,
        )

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        for gid in sorted(
            np.unique(
                groups
            )
        ):
            mask = (
                groups == gid
            ) & finite

            vals = values[
                mask
            ]

            # clip only for plotting; original CSV retains exact values
            vals_plot = np.clip(
                vals,
                low,
                high,
            )

            ax.hist(
                vals_plot,
                bins=bins,
                density=True,
                alpha=0.42,
                label=(
                    f"g{gid}: "
                    f"{self._group_name(gid)} "
                    f"(n={mask.sum()})"
                ),
            )

        ax.set_title(
            f"{metric_name} by Waterbirds group\n"
            f"{split_name} | epoch {epoch}"
        )

        ax.set_xlabel(
            metric_name
        )

        ax.set_ylabel(
            "Density"
        )

        ax.legend(
            fontsize=9
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.3,
        )

        fig.tight_layout()

        plot_dir = (
            Path(output_dir)
            / "gradient_diagnostics"
            / f"epoch_{epoch:03d}"
            / "plots"
            / split_name
        )

        plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            plot_dir
            / f"{metric_name}_hist.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    def _save_gradient_boxplot(
        self,
        output_dir,
        split_name,
        epoch,
        metrics,
    ):
        values = np.asarray(
            metrics[
                "param_grad_norm"
            ],
            dtype=np.float64,
        )

        groups = np.asarray(
            metrics[
                "groups"
            ],
            dtype=np.int64,
        )

        unique_groups = sorted(
            np.unique(
                groups
            )
        )

        data = [
            values[
                groups == gid
            ]
            for gid in unique_groups
        ]

        labels = [
            f"g{gid}\n"
            f"{self._group_name(gid)}"
            for gid in unique_groups
        ]

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
        )

        ax.set_title(
            "Per-sample classifier parameter-gradient magnitude\n"
            f"{split_name} | epoch {epoch}"
        )

        ax.set_ylabel(
            "|| d CE_i / d(W,b) ||_2"
        )

        ax.grid(
            True,
            axis="y",
            linestyle=":",
            alpha=0.3,
        )

        fig.tight_layout()

        plot_dir = (
            Path(output_dir)
            / "gradient_diagnostics"
            / f"epoch_{epoch:03d}"
            / "plots"
            / split_name
        )

        plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            plot_dir
            / "param_grad_norm_boxplot.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    def _save_epoch_plots(
        self,
        output_dir,
        split_name,
        epoch,
        metrics,
    ):
        for metric_name in self.plot_metrics:
            self._save_histogram(
                output_dir,
                split_name,
                epoch,
                metric_name,
                metrics,
            )

        self._save_gradient_boxplot(
            output_dir,
            split_name,
            epoch,
            metrics,
        )

    def _save_group_trajectory_plots(
        self,
        output_dir,
    ):
        if not self.group_epoch_rows:
            return

        plot_metrics = [
            "param_grad_norm",
            "feature_grad_norm",
            "logit_grad_norm",
            "grad_cos_split_mean",
            "grad_cos_class_mean",
            "param_grad_norm_within_class_z",
            "loss",
        ]

        trajectory_dir = (
            Path(output_dir)
            / "gradient_diagnostics"
            / "trajectory_plots"
        )

        trajectory_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for split_name in sorted(
            {
                row["split"]
                for row
                in self.group_epoch_rows
            }
        ):
            for metric_name in plot_metrics:

                rows = [
                    row
                    for row
                    in self.group_epoch_rows
                    if (
                        row[
                            "split"
                        ]
                        == split_name
                        and row[
                            "metric"
                        ]
                        == metric_name
                    )
                ]

                if not rows:
                    continue

                fig, ax = plt.subplots(
                    figsize=(11, 7)
                )

                groups = sorted(
                    {
                        row["group"]
                        for row in rows
                    }
                )

                for gid in groups:

                    group_rows = sorted(
                        [
                            row
                            for row
                            in rows
                            if row[
                                "group"
                            ]
                            == gid
                        ],
                        key=lambda row: row[
                            "epoch"
                        ],
                    )

                    ax.plot(
                        [
                            row[
                                "epoch"
                            ]
                            for row
                            in group_rows
                        ],
                        [
                            row[
                                "mean"
                            ]
                            for row
                            in group_rows
                        ],
                        marker="o",
                        markersize=3,
                        linewidth=1.5,
                        label=(
                            f"g{gid}: "
                            f"{self._group_name(gid)}"
                        ),
                    )

                ax.set_title(
                    f"Mean {metric_name} across epochs\n"
                    f"{split_name}"
                )

                ax.set_xlabel(
                    "Epoch"
                )

                ax.set_ylabel(
                    f"Group mean {metric_name}"
                )

                ax.legend(
                    fontsize=9
                )

                ax.grid(
                    True,
                    linestyle=":",
                    alpha=0.3,
                )

                fig.tight_layout()

                fig.savefig(
                    trajectory_dir
                    / (
                        f"{split_name}_"
                        f"{metric_name}_"
                        f"group_means.png"
                    ),
                    dpi=160,
                    bbox_inches="tight",
                )

                plt.close(
                    fig
                )

    # ========================================================
    # Training-sample trajectory statistics
    # ========================================================

    def _update_train_trajectory(
        self,
        epoch,
        metrics,
    ):
        n = len(
            metrics[
                "labels"
            ]
        )

        if self.train_trajectory is None:
            self.train_trajectory = {
                "sum_param_grad_norm": np.zeros(
                    n,
                    dtype=np.float64,
                ),
                "max_param_grad_norm": np.full(
                    n,
                    -np.inf,
                    dtype=np.float64,
                ),
                "sum_feature_grad_norm": np.zeros(
                    n,
                    dtype=np.float64,
                ),
                "misclassified_count": np.zeros(
                    n,
                    dtype=np.int64,
                ),
                "high_within_class_grad_count": np.zeros(
                    n,
                    dtype=np.int64,
                ),
                "num_recorded_epochs": 0,
                "last_param_grad_norm": np.zeros(
                    n,
                    dtype=np.float64,
                ),
            }

        tr = self.train_trajectory

        tr[
            "sum_param_grad_norm"
        ] += metrics[
            "param_grad_norm"
        ]

        tr[
            "max_param_grad_norm"
        ] = np.maximum(
            tr[
                "max_param_grad_norm"
            ],
            metrics[
                "param_grad_norm"
            ],
        )

        tr[
            "sum_feature_grad_norm"
        ] += metrics[
            "feature_grad_norm"
        ]

        tr[
            "misclassified_count"
        ] += (
            1
            - metrics[
                "correct"
            ].astype(
                np.int64
            )
        )

        tr[
            "high_within_class_grad_count"
        ] += (
            metrics[
                "param_grad_norm_within_class_percentile"
            ]
            >= 0.90
        ).astype(
            np.int64
        )

        tr[
            "last_param_grad_norm"
        ] = metrics[
            "param_grad_norm"
        ].copy()

        tr[
            "num_recorded_epochs"
        ] += 1

    def _save_train_trajectory_summary(
        self,
        output_dir,
        train_dataset,
    ):
        if self.train_trajectory is None:
            return

        tr = self.train_trajectory

        n_epochs = max(
            int(
                tr[
                    "num_recorded_epochs"
                ]
            ),
            1,
        )

        path = (
            Path(output_dir)
            / "gradient_diagnostics"
            / "train_sample_trajectory_summary.csv"
        )

        with path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.writer(
                f
            )

            writer.writerow(
                [
                    "sample_index",
                    "label",
                    "group",
                    "attr",
                    "mean_param_grad_norm",
                    "max_param_grad_norm",
                    "last_param_grad_norm",
                    "mean_feature_grad_norm",
                    "misclassified_epochs",
                    "misclassified_fraction",
                    "top10pct_within_class_grad_epochs",
                    "top10pct_within_class_grad_fraction",
                ]
            )

            for i in range(
                len(
                    train_dataset
                )
            ):
                writer.writerow(
                    [
                        i,
                        int(
                            train_dataset
                            .y_array[i]
                        ),
                        int(
                            train_dataset
                            .group_array[i]
                        ),
                        int(
                            train_dataset
                            .confounder_array[i]
                        ),
                        float(
                            tr[
                                "sum_param_grad_norm"
                            ][i]
                            / n_epochs
                        ),
                        float(
                            tr[
                                "max_param_grad_norm"
                            ][i]
                        ),
                        float(
                            tr[
                                "last_param_grad_norm"
                            ][i]
                        ),
                        float(
                            tr[
                                "sum_feature_grad_norm"
                            ][i]
                            / n_epochs
                        ),
                        int(
                            tr[
                                "misclassified_count"
                            ][i]
                        ),
                        float(
                            tr[
                                "misclassified_count"
                            ][i]
                            / n_epochs
                        ),
                        int(
                            tr[
                                "high_within_class_grad_count"
                            ][i]
                        ),
                        float(
                            tr[
                                "high_within_class_grad_count"
                            ][i]
                            / n_epochs
                        ),
                    ]
                )

    # ========================================================
    # Analysis wrapper
    # ========================================================

    @torch.no_grad()
    def _analyze_split(
        self,
        output_dir,
        split_name,
        epoch,
        dataset,
        update_train_trajectory=False,
    ):
        metrics = (
            self._compute_per_sample_metrics(
                dataset
            )
        )

        summary = (
            self._classification_summary(
                metrics[
                    "pred"
                ],
                metrics[
                    "labels"
                ],
                metrics[
                    "groups"
                ],
                metrics[
                    "loss"
                ],
            )
        )

        mean_grad = float(
            np.mean(
                metrics[
                    "param_grad_norm"
                ]
            )
        )

        log(
            f"[EXAMINE epoch {epoch:03d} / {split_name}] "
            f"loss={summary['loss']:.6f}, "
            f"acc={100.0 * summary['accuracy']:.2f}%, "
            f"WGA={100.0 * summary['worst_group_accuracy']:.2f}%, "
            f"mean_param_grad_norm={mean_grad:.6f}"
        )

        log(
            f"[EXAMINE epoch {epoch:03d} / {split_name} groups] "
            f"{self._format_groups(summary)}"
        )

        # Print group gradient means prominently.
        for gid in sorted(
            np.unique(
                metrics[
                    "groups"
                ]
            )
        ):
            mask = (
                metrics[
                    "groups"
                ]
                == gid
            )

            log(
                f"  g{gid} "
                f"({self._group_name(gid)}): "
                f"param_grad_mean="
                f"{metrics['param_grad_norm'][mask].mean():.6f}, "
                f"feature_grad_mean="
                f"{metrics['feature_grad_norm'][mask].mean():.6f}, "
                f"logit_grad_mean="
                f"{metrics['logit_grad_norm'][mask].mean():.6f}, "
                f"mean_cos_to_split_grad="
                f"{metrics['grad_cos_split_mean'][mask].mean():.6f}, "
                f"mean_cos_to_same_class_grad="
                f"{metrics['grad_cos_class_mean'][mask].mean():.6f}"
            )

        self._save_per_sample_csv(
            output_dir,
            split_name,
            epoch,
            metrics,
        )

        self._append_group_epoch_summary(
            split_name,
            epoch,
            metrics,
        )

        self._write_group_epoch_summary(
            output_dir
        )

        self._save_epoch_plots(
            output_dir,
            split_name,
            epoch,
            metrics,
        )

        if update_train_trajectory:
            self._update_train_trajectory(
                epoch,
                metrics,
            )

        return (
            metrics,
            summary,
        )

    # ========================================================
    # Ordinary classifier training
    # ========================================================

    def _train_one_epoch(
        self,
        train_loader,
    ):
        self.classifier.train()

        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for (
            indices,
            z,
            y,
            g,
            a,
        ) in train_loader:

            z = z.to(
                self.device,
                non_blocking=True,
            )

            y = y.to(
                self.device,
                non_blocking=True,
            )

            logits = self.classifier(
                z
            )

            # Ordinary unweighted mean CE.
            loss = F.cross_entropy(
                logits,
                y,
                reduction="mean",
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_n = y.numel()

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            total_correct += (
                logits.argmax(
                    dim=1
                )
                .eq(y)
                .sum()
                .item()
            )

            total_n += (
                batch_n
            )

        return {
            "loss": (
                total_loss
                / max(
                    total_n,
                    1,
                )
            ),
            "accuracy": (
                total_correct
                / max(
                    total_n,
                    1,
                )
            ),
        }

    # ========================================================
    # Public API expected by main.py
    # ========================================================

    def train(
        self,
        output_dir,
        split="train",
    ):
        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load frozen cached embeddings
        # ----------------------------------------------------
        datasets = {
            "train_no_aug": (
                self._load_cached_dataset(
                    "train_no_aug"
                )
            ),
            "val": (
                self._load_cached_dataset(
                    "val"
                )
            ),
            "test": (
                self._load_cached_dataset(
                    "test"
                )
            ),
        }

        train_dataset = datasets[
            "train_no_aug"
        ]

        self.feature_dim = int(
            train_dataset
            .embeddings
            .shape[1]
        )

        self.n_classes = int(
            train_dataset.n_classes
        )

        # ----------------------------------------------------
        # Fresh linear classifier
        # ----------------------------------------------------
        torch.manual_seed(
            int(
                getattr(
                    self.config,
                    "seed",
                    0,
                )
            )
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                int(
                    getattr(
                        self.config,
                        "seed",
                        0,
                    )
                )
            )

        self.classifier = (
            LinearEmbeddingClassifier(
                input_dim=self.feature_dim,
                num_classes=self.n_classes,
            )
            .to(
                self.device
            )
        )

        self.optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        generator = torch.Generator()
        generator.manual_seed(
            int(
                getattr(
                    self.config,
                    "seed",
                    0,
                )
            )
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
        )

        # ----------------------------------------------------
        # Run configuration
        # ----------------------------------------------------
        run_config = {
            "algorithm": "examine",
            "purpose": (
                "frozen-embedding "
                "per-sample gradient diagnostics"
            ),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
            "include_epoch0": (
                self.include_epoch0
            ),
            "analyze_splits": (
                self.analyze_splits
            ),
            "test_each_epoch": (
                self.test_each_epoch
            ),
            "plot_metrics": (
                self.plot_metrics
            ),
            "embedding_cache": str(
                self.cache_dir
            ),
        }

        with (
            output_dir
            / "examine_gradient_config.json"
        ).open(
            "w"
        ) as f:
            json.dump(
                run_config,
                f,
                indent=2,
            )

        history_path = (
            output_dir
            / "classifier_history.csv"
        )

        history_rows = []

        best_val_accuracy = (
            -float("inf")
        )

        best_epoch = None

        best_path = (
            output_dir
            / "best_val_accuracy_classifier.pt"
        )

        # ----------------------------------------------------
        # Epoch 0 diagnostics
        # ----------------------------------------------------
        if self.include_epoch0:

            for split_name in self.analyze_splits:

                if (
                    split_name == "test"
                    and not self.test_each_epoch
                ):
                    continue

                if split_name not in datasets:
                    continue

                self._analyze_split(
                    output_dir=output_dir,
                    split_name=split_name,
                    epoch=0,
                    dataset=datasets[
                        split_name
                    ],
                    update_train_trajectory=False,
                )

        # ----------------------------------------------------
        # Train ordinary classifier
        # ----------------------------------------------------
        for epoch in range(
            1,
            self.epochs + 1,
        ):

            train_online = (
                self._train_one_epoch(
                    train_loader
                )
            )

            epoch_summaries = {}

            for split_name in self.analyze_splits:

                if (
                    split_name == "test"
                    and not self.test_each_epoch
                ):
                    continue

                if split_name not in datasets:
                    continue

                (
                    metrics,
                    summary,
                ) = self._analyze_split(
                    output_dir=output_dir,
                    split_name=split_name,
                    epoch=epoch,
                    dataset=datasets[
                        split_name
                    ],
                    update_train_trajectory=(
                        split_name
                        == "train_no_aug"
                    ),
                )

                epoch_summaries[
                    split_name
                ] = summary

            val_summary = (
                epoch_summaries.get(
                    "val"
                )
            )

            if val_summary is None:
                # Always compute val once for model selection.
                (
                    _,
                    val_summary,
                ) = self._analyze_split(
                    output_dir=output_dir,
                    split_name="val",
                    epoch=epoch,
                    dataset=datasets[
                        "val"
                    ],
                    update_train_trajectory=False,
                )

            if (
                val_summary[
                    "accuracy"
                ]
                > best_val_accuracy
            ):
                best_val_accuracy = float(
                    val_summary[
                        "accuracy"
                    ]
                )

                best_epoch = int(
                    epoch
                )

                torch.save(
                    {
                        "classifier_sd": (
                            self.classifier
                            .state_dict()
                        ),
                        "epoch": (
                            epoch
                        ),
                        "val_accuracy": (
                            best_val_accuracy
                        ),
                        "feature_dim": (
                            self.feature_dim
                        ),
                        "n_classes": (
                            self.n_classes
                        ),
                        "run_config": (
                            run_config
                        ),
                    },
                    best_path,
                )

            train_summary = (
                epoch_summaries.get(
                    "train_no_aug"
                )
            )

            history_rows.append(
                {
                    "epoch": epoch,
                    "online_train_loss": (
                        train_online[
                            "loss"
                        ]
                    ),
                    "online_train_accuracy": (
                        train_online[
                            "accuracy"
                        ]
                    ),
                    "train_eval_loss": (
                        train_summary[
                            "loss"
                        ]
                        if train_summary
                        else float("nan")
                    ),
                    "train_eval_accuracy": (
                        train_summary[
                            "accuracy"
                        ]
                        if train_summary
                        else float("nan")
                    ),
                    "train_wga": (
                        train_summary[
                            "worst_group_accuracy"
                        ]
                        if train_summary
                        else float("nan")
                    ),
                    "val_loss": (
                        val_summary[
                            "loss"
                        ]
                    ),
                    "val_accuracy": (
                        val_summary[
                            "accuracy"
                        ]
                    ),
                    "val_wga": (
                        val_summary[
                            "worst_group_accuracy"
                        ]
                    ),
                }
            )

            with history_path.open(
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        history_rows[
                            0
                        ].keys()
                    ),
                )

                writer.writeheader()
                writer.writerows(
                    history_rows
                )

            log(
                f"[EXAMINE classifier epoch {epoch:03d}] "
                f"train_online_loss="
                f"{train_online['loss']:.6f}, "
                f"train_online_acc="
                f"{100.0 * train_online['accuracy']:.2f}%, "
                f"val_acc="
                f"{100.0 * val_summary['accuracy']:.2f}%, "
                f"val_WGA="
                f"{100.0 * val_summary['worst_group_accuracy']:.2f}%"
            )

        # ----------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------
        final_path = (
            output_dir
            / "final_classifier.pt"
        )

        torch.save(
            {
                "classifier_sd": (
                    self.classifier
                    .state_dict()
                ),
                "epoch": (
                    self.epochs
                ),
                "feature_dim": (
                    self.feature_dim
                ),
                "n_classes": (
                    self.n_classes
                ),
                "run_config": (
                    run_config
                ),
            },
            final_path,
        )

        # ----------------------------------------------------
        # Final TEST analysis only.
        # No test-based model selection.
        # ----------------------------------------------------
        (
            _,
            final_test_summary,
        ) = self._analyze_split(
            output_dir=output_dir,
            split_name="test",
            epoch=self.epochs,
            dataset=datasets[
                "test"
            ],
            update_train_trajectory=False,
        )

        self._save_train_trajectory_summary(
            output_dir,
            train_dataset,
        )

        self._write_group_epoch_summary(
            output_dir
        )

        self._save_group_trajectory_plots(
            output_dir
        )

        log("=" * 96)
        log("[EXAMINE] COMPLETE")
        log(
            f"[EXAMINE] best val-accuracy epoch: "
            f"{best_epoch}"
        )
        log(
            f"[EXAMINE] best val accuracy: "
            f"{100.0 * best_val_accuracy:.2f}%"
        )
        log(
            f"[EXAMINE] FINAL test accuracy: "
            f"{100.0 * final_test_summary['accuracy']:.2f}%"
        )
        log(
            f"[EXAMINE] FINAL test WGA: "
            f"{100.0 * final_test_summary['worst_group_accuracy']:.2f}%"
        )
        log(
            f"[EXAMINE] outputs: "
            f"{output_dir / 'gradient_diagnostics'}"
        )
        log("=" * 96)

    def _ensure_classifier_loaded(
        self,
        output_dir,
    ):
        if self.classifier is not None:
            return

        final_path = (
            Path(output_dir)
            / "final_classifier.pt"
        )

        if not final_path.exists():
            raise FileNotFoundError(
                f"Final classifier checkpoint missing: "
                f"{final_path}"
            )

        payload = torch.load(
            final_path,
            map_location="cpu",
            weights_only=False,
        )

        self.feature_dim = int(
            payload[
                "feature_dim"
            ]
        )

        self.n_classes = int(
            payload[
                "n_classes"
            ]
        )

        self.classifier = (
            LinearEmbeddingClassifier(
                input_dim=self.feature_dim,
                num_classes=self.n_classes,
            )
            .to(
                self.device
            )
        )

        self.classifier.load_state_dict(
            payload[
                "classifier_sd"
            ]
        )

        self.classifier.eval()

    def test(
        self,
        output_dir,
        split=("test",),
        result_path="",
    ):
        """
        main.py calls this after train().

        The final test diagnostics were already saved in train(), so this
        hook mainly writes a concise final classifier report.
        """

        self._ensure_classifier_loaded(
            output_dir
        )

        if isinstance(
            split,
            str,
        ):
            split = [
                split
            ]

        lines = []

        for split_name in split:

            if split_name not in {
                "train",
                "train_no_aug",
                "val",
                "test",
            }:
                continue

            canonical = (
                "train_no_aug"
                if split_name
                in {
                    "train",
                    "train_no_aug",
                }
                else split_name
            )

            dataset = (
                self._load_cached_dataset(
                    canonical
                )
            )

            metrics = (
                self._compute_per_sample_metrics(
                    dataset
                )
            )

            summary = (
                self._classification_summary(
                    metrics[
                        "pred"
                    ],
                    metrics[
                        "labels"
                    ],
                    metrics[
                        "groups"
                    ],
                    metrics[
                        "loss"
                    ],
                )
            )

            line = (
                f"EXAMINE final {canonical}: "
                f"loss={summary['loss']:.6f}, "
                f"accuracy="
                f"{100.0 * summary['accuracy']:.2f}%, "
                f"WGA="
                f"{100.0 * summary['worst_group_accuracy']:.2f}%"
            )

            log(
                line
            )

            lines.append(
                line
            )

        if result_path:
            with open(
                result_path,
                "a",
            ) as f:
                for line in lines:
                    f.write(
                        line + "\n"
                    )
