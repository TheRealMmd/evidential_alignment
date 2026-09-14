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

        The mapping from raw Rater score to training weight is configurable:

        SOFTMAX:
            w_i = softmax((r_eta(z_i) - mean_batch) / tau)

        SIGMOID:
            z_i_std = (r_eta(z_i) - mean_batch) / (std_batch + eps)
            w_i = sigmoid(z_i_std / tau)

        In BOTH modes, following the requested experimental definition:

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
      * after meta-training, freezes the learned rater and trains a fresh
        final classifier using the learned rater weights;
      * evaluates the final classifier on validation/test splits.
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
        # Raw-score -> training-weight mapping.
        #
        #   softmax:
        #       centered = score - batch_mean
        #       weight = softmax(centered / temperature)
        #
        #   sigmoid:
        #       standardized = (score - batch_mean) / (batch_std + eps)
        #       weight = sigmoid(standardized / temperature)
        #
        # In both cases the inner loss is:
        #       sum_i weight_i * CE_i
        #
        # IMPORTANT: for sigmoid the weights do NOT sum to one.
        # Therefore its inner-loss/gradient scale can be much larger
        # than in softmax mode. This is intentional in this experiment.
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
        self.final_selection = str(
            getattr(self.config, "rater_final_selection", "accuracy")
        ).lower()

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
            f"[Rater] Weighting transform = {self.weighting}; "
            f"temperature = {self.temperature}"
        )

        if self.weighting == "sigmoid":
            log(
                "[Rater] Sigmoid mode uses standardized batch scores and "
                "L_inner = sum_i w_i * CE_i WITHOUT dividing by sum(w)."
            )

        if self.config.check_point:
            self._load_rater_checkpoint(self.config.check_point)

    # ========================================================
    # Embedding extraction
    # ========================================================

    @torch.no_grad()
    def _extract_embedding_dataset(self, split, cache_dir):
        """Extract and cache frozen ImageNet backbone embeddings."""

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"{self.config.backbone}_imagenet_{split}.pt",
        )

        if os.path.exists(cache_file):
            log(f"[Rater] Loading cached embeddings: {cache_file}")
            payload = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
            )
            return EmbeddingTensorDataset(
                payload["embeddings"],
                payload["labels"],
                payload["groups"],
                payload["attrs"],
            )

        if split not in self.dataloaders:
            raise ValueError(
                f"Unknown split '{split}'. Available splits: "
                f"{list(self.dataloaders.keys())}"
            )

        loader = self.dataloaders[split]
        self.feature_model.eval()

        embeddings, labels, groups, attrs = [], [], [], []

        for x, y, g, a in tqdm(
            loader,
            desc=f"Extracting {split} embeddings",
        ):
            x = x.to(self.device, non_blocking=True)

            # The rater never sees the image directly.
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
        }
        torch.save(payload, cache_file)

        log(
            f"[Rater] Saved {len(labels)} {split} embeddings to {cache_file}"
        )

        return EmbeddingTensorDataset(
            embeddings,
            labels,
            groups,
            attrs,
        )

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
    # Score -> training-weight transformation
    # ========================================================

    def _scores_to_weights(self, raw_scores):
        """
        Convert unrestricted Rater outputs into the actual weights used
        by the inner/final classifier loss.

        SOFTMAX mode
        ------------
        centered_i = s_i - mean(s)
        w_i = softmax(centered_i / temperature)

        Properties:
            * 0 < w_i < 1
            * sum_i w_i = 1
            * average weight = 1 / batch_size

        SIGMOID mode
        ------------
        standardized_i =
            (s_i - mean(s)) / (std(s) + 1e-6)

        w_i = sigmoid(standardized_i / temperature)

        Properties:
            * 0 < w_i < 1
            * weights do NOT sum to one
            * standardized scores avoid saturation caused only by
              a drifting / expanding raw-score scale

        IMPORTANT
        ---------
        We intentionally DO NOT normalize sigmoid weights by their sum
        later. The requested inner objective is always:

            sum_i w_i * CE_i
        """

        if self.weighting == "softmax":
            centered_scores = (
                raw_scores - raw_scores.mean()
            )

            return torch.softmax(
                centered_scores / self.temperature,
                dim=0,
            )

        # self.weighting == "sigmoid"
        score_mean = raw_scores.mean()

        # unbiased=False is stable even for very small batches.
        score_std = raw_scores.std(
            unbiased=False
        )

        standardized_scores = (
            raw_scores - score_mean
        ) / (
            score_std + 1e-6
        )

        return torch.sigmoid(
            standardized_scores / self.temperature
        )

    # ========================================================
    # Differentiable inner optimization
    # ========================================================

    def _inner_unroll(self, inner_model, train_iterator, train_loader):
        """Perform differentiable weighted inner updates."""

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

            raw_scores = self.rater(z)

            # Actual training weights. The transform is selected by:
            #     --rater_weighting softmax
            # or:
            #     --rater_weighting sigmoid
            weights = self._scores_to_weights(
                raw_scores
            )

            logits = F.linear(z, fast_weight, fast_bias)
            per_sample_loss = F.cross_entropy(
                logits,
                y,
                reduction="none",
            )

            # Requested objective:
            #
            #     L_inner = sum_i w_i * CE_i
            #
            # There is deliberately NO division by weights.sum(),
            # including in sigmoid mode.
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
            last_raw_scores = raw_scores

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
        Compare Rater outputs with per-example classifier loss.

        For each embedding we compute:
          * raw rater score r_eta(z);
          * FINAL SCORE used by the inner loss, computed with the selected
            weighting transform (softmax or standardized sigmoid);
          * CE loss for every inner model;
          * mean CE loss across the inner-model population;
          * mean correctness across the population;
          * Waterbirds group / label / attribute.

        Important: the final score is computed batch-wise exactly like the
        weight used during inner training. The diagnostic DataLoader uses
        shuffle=False so this quantity is reproducible across evaluations.
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

            raw_scores = self.rater(z)
            final_scores = self._scores_to_weights(
                raw_scores
            )

            model_losses = []
            model_correct = []

            for model in models:
                logits = model(z)
                per_loss = F.cross_entropy(
                    logits,
                    y_device,
                    reduction="none",
                )
                correct = (
                    logits.argmax(dim=1) == y_device
                ).float()

                model_losses.append(per_loss)
                model_correct.append(correct)

            mean_loss = torch.stack(
                model_losses,
                dim=0,
            ).mean(dim=0)

            mean_correct = torch.stack(
                model_correct,
                dim=0,
            ).mean(dim=0)

            all_scores.append(raw_scores.cpu())
            all_final_scores.append(final_scores.cpu())
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

        if (
            len(scores) >= 2
            and np.std(scores) > 0
            and np.std(correctness) > 0
        ):
            spearman_correct = float(
                stats.spearmanr(scores, correctness).statistic
            )
        else:
            spearman_correct = 0.0

        if (
            len(final_scores) >= 2
            and np.std(final_scores) > 0
            and np.std(correctness) > 0
        ):
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
                        f"final_{self.weighting}_weight",
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
            histogram of the FINAL SCORE actually used in the weighted
            inner loss, using the currently selected weighting transform.

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
                f"Final score ({self.weighting} training weight)"
            )
            title_name = (
                f"Final {self.weighting} weight distribution"
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

        FINAL SCORE means the exact positive weight produced after the
        selected batch-wise weighting transform used by inner training.

        Thus this plot complements the existing raw-score-vs-loss plot.
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
            f"Final score ({self.weighting} training weight)"
        )
        ax.set_title(
            f"Final {self.weighting} weight vs inner-model loss | "
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
        """Raw/final Rater score vs loss for one downstream classifier."""

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
            final_scores = self._scores_to_weights(
                raw_scores
            )

            logits = model(z)
            losses = F.cross_entropy(
                logits,
                y_device,
                reduction="none",
            )
            correct = (
                logits.argmax(dim=1) == y_device
            ).float()

            all_scores.append(raw_scores.cpu())
            all_final_scores.append(final_scores.cpu())
            all_losses.append(losses.cpu())
            all_correct.append(correct.cpu())
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

        if (
            len(final_scores) >= 2
            and np.std(final_scores) > 0
            and np.std(correctness) > 0
        ):
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
    def _compute_scores(self, dataset):
        """Save raw Rater scores and the selected final training weights."""

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.rater.eval()
        scores, final_scores, labels, groups, attrs = [], [], [], [], []

        for z, y, g, a in loader:
            z = z.to(self.device, non_blocking=True)
            batch_scores = self.rater(z)
            batch_final_scores = self._scores_to_weights(
                batch_scores
            )

            scores.append(batch_scores.cpu())
            final_scores.append(batch_final_scores.cpu())
            labels.append(y.cpu())
            groups.append(g.cpu())
            attrs.append(a.cpu())

        return {
            "scores": torch.cat(scores),
            "final_scores": torch.cat(final_scores),
            "training_weights": torch.cat(final_scores),
            "labels": torch.cat(labels),
            "groups": torch.cat(groups),
            "attrs": torch.cat(attrs),
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
                f"final {self.weighting} weight mean="
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

    def _resolve_meta_splits(self, split):
        """
        Default:
            inner/meta-train = train_no_aug
            outer/meta-val   = val
        """

        if split == "train":
            return "train_no_aug", "val"

        if split == "train_no_aug":
            return "train_no_aug", "val"

        if split == "train_subset1":
            if "train_no_aug_subset1" in self.dataloaders:
                return "train_no_aug_subset1", "val"
            return split, "val"

        if split == "val_subset1":
            if "val_subset2" not in self.dataloaders:
                raise ValueError(
                    "val_subset1 requires --split_val < 1 so that "
                    "val_subset2 exists."
                )
            return "val_subset1", "val_subset2"

        return split, "val"

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
        train_dataset,
        val_dataset,
        output_dir,
    ):
        """
        Freeze the learned rater and train a fresh classifier with its
        per-example weights. This is the downstream classifier whose
        accuracy/WGA should normally be reported.
        """

        log("[Rater] Training final weighted classifier...")

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
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        best_value = -float("inf")
        best_state = None
        history = []

        for epoch in range(1, self.final_epochs + 1):
            model.train()

            running_weighted_loss = 0.0
            total_correct = 0
            total_examples = 0
            num_batches = 0

            for z, y, _, _ in train_loader:
                z = z.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.no_grad():
                    raw_scores = self.rater(z)
                    weights = self._scores_to_weights(
                        raw_scores
                    )

                logits = model(z)
                per_sample_loss = F.cross_entropy(
                    logits,
                    y,
                    reduction="none",
                )
                weighted_loss = (per_sample_loss * weights).sum()

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

                running_weighted_loss += weighted_loss.item()
                total_correct += (
                    logits.argmax(dim=1) == y
                ).sum().item()
                total_examples += y.numel()
                num_batches += 1

            train_acc = total_correct / max(total_examples, 1)
            train_weighted_loss = (
                running_weighted_loss / max(num_batches, 1)
            )

            val_metrics = self._evaluate_classifier(model, val_dataset)
            selection_value = self._selection_value(val_metrics)

            history.append(
                [
                    epoch,
                    train_weighted_loss,
                    train_acc,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["worst_group_accuracy"],
                ]
            )

            log(
                f"[Final classifier epoch {epoch:03d}] "
                f"weighted_train_loss={train_weighted_loss:.6f}, "
                f"train_acc={100.0 * train_acc:.2f}%, "
                f"val_loss={val_metrics['loss']:.6f}, "
                f"val_acc={100.0 * val_metrics['accuracy']:.2f}%, "
                f"val_WGA={100.0 * val_metrics['worst_group_accuracy']:.2f}%"
            )

            if selection_value > best_value:
                best_value = selection_value
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError("Final classifier training produced no model.")

        model.load_state_dict(best_state)
        model.to(self.device)
        model.eval()

        history_path = os.path.join(
            output_dir,
            "final_classifier_history.csv",
        )
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "weighted_train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "val_wga",
                ]
            )
            writer.writerows(history)

        for p in self.rater.parameters():
            p.requires_grad_(True)

        return model

    # ========================================================
    # Main training
    # ========================================================

    def train(self, output_dir, split="train"):
        os.makedirs(output_dir, exist_ok=True)

        cache_dir = os.path.join(output_dir, "embedding_cache")
        inner_split, outer_split = self._resolve_meta_splits(split)

        log(f"[Rater] Inner/meta-train split: {inner_split}")
        log(f"[Rater] Outer/held-out split: {outer_split}")

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

        log(
            f"[Rater] Meta-training complete. "
            f"Best outer loss = {best_outer_loss:.6f}"
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
        # Train a fresh final classifier with learned weights.
        # ----------------------------------------------------
        self.final_classifier = self._train_final_classifier(
            train_dataset,
            val_dataset,
            output_dir,
        )

        final_model_path = os.path.join(
            output_dir,
            "final_weighted_classifier.pt",
        )
        self._save_final_classifier(
            final_model_path,
            self.final_classifier,
        )

        final_val_metrics = self._evaluate_classifier(
            self.final_classifier,
            val_dataset,
        )

        log(
            f"[Rater FINAL VAL] "
            f"loss={final_val_metrics['loss']:.6f}, "
            f"acc={100.0 * final_val_metrics['accuracy']:.2f}%, "
            f"WGA={100.0 * final_val_metrics['worst_group_accuracy']:.2f}%"
        )

        for gid, acc in final_val_metrics["group_accuracy"].items():
            log(
                f"[Rater FINAL VAL] group {gid}: "
                f"{100.0 * acc:.2f}%"
            )

        final_val_relationship = self._classifier_score_loss_relationship(
            val_dataset,
            self.final_classifier,
        )
        self._save_all_score_diagnostics(
            relationship=final_val_relationship,
            output_dir=output_dir,
            meta_step=self.meta_steps,
            split_name=f"{outer_split}_final_classifier",
            tag_prefix="final",
        )

    # ========================================================
    # Test / final evaluation
    # ========================================================

    def test(self, output_dir, split=("test",), result_path=""):
        """
        For each requested split:
          1. compute and save rater scores;
          2. if a final weighted classifier is available, evaluate its
             loss, overall accuracy, per-group accuracy and WGA.
        """

        cache_dir = os.path.join(output_dir, "embedding_cache")

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
            payload = self._compute_scores(dataset)
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
