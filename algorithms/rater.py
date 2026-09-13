import os
import csv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .algorithm import Algorithm
from .register import register_algorithm

from models.classifier import Classifier
from utils import log


# ============================================================
# Embedding datasets
# ============================================================

class EmbeddingTensorDataset(Dataset):
    """
    Dataset containing frozen backbone embeddings.

    Each sample:
        z : [embedding_dim]
        y : class label
        g : group label
        a : spurious/confounder attribute
    """

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
    """
    Small rater for ResNet-50 embeddings.

    Input:
        ResNet-50 pooled feature vector, normally 2048-D.

    Output:
        One raw scalar rating per embedding.
    """

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
    """
    Higher-capacity embedding rater.

    This is the feature-space analogue of your medium CIFAR
    image rater. Since z is already a semantic representation,
    convolutional layers are no longer appropriate.
    """

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
# Inner model
# ============================================================

class InnerLinearClassifier(nn.Module):
    """
    Inner task model used for bilevel optimization.

    IMPORTANT:
    The ResNet-50 is NOT trained here.

    ResNet-50 produces z.
    This classifier learns:
        z -> class logits
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, z):
        return self.linear(z)


# ============================================================
# RATER ALGORITHM
# ============================================================

@register_algorithm("rater")
class Rater(Algorithm):
    """
    Meta-learned rater for frozen ImageNet ResNet-50 embeddings.

    ------------------------------------------------------------
    High-level procedure
    ------------------------------------------------------------

    1. Freeze ImageNet-pretrained ResNet-50:

            x -> phi(x) = z

    2. Learn a rater:

            r_eta(z) -> scalar score

    3. Inner optimization:

            score_i = r_eta(z_i)

            w_i = softmax(score_i / temperature)

            L_inner =
                sum_i w_i * CE(h_theta(z_i), y_i)

            theta' =
                theta - lr * grad_theta L_inner

       Repeat for T differentiable inner steps.

    4. Outer optimization:

            L_outer =
                CE(h_theta'(z_val), y_val)

            eta <- eta - outer_lr *
                   grad_eta L_outer

       The gradient reaches eta through the inner optimization.

    The rater therefore learns which *representations* are useful
    for training the classifier.

    ------------------------------------------------------------
    Default meta parameters
    ------------------------------------------------------------

    meta_steps          = config.epoch
    inner_steps         = 2
    num_inner_models    = 4
    inner_lr            = 0.01
    outer_lr            = 3e-4
    temperature         = 2.0
    refresh_steps       = 100
    grad_clip           = 5.0

    Optional config attributes can override these:

        rater_meta_steps
        rater_inner_steps
        rater_num_inner_models
        rater_inner_lr
        rater_outer_lr
        rater_temperature
        rater_refresh_steps
        rater_grad_clip
        rater_capacity
        rater_outer_reg_weight
        rater_inner_reg_weight
        rater_score_reg_weight
    """

    def __init__(self, config):

        # We explicitly operate on frozen features.
        config.last_layer = True

        super().__init__(config)

        self.device = f"cuda:{self.config.gpu}"

        self.n_classes = self.datasets["train"].n_classes

        # ----------------------------------------------------
        # Require the representation requested for your method
        # ----------------------------------------------------

        if self.config.backbone != "resnet50":
            log(
                f"[Rater] Warning: requested backbone is "
                f"{self.config.backbone}, not resnet50."
            )

        if not self.config.pretrained:
            raise ValueError(
                "Rater is designed to use an ImageNet-pretrained "
                "backbone. Run with --pretrained True."
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
        # Hyperparameters
        # ----------------------------------------------------

        self.meta_steps = int(
            getattr(
                self.config,
                "rater_meta_steps",
                self.config.epoch,
            )
        )

        self.inner_steps = int(
            getattr(
                self.config,
                "rater_inner_steps",
                2,
            )
        )

        self.num_inner_models = int(
            getattr(
                self.config,
                "rater_num_inner_models",
                4,
            )
        )

        self.inner_lr = float(
            getattr(
                self.config,
                "rater_inner_lr",
                1e-2,
            )
        )

        self.outer_lr = float(
            getattr(
                self.config,
                "rater_outer_lr",
                3e-4,
            )
        )

        self.temperature = float(
            getattr(
                self.config,
                "rater_temperature",
                2.0,
            )
        )

        self.refresh_steps = int(
            getattr(
                self.config,
                "rater_refresh_steps",
                100,
            )
        )

        self.grad_clip = float(
            getattr(
                self.config,
                "rater_grad_clip",
                5.0,
            )
        )

        self.inner_reg_weight = float(
            getattr(
                self.config,
                "rater_inner_reg_weight",
                0.0,
            )
        )

        self.outer_reg_weight = float(
            getattr(
                self.config,
                "rater_outer_reg_weight",
                0.0,
            )
        )

        self.score_reg_weight = float(
            getattr(
                self.config,
                "rater_score_reg_weight",
                0.0,
            )
        )

        self.rater_capacity = getattr(
            self.config,
            "rater_capacity",
            "medium",
        )

        if self.inner_steps < 1:
            raise ValueError("rater_inner_steps must be >= 1.")

        # ----------------------------------------------------
        # Construct rater
        # ----------------------------------------------------

        if self.rater_capacity == "small":

            self.rater = EmbeddingRaterSmall(
                self.feature_dim
            ).to(self.device)

        elif self.rater_capacity == "medium":

            self.rater = EmbeddingRaterMedium(
                self.feature_dim
            ).to(self.device)

        else:
            raise ValueError(
                f"Unknown rater capacity: {self.rater_capacity}"
            )

        self.outer_optimizer = torch.optim.Adam(
            self.rater.parameters(),
            lr=self.outer_lr,
        )

        # Population of inner classifiers
        self.inner_models = []

        # Load trained rater if supplied
        if self.config.check_point:
            self._load_rater_checkpoint(
                self.config.check_point
            )

    # ========================================================
    # Backbone / embedding extraction
    # ========================================================

    @torch.no_grad()
    def _extract_embedding_dataset(
        self,
        split,
        cache_dir,
    ):
        """
        Convert images to frozen ImageNet ResNet-50 embeddings.

        No gradient passes through ResNet-50.

        Embeddings are cached because they do not change during
        rater meta-training.
        """

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir,
            f"{self.config.backbone}_imagenet_{split}.pt",
        )

        if os.path.exists(cache_file):

            log(
                f"[Rater] Loading cached embeddings: "
                f"{cache_file}"
            )

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
                f"Unknown split '{split}'. "
                f"Available splits: "
                f"{list(self.dataloaders.keys())}"
            )

        loader = self.dataloaders[split]

        self.feature_model.eval()

        embeddings = []
        labels = []
        groups = []
        attrs = []

        for x, y, g, a in tqdm(
            loader,
            desc=f"Extracting {split} embeddings",
        ):

            x = x.to(
                self.device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # This is the crucial change from your CIFAR rater.
            #
            # We do NOT feed x to the rater.
            #
            # x -> frozen ResNet50 -> z
            # ------------------------------------------------

            z = self.feature_model.backbone(x)

            embeddings.append(
                z.detach().cpu()
            )

            labels.append(
                torch.as_tensor(y).cpu()
            )

            groups.append(
                torch.as_tensor(g).cpu()
            )

            attrs.append(
                torch.as_tensor(a).cpu()
            )

        embeddings = torch.cat(
            embeddings,
            dim=0,
        )

        labels = torch.cat(
            labels,
            dim=0,
        )

        groups = torch.cat(
            groups,
            dim=0,
        )

        attrs = torch.cat(
            attrs,
            dim=0,
        )

        payload = {
            "embeddings": embeddings,
            "labels": labels,
            "groups": groups,
            "attrs": attrs,
        }

        torch.save(
            payload,
            cache_file,
        )

        log(
            f"[Rater] Saved {len(labels)} "
            f"{split} embeddings to {cache_file}"
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
        """
        Fresh classifier head operating on frozen embeddings.
        """

        return InnerLinearClassifier(
            input_dim=self.feature_dim,
            num_classes=self.n_classes,
        ).to(self.device)

    def _initialize_inner_population(self):

        self.inner_models = [
            self._new_inner_model()
            for _ in range(self.num_inner_models)
        ]

    # ========================================================
    # Iterator helper
    # ========================================================

    @staticmethod
    def _next_batch(
        iterator,
        loader,
    ):

        try:
            batch = next(iterator)

        except StopIteration:

            iterator = iter(loader)
            batch = next(iterator)

        return batch, iterator

    # ========================================================
    # Differentiable inner optimization
    # ========================================================

    def _inner_unroll(
        self,
        inner_model,
        train_iterator,
        train_loader,
    ):
        """
        Perform T differentiable gradient steps.

        The inner model is a linear classifier on z.

        Rater:
            r_eta(z_i) -> score_i

        Batch weights:
            w_i = softmax(score_i / tau)

        Inner loss:
            sum_i w_i CE(h_theta(z_i), y_i)

        Gradients are created with create_graph=True, which
        allows the outer validation loss to differentiate
        through these updates into the rater.
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

            z = z.to(
                self.device,
                non_blocking=True,
            )

            y = y.to(
                self.device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # RATER NOW SEES EMBEDDINGS
            # ------------------------------------------------

            raw_scores = self.rater(z)

            # Centering does not change softmax probabilities,
            # but improves numerical behavior.
            centered_scores = (
                raw_scores - raw_scores.mean()
            )

            weights = torch.softmax(
                centered_scores / self.temperature,
                dim=0,
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

            # Softmax weights sum to 1.
            inner_loss = (
                per_sample_loss * weights
            ).sum()

            # Optional regularization of inner classifier.
            if self.inner_reg_weight > 0:

                inner_reg = (
                    fast_weight.pow(2).sum()
                    +
                    fast_bias.pow(2).sum()
                )

                inner_loss = (
                    inner_loss
                    +
                    self.inner_reg_weight
                    * inner_reg
                )

            grad_w, grad_b = torch.autograd.grad(
                inner_loss,
                [fast_weight, fast_bias],
                create_graph=True,
            )

            fast_weight = (
                fast_weight
                -
                self.inner_lr * grad_w
            )

            fast_bias = (
                fast_bias
                -
                self.inner_lr * grad_b
            )

            last_raw_scores = raw_scores

        fast_params = {
            "weight": fast_weight,
            "bias": fast_bias,
        }

        return (
            fast_params,
            train_iterator,
            last_raw_scores,
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
        One complete outer-loop update of the embedding rater.
        """

        self.rater.train()

        outer_batch, val_iterator = self._next_batch(
            val_iterator,
            val_loader,
        )

        z_outer, y_outer, _, _ = outer_batch

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
            ) = self._inner_unroll(
                inner_model,
                train_iterator,
                train_loader,
            )

            # ------------------------------------------------
            # Unweighted outer loss.
            #
            # Rater is optimized ONLY through how its ratings
            # changed inner-model learning.
            # ------------------------------------------------

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

        # Average meta-objective over inner model population.
        meta_loss = torch.stack(
            outer_losses
        ).mean()

        # ----------------------------------------------------
        # Optional L2 regularization on the rater.
        # ----------------------------------------------------

        if self.outer_reg_weight > 0:

            outer_reg = sum(
                p.pow(2).sum()
                for p in self.rater.parameters()
            )

            meta_loss = (
                meta_loss
                +
                self.outer_reg_weight
                * outer_reg
            )

        # ----------------------------------------------------
        # Optional diversity regularizer.
        #
        # Negative variance encourages the rater not to collapse
        # to the same score for every embedding.
        #
        # Default = 0, so this does nothing unless enabled.
        # ----------------------------------------------------

        if self.score_reg_weight > 0:

            outer_scores = self.rater(
                z_outer
            )

            score_reg = -torch.var(
                outer_scores
            )

            meta_loss = (
                meta_loss
                +
                self.score_reg_weight
                * score_reg
            )

        self.outer_optimizer.zero_grad()

        meta_loss.backward()

        if self.grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(
                self.rater.parameters(),
                self.grad_clip,
            )

        self.outer_optimizer.step()

        # ----------------------------------------------------
        # Advance the persistent inner models.
        #
        # The differentiable fast parameters are detached before
        # copying them back.
        # ----------------------------------------------------

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
    # Rating utilities
    # ========================================================

    @torch.no_grad()
    def _compute_scores(
        self,
        dataset,
    ):
        """
        Compute one raw scalar score for every embedding.
        """

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.rater.eval()

        scores = []
        labels = []
        groups = []
        attrs = []

        for z, y, g, a in loader:

            z = z.to(
                self.device,
                non_blocking=True,
            )

            batch_scores = self.rater(z)

            scores.append(
                batch_scores.cpu()
            )

            labels.append(
                y.cpu()
            )

            groups.append(
                g.cpu()
            )

            attrs.append(
                a.cpu()
            )

        return {
            "scores": torch.cat(scores),
            "labels": torch.cat(labels),
            "groups": torch.cat(groups),
            "attrs": torch.cat(attrs),
        }

    @staticmethod
    def _rating_summary(payload):

        scores = payload["scores"]
        labels = payload["labels"]
        groups = payload["groups"]

        lines = []

        lines.append(
            f"score mean={scores.mean():.6f}, "
            f"std={scores.std():.6f}, "
            f"min={scores.min():.6f}, "
            f"max={scores.max():.6f}"
        )

        for c in torch.unique(labels):

            mask = labels == c

            lines.append(
                f"class {int(c)}: "
                f"n={int(mask.sum())}, "
                f"mean_score={scores[mask].mean():.6f}"
            )

        for g in torch.unique(groups):

            mask = groups == g

            lines.append(
                f"group {int(g)}: "
                f"n={int(mask.sum())}, "
                f"mean_score={scores[mask].mean():.6f}"
            )

        return lines

    # ========================================================
    # Checkpoint utilities
    # ========================================================

    def _save_rater_checkpoint(
        self,
        path,
        meta_step,
        outer_loss,
    ):

        torch.save(
            {
                "rater_sd": self.rater.state_dict(),
                "outer_optimizer_sd":
                    self.outer_optimizer.state_dict(),
                "meta_step": meta_step,
                "outer_loss": outer_loss,
                "feature_dim": self.feature_dim,
                "backbone": self.config.backbone,
                "pretrained": True,
                "rater_capacity": self.rater_capacity,
            },
            path,
        )

    def _load_rater_checkpoint(
        self,
        path,
    ):

        if not os.path.exists(path):
            raise ValueError(
                f"Rater checkpoint does not exist: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if "rater_sd" not in checkpoint:
            raise ValueError(
                f"{path} is not a Rater checkpoint."
            )

        self.rater.load_state_dict(
            checkpoint["rater_sd"]
        )

        log(
            f"[Rater] Loaded checkpoint "
            f"from {path}"
        )

    # ========================================================
    # Split handling
    # ========================================================

    def _resolve_meta_splits(
        self,
        split,
    ):
        """
        Select disjoint inner and outer data.

        Default:
            inner = train_no_aug
            outer = val

        If you intentionally run:
            --train_split val_subset1
            --split_val < 1

        then:
            inner = val_subset1
            outer = val_subset2
        """

        if split == "train":

            inner_split = "train_no_aug"
            outer_split = "val"

        elif split == "train_no_aug":

            inner_split = "train_no_aug"
            outer_split = "val"

        elif split == "train_subset1":

            if "train_no_aug_subset1" in self.dataloaders:
                inner_split = "train_no_aug_subset1"
            else:
                inner_split = split

            outer_split = "val"

        elif split == "val_subset1":

            inner_split = "val_subset1"

            if "val_subset2" not in self.dataloaders:
                raise ValueError(
                    "val_subset1 requires --split_val < 1 "
                    "so that val_subset2 exists."
                )

            outer_split = "val_subset2"

        else:

            inner_split = split
            outer_split = "val"

        return (
            inner_split,
            outer_split,
        )

    # ========================================================
    # Main training
    # ========================================================

    def train(
        self,
        output_dir,
        split="train",
    ):

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        cache_dir = os.path.join(
            output_dir,
            "embedding_cache",
        )

        (
            inner_split,
            outer_split,
        ) = self._resolve_meta_splits(
            split
        )

        log(
            f"[Rater] Inner/meta-train split: "
            f"{inner_split}"
        )

        log(
            f"[Rater] Outer/held-out split: "
            f"{outer_split}"
        )

        # ----------------------------------------------------
        # Extract ImageNet-pretrained ResNet-50 representations.
        # ----------------------------------------------------

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

        train_iterator = iter(
            train_loader
        )

        val_iterator = iter(
            val_loader
        )

        self._initialize_inner_population()

        # ----------------------------------------------------
        # Staggered population refresh
        # ----------------------------------------------------

        refresh_period = max(
            1,
            self.refresh_steps,
        )

        refresh_offsets = [
            (i * refresh_period)
            // self.num_inner_models
            for i in range(
                self.num_inner_models
            )
        ]

        best_outer_loss = float("inf")

        best_path = os.path.join(
            output_dir,
            "best_rater.pt",
        )

        latest_path = os.path.join(
            output_dir,
            "latest_rater.pt",
        )

        history_path = os.path.join(
            output_dir,
            "rater_history.csv",
        )

        history = []

        log(
            "[Rater] Starting feature-rater "
            "bilevel optimization"
        )

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

            # ----------------------------------------------
            # Staggered reset of inner models.
            # ----------------------------------------------

            if meta_step > 1:

                position = (
                    (meta_step - 1)
                    % refresh_period
                )

                for i, offset in enumerate(
                    refresh_offsets
                ):

                    if position == offset:

                        self.inner_models[i] = (
                            self._new_inner_model()
                        )

            (
                outer_loss,
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
                    meta_step,
                    outer_loss,
                ]
            )

            # ----------------------------------------------
            # Save best rater
            # ----------------------------------------------

            if outer_loss < best_outer_loss:

                best_outer_loss = outer_loss

                self._save_rater_checkpoint(
                    best_path,
                    meta_step,
                    outer_loss,
                )

            # ----------------------------------------------
            # Logging
            # ----------------------------------------------

            if (
                meta_step == 1
                or
                meta_step
                % self.config.eval_freq
                == 0
            ):

                with torch.no_grad():

                    sample_z = (
                        val_dataset.embeddings[
                            : min(
                                1024,
                                len(val_dataset),
                            )
                        ]
                        .to(self.device)
                    )

                    sample_scores = self.rater(
                        sample_z
                    )

                log(
                    f"[Rater step {meta_step}] "
                    f"outer_loss={outer_loss:.6f}, "
                    f"score_mean="
                    f"{sample_scores.mean().item():.6f}, "
                    f"score_std="
                    f"{sample_scores.std().item():.6f}"
                )

        # ----------------------------------------------------
        # Save final rater
        # ----------------------------------------------------

        self._save_rater_checkpoint(
            latest_path,
            self.meta_steps,
            history[-1][1],
        )

        # Save history
        with open(
            history_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "meta_step",
                    "outer_loss",
                ]
            )

            writer.writerows(
                history
            )

        # ----------------------------------------------------
        # Use best outer-loss rater for downstream scoring.
        # ----------------------------------------------------

        checkpoint = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )

        self.rater.load_state_dict(
            checkpoint["rater_sd"]
        )

        log(
            f"[Rater] Training complete. "
            f"Best outer loss = "
            f"{best_outer_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save scores for the inner/meta-training embeddings.
        # ----------------------------------------------------

        train_scores = self._compute_scores(
            train_dataset
        )

        torch.save(
            train_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{inner_split}.pt",
            ),
        )

        # Save held-out scores as well.
        val_scores = self._compute_scores(
            val_dataset
        )

        torch.save(
            val_scores,
            os.path.join(
                output_dir,
                f"rater_scores_{outer_split}.pt",
            ),
        )

    # ========================================================
    # Test / rating
    # ========================================================

    def test(
        self,
        output_dir,
        split=["test"],
        result_path="",
    ):
        """
        Rate embeddings in the requested split(s).

        Since the rater itself is not a classifier, "test" here
        means:
            - extract frozen ResNet-50 embeddings
            - assign each embedding a raw rating
            - save ratings + labels + groups + attrs
            - report score statistics
        """

        cache_dir = os.path.join(
            output_dir,
            "embedding_cache",
        )

        for sp in split:

            dataset = self._extract_embedding_dataset(
                sp,
                cache_dir,
            )

            payload = self._compute_scores(
                dataset
            )

            score_path = os.path.join(
                output_dir,
                f"rater_scores_{sp}.pt",
            )

            torch.save(
                payload,
                score_path,
            )

            summary = self._rating_summary(
                payload
            )

            log(
                f"[Rater] {sp} ratings "
                f"saved to {score_path}"
            )

            for line in summary:
                log(
                    f"[Rater/{sp}] {line}"
                )

            if result_path:

                with open(
                    result_path,
                    "a",
                ) as fout:

                    fout.write(
                        f"Rater {sp}\n"
                    )

                    for line in summary:
                        fout.write(
                            line + "\n"
                        )
