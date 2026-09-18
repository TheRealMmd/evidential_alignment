import csv
import os

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


class EmbeddingTensorDataset(Dataset):
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


class EmbeddingRaterSmall(nn.Module):
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


class InnerLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, z):
        return self.linear(z)


def _env_bool(name, default):
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _corr(x, y, mode="pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if mode == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


@register_algorithm("examine")
class Examine(Algorithm):
    """
    Direct gradient-utility Rater.

    For a current linear inner classifier theta:
        g_i     = grad_theta CE_i
        g_outer = grad_theta L_outer
        u_i     = <g_outer, g_i>

    The Rater is trained to predict u_i from the frozen ResNet embedding.

    Group/background labels are NEVER used in the training loss. They are
    used only for diagnostics and plots.

    Existing rater CLI arguments are reused. EXAMINE-specific switches are
    environment variables:

      EXAMINE_UTILITY_MODE=dot|cosine                 default dot
      EXAMINE_CLASS_BALANCED_OUTER=1|0               default 1
      EXAMINE_TARGET_NORM=within_class|batch|none     default within_class
      EXAMINE_RATER_INPUT=raw|class_centered          default raw
      EXAMINE_REGRESSION_LOSS=huber|mse               default huber
      EXAMINE_HUBER_DELTA=1.0
      EXAMINE_CLASS_MEAN_PENALTY=0.0
      EXAMINE_SANITY_CHECK=1|0                        default 1

    Recommended Waterbirds protocol:
      train_no_aug -> inner model training
      val_subset1  -> outer gradient defining utility
      val_subset2  -> Rater checkpoint selection/diagnostics
      test         -> final diagnostics only
    """

    def __init__(self, config):
        config.last_layer = True
        super().__init__(config)

        self.device = f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu"
        self.n_classes = self.datasets["train"].n_classes

        if not self.config.pretrained:
            raise ValueError("EXAMINE requires --pretrained True.")

        self.feature_model = Classifier(
            backbone=self.config.backbone,
            num_classes=self.n_classes,
            pretrained=True,
        ).to(self.device)
        self.feature_model.eval()
        for p in self.feature_model.parameters():
            p.requires_grad_(False)
        self.feature_dim = self.feature_model.backbone.num_features

        self.embedding_cache_root = str(
            getattr(self.config, "rater_embedding_cache_root", "")
            or os.environ.get("RATER_EMBEDDING_CACHE_ROOT", "")
        ).strip()

        self.meta_steps = int(getattr(self.config, "rater_meta_steps", self.config.epoch))
        self.inner_steps = int(getattr(self.config, "rater_inner_steps", 1))
        self.num_inner_models = int(getattr(self.config, "rater_num_inner_models", 2))
        self.inner_lr = float(getattr(self.config, "rater_inner_lr", 2e-4))
        self.rater_lr = float(getattr(self.config, "rater_outer_lr", 3e-4))
        self.grad_clip = float(getattr(self.config, "rater_grad_clip", 5.0))
        self.refresh_steps = int(getattr(self.config, "rater_refresh_steps", 1_000_000))
        self.rater_capacity = str(getattr(self.config, "rater_capacity", "medium")).lower()
        self.eval_freq = max(1, int(getattr(self.config, "eval_freq", 1)))

        self.utility_mode = os.environ.get("EXAMINE_UTILITY_MODE", "dot").strip().lower()
        self.class_balanced_outer = _env_bool("EXAMINE_CLASS_BALANCED_OUTER", True)
        self.target_norm = os.environ.get("EXAMINE_TARGET_NORM", "within_class").strip().lower()
        self.rater_input_mode = os.environ.get("EXAMINE_RATER_INPUT", "raw").strip().lower()
        self.regression_loss = os.environ.get("EXAMINE_REGRESSION_LOSS", "huber").strip().lower()
        self.huber_delta = float(os.environ.get("EXAMINE_HUBER_DELTA", "1.0"))
        self.class_mean_penalty = float(os.environ.get("EXAMINE_CLASS_MEAN_PENALTY", "0.0"))
        self.sanity_check = _env_bool("EXAMINE_SANITY_CHECK", True)

        if self.utility_mode not in {"dot", "cosine"}:
            raise ValueError("EXAMINE_UTILITY_MODE must be dot or cosine.")
        if self.target_norm not in {"none", "batch", "within_class"}:
            raise ValueError("EXAMINE_TARGET_NORM must be none, batch, or within_class.")
        if self.rater_input_mode not in {"raw", "class_centered"}:
            raise ValueError("EXAMINE_RATER_INPUT must be raw or class_centered.")
        if self.regression_loss not in {"huber", "mse"}:
            raise ValueError("EXAMINE_REGRESSION_LOSS must be huber or mse.")

        if self.rater_capacity == "small":
            self.rater = EmbeddingRaterSmall(self.feature_dim).to(self.device)
        elif self.rater_capacity == "medium":
            self.rater = EmbeddingRaterMedium(self.feature_dim).to(self.device)
        else:
            raise ValueError(f"Unknown rater capacity: {self.rater_capacity}")

        self.rater_optimizer = torch.optim.Adam(self.rater.parameters(), lr=self.rater_lr)
        self.inner_models = []
        self.inner_optimizers = []
        self.class_centroids = None
        self._did_sanity_check = False

        log("=" * 90)
        log("[EXAMINE] Direct gradient-utility Rater")
        log(f"[EXAMINE] utility={self.utility_mode}, target_norm={self.target_norm}, "
            f"class_balanced_outer={self.class_balanced_outer}")
        log(f"[EXAMINE] rater_input={self.rater_input_mode}, regression={self.regression_loss}")
        log("[EXAMINE] Group/background labels are diagnostics only.")
        log("=" * 90)

    # ---------------- persistent embedding cache ----------------

    @staticmethod
    def _safe_token(v):
        return str(v).replace("/", "_").replace(chr(92), "_").replace(" ", "_").replace(".", "p")

    def _cache_namespace(self):
        dataset = self._safe_token(getattr(self.config, "dataset", "dataset"))
        backbone = self._safe_token(self.config.backbone)
        resolution = self._safe_token(getattr(self.config, "resolution", 224))
        ptag = "imagenet_pretrained" if bool(self.config.pretrained) else "not_pretrained"
        return os.path.join(dataset, f"{backbone}_{ptag}", f"resolution_{resolution}")

    def _cache_dir(self, output_dir):
        if self.embedding_cache_root:
            path = os.path.join(self.embedding_cache_root, self._cache_namespace())
        else:
            path = os.path.join(output_dir, "embedding_cache")
        os.makedirs(path, exist_ok=True)
        return path

    def _cache_filename(self, split):
        name = self._safe_token(split)
        if "subset" in str(split):
            name += (
                f"_splittrain_{self._safe_token(getattr(self.config, 'split_train', 1.0))}"
                f"_splitval_{self._safe_token(getattr(self.config, 'split_val', 1.0))}"
                f"_seed_{self._safe_token(getattr(self.config, 'seed', 0))}"
            )
        return f"{name}.pt"

    @torch.no_grad()
    def _embedding_dataset(self, split, cache_dir):
        path = os.path.join(cache_dir, self._cache_filename(split))
        if os.path.exists(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if all(k in payload for k in ("embeddings", "labels", "groups", "attrs")):
                if payload["embeddings"].ndim == 2 and payload["embeddings"].shape[1] == self.feature_dim:
                    log(f"[EXAMINE embeddings] USING {split}: {path}")
                    return EmbeddingTensorDataset(
                        payload["embeddings"], payload["labels"], payload["groups"], payload["attrs"]
                    )

        if split not in self.dataloaders:
            raise ValueError(f"Split {split!r} unavailable; available={list(self.dataloaders.keys())}")

        loader = self.dataloaders[split]
        emb, ys, gs, attrs = [], [], [], []
        self.feature_model.eval()
        for x, y, g, a in tqdm(loader, desc=f"EXAMINE embedding extraction: {split}"):
            x = x.to(self.device, non_blocking=True)
            z = self.feature_model.backbone(x)
            emb.append(z.detach().cpu())
            ys.append(torch.as_tensor(y).cpu())
            gs.append(torch.as_tensor(g).cpu())
            attrs.append(torch.as_tensor(a).cpu())

        payload = {
            "embeddings": torch.cat(emb),
            "labels": torch.cat(ys),
            "groups": torch.cat(gs),
            "attrs": torch.cat(attrs),
        }
        torch.save(payload, path)
        log(f"[EXAMINE embeddings] SAVED {split}: {path}")
        return EmbeddingTensorDataset(
            payload["embeddings"], payload["labels"], payload["groups"], payload["attrs"]
        )

    # ---------------- protocol / model population ----------------

    def _resolve_splits(self):
        has_subsets = (
            float(getattr(self.config, "split_val", 1.0)) < 1.0
            and "val_subset1" in self.dataloaders
            and "val_subset2" in self.dataloaders
        )
        if has_subsets:
            return "train_no_aug", "val_subset1", "val_subset2"
        return "train_no_aug", "val", "val"

    def _new_inner(self):
        return InnerLinearClassifier(self.feature_dim, self.n_classes).to(self.device)

    def _init_population(self):
        self.inner_models = [self._new_inner() for _ in range(self.num_inner_models)]
        self.inner_optimizers = [torch.optim.SGD(m.parameters(), lr=self.inner_lr) for m in self.inner_models]

    def _refresh_inner(self, idx):
        self.inner_models[idx] = self._new_inner()
        self.inner_optimizers[idx] = torch.optim.SGD(self.inner_models[idx].parameters(), lr=self.inner_lr)

    @staticmethod
    def _next_batch(iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    def _compute_centroids(self, dataset):
        centroids = []
        for c in range(self.n_classes):
            mask = dataset.y_array == c
            centroids.append(dataset.embeddings[mask].mean(dim=0))
        self.class_centroids = torch.stack(centroids).to(self.device)

    def _rater_input(self, z, y):
        if self.rater_input_mode == "raw":
            return z
        return z - self.class_centroids[y]

    # ---------------- gradient utility ----------------

    def _outer_coeff(self, y):
        n = y.numel()
        if not self.class_balanced_outer:
            return torch.full((n,), 1.0 / max(n, 1), device=y.device, dtype=torch.float32)

        coeff = torch.zeros(n, device=y.device, dtype=torch.float32)
        classes = torch.unique(y)
        class_mass = 1.0 / max(len(classes), 1)
        for c in classes:
            mask = y == c
            coeff[mask] = class_mass / int(mask.sum().item())
        return coeff

    def _balanced_outer_loss(self, logits, y):
        per = F.cross_entropy(logits, y, reduction="none")
        return (per * self._outer_coeff(y).to(per.dtype)).sum()

    @torch.no_grad()
    def _outer_grad(self, model, z, y):
        logits = model(z)
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(y, num_classes=self.n_classes).to(probs.dtype)
        delta = probs - onehot
        coeff = self._outer_coeff(y).to(delta.dtype)
        d = delta * coeff.unsqueeze(1)
        gw = torch.einsum("nk,nd->kd", d, z)
        gb = d.sum(dim=0)
        return gw, gb

    @torch.no_grad()
    def _utility(self, model, z, y, outer_gw, outer_gb):
        logits = model(z)
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(y, num_classes=self.n_classes).to(probs.dtype)
        delta = probs - onehot

        projection = z @ outer_gw.t() + outer_gb.unsqueeze(0)
        dot = (delta * projection).sum(dim=1)
        if self.utility_mode == "dot":
            return dot

        sample_norm = torch.sqrt(
            delta.pow(2).sum(dim=1) * (z.pow(2).sum(dim=1) + 1.0) + 1e-12
        )
        outer_norm = torch.sqrt(outer_gw.pow(2).sum() + outer_gb.pow(2).sum() + 1e-12)
        return dot / (sample_norm * outer_norm + 1e-12)

    def _normalize_target(self, utility, y):
        if self.target_norm == "none":
            return utility
        if self.target_norm == "batch":
            return (utility - utility.mean()) / (utility.std(unbiased=False) + 1e-6)

        target = torch.zeros_like(utility)
        global_mean = utility.mean()
        global_std = utility.std(unbiased=False)
        for c in torch.unique(y):
            mask = y == c
            v = utility[mask]
            if v.numel() >= 2:
                mean, std = v.mean(), v.std(unbiased=False)
            else:
                mean, std = global_mean, global_std
            target[mask] = (v - mean) / (std + 1e-6)
        return target

    def _regression_loss(self, pred, target, y):
        if self.regression_loss == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.huber_loss(pred, target, delta=self.huber_delta)

        if self.class_mean_penalty > 0:
            means = [pred[y == c].mean() for c in torch.unique(y) if bool((y == c).any())]
            if len(means) >= 2:
                means = torch.stack(means)
                loss = loss + self.class_mean_penalty * (means - means.mean()).pow(2).mean()
        return loss

    def _sanity(self, model, z_train, y_train, z_outer, y_outer):
        if not self.sanity_check or self._did_sanity_check:
            return
        self._did_sanity_check = True

        with torch.no_grad():
            gw, gb = self._outer_grad(model, z_outer, y_outer)
            analytic = self._utility(model, z_train, y_train, gw, gb)[0]

        outer_loss = self._balanced_outer_loss(model(z_outer), y_outer)
        go_w, go_b = torch.autograd.grad(outer_loss, [model.linear.weight, model.linear.bias])
        sample_loss = F.cross_entropy(model(z_train[:1]), y_train[:1])
        gi_w, gi_b = torch.autograd.grad(sample_loss, [model.linear.weight, model.linear.bias])
        dot = (go_w * gi_w).sum() + (go_b * gi_b).sum()
        if self.utility_mode == "cosine":
            explicit = dot / (
                torch.sqrt(go_w.pow(2).sum() + go_b.pow(2).sum() + 1e-12)
                * torch.sqrt(gi_w.pow(2).sum() + gi_b.pow(2).sum() + 1e-12)
                + 1e-12
            )
        else:
            explicit = dot

        a = float(analytic.detach().cpu())
        e = float(explicit.detach().cpu())
        abs_err = abs(a - e)
        rel_err = abs_err / (abs(e) + 1e-12)
        log(f"[EXAMINE sanity] analytic={a:.9e}, autograd={e:.9e}, rel_error={rel_err:.3e}")
        if rel_err > 1e-3 and abs_err > 1e-6:
            raise RuntimeError("Gradient-utility sanity check failed.")

    def _update(self, model, inner_optimizer, train_batch, outer_batch):
        z, y, _, _ = train_batch
        zo, yo, _, _ = outer_batch
        z = z.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        zo = zo.to(self.device, non_blocking=True)
        yo = yo.to(self.device, non_blocking=True)

        self._sanity(model, z, y, zo, yo)

        with torch.no_grad():
            gw, gb = self._outer_grad(model, zo, yo)
            raw_utility = self._utility(model, z, y, gw, gb)
            target = self._normalize_target(raw_utility, y)

        pred = self.rater(self._rater_input(z, y))
        rater_loss = self._regression_loss(pred, target, y)
        self.rater_optimizer.zero_grad()
        rater_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.rater.parameters(), self.grad_clip)
        self.rater_optimizer.step()

        # Inner models evolve with ordinary CE only.
        inner_loss = F.cross_entropy(model(z), y)
        inner_optimizer.zero_grad()
        inner_loss.backward()
        inner_optimizer.step()

        return float(rater_loss.detach()), float(inner_loss.detach()), float(raw_utility.mean())

    # ---------------- held-out utility evaluation ----------------

    @torch.no_grad()
    def _evaluate_utility(self, model, selection_dataset, outer_dataset):
        zo = outer_dataset.embeddings.to(self.device)
        yo = outer_dataset.y_array.to(self.device)
        gw, gb = self._outer_grad(model, zo, yo)

        z = selection_dataset.embeddings.to(self.device)
        y = selection_dataset.y_array.to(self.device)
        raw = self._utility(model, z, y, gw, gb)
        target = self._normalize_target(raw, y)
        pred = self.rater(self._rater_input(z, y))

        if self.regression_loss == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.huber_loss(pred, target, delta=self.huber_delta)

        p = pred.detach().cpu().numpy()
        t = target.detach().cpu().numpy()
        r = raw.detach().cpu().numpy()
        return {
            "loss": float(loss),
            "pearson_target": _corr(p, t, "pearson"),
            "spearman_target": _corr(p, t, "spearman"),
            "pearson_raw": _corr(p, r, "pearson"),
            "spearman_raw": _corr(p, r, "spearman"),
        }

    # ---------------- score diagnostics ----------------

    @torch.no_grad()
    def _scores(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        scores, labels, groups, attrs = [], [], [], []
        self.rater.eval()
        for z, y, g, a in loader:
            z = z.to(self.device, non_blocking=True)
            yd = y.to(self.device, non_blocking=True)
            s = self.rater(self._rater_input(z, yd))
            scores.append(s.cpu())
            labels.append(y.cpu())
            groups.append(g.cpu())
            attrs.append(a.cpu())
        return {
            "scores": torch.cat(scores),
            "labels": torch.cat(labels),
            "groups": torch.cat(groups),
            "attrs": torch.cat(attrs),
        }

    @staticmethod
    def _summary(payload):
        s, y, g = payload["scores"], payload["labels"], payload["groups"]
        lines = [
            f"score mean={s.mean():.6f}, std={s.std(unbiased=False):.6f}, "
            f"min={s.min():.6f}, max={s.max():.6f}"
        ]
        cm, gm = {}, {}
        for c in torch.unique(y):
            mask = y == c
            cm[int(c)] = float(s[mask].mean())
            lines.append(f"class {int(c)}: n={int(mask.sum())}, mean_score={cm[int(c)]:.6f}")
        for gid in torch.unique(g):
            mask = g == gid
            gm[int(gid)] = float(s[mask].mean())
            lines.append(f"group {int(gid)}: n={int(mask.sum())}, mean_score={gm[int(gid)]:.6f}")
        if 0 in cm and 1 in cm:
            lines.append(f"CLASS GAP |y1-y0|={abs(cm[1]-cm[0]):.6f}")
        if all(k in gm for k in (0, 1, 2, 3)):
            lines.append(f"WITHIN-CLASS GAP y0 (g1-g0)={gm[1]-gm[0]:.6f}")
            lines.append(f"WITHIN-CLASS GAP y1 (g2-g3)={gm[2]-gm[3]:.6f}")
        return lines

    def _plot_scores(self, payload, output_dir, split, step):
        d = os.path.join(output_dir, "plots", "examine_raw_score_histogram")
        os.makedirs(d, exist_ok=True)
        s = payload["scores"].numpy()
        g = payload["groups"].numpy()
        names = {
            0: "Group 0: landbird / land",
            1: "Group 1: landbird / water",
            2: "Group 2: waterbird / land",
            3: "Group 3: waterbird / water",
        }
        fig, ax = plt.subplots(figsize=(11, 7))
        for gid in sorted(np.unique(g)):
            mask = g == gid
            ax.hist(s[mask], bins=35, density=True, alpha=0.45,
                    label=f"{names.get(int(gid), f'Group {gid}')} (n={int(mask.sum())})")
        ax.set_xlabel("Raw EXAMINE Rater score")
        ax.set_ylabel("Density")
        ax.set_title(f"Gradient-utility Rater scores by group | {split} | step {step}")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(d, f"step_{step:06d}_{split}_raw_score_hist.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        log(f"[EXAMINE plot] {path}")

    # ---------------- checkpoint ----------------

    def _save_checkpoint(self, path, step, selection_loss):
        torch.save({
            "rater_sd": self.rater.state_dict(),
            "rater_optimizer_sd": self.rater_optimizer.state_dict(),
            "step": step,
            "selection_utility_loss": selection_loss,
            "feature_dim": self.feature_dim,
            "backbone": self.config.backbone,
            "pretrained": True,
            "rater_capacity": self.rater_capacity,
            "utility_mode": self.utility_mode,
            "class_balanced_outer": self.class_balanced_outer,
            "target_norm": self.target_norm,
            "rater_input_mode": self.rater_input_mode,
            "regression_loss": self.regression_loss,
        }, path)

    def _load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.rater.load_state_dict(payload["rater_sd"] if "rater_sd" in payload else payload)
        self.rater.to(self.device)

    # ---------------- main train ----------------

    def train(self, output_dir, split="train"):
        os.makedirs(output_dir, exist_ok=True)
        cache_dir = self._cache_dir(output_dir)
        inner_split, outer_split, selection_split = self._resolve_splits()

        log(f"[EXAMINE] inner train={inner_split}")
        log(f"[EXAMINE] outer utility={outer_split}")
        log(f"[EXAMINE] Rater selection={selection_split}")

        train_ds = self._embedding_dataset(inner_split, cache_dir)
        outer_ds = self._embedding_dataset(outer_split, cache_dir)
        select_ds = self._embedding_dataset(selection_split, cache_dir)
        test_ds = self._embedding_dataset("test", cache_dir) if "test" in self.dataloaders else None

        self._compute_centroids(train_ds)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers, pin_memory=True)
        outer_loader = DataLoader(outer_ds, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers, pin_memory=True)
        train_it, outer_it = iter(train_loader), iter(outer_loader)
        self._init_population()

        best_loss = float("inf")
        best_step = None
        best_path = os.path.join(output_dir, "best_examine_rater.pt")
        latest_path = os.path.join(output_dir, "latest_examine_rater.pt")
        history_path = os.path.join(output_dir, "examine_history.csv")
        rows = []

        for step in tqdm(range(1, self.meta_steps + 1), desc="EXAMINE utility training"):
            self.rater.train()
            rater_losses, inner_losses, utilities = [], [], []

            for m_idx in range(self.num_inner_models):
                model = self.inner_models[m_idx]
                opt = self.inner_optimizers[m_idx]
                for _ in range(self.inner_steps):
                    tb, train_it = self._next_batch(train_it, train_loader)
                    ob, outer_it = self._next_batch(outer_it, outer_loader)
                    rl, il, um = self._update(model, opt, tb, ob)
                    rater_losses.append(rl)
                    inner_losses.append(il)
                    utilities.append(um)

            if 0 < self.refresh_steps < 1_000_000 and step % self.refresh_steps == 0:
                idx = (step // self.refresh_steps - 1) % self.num_inner_models
                self._refresh_inner(idx)

            train_rater_loss = float(np.mean(rater_losses))
            mean_inner_ce = float(np.mean(inner_losses))
            mean_utility = float(np.mean(utilities))

            selection_loss = float("nan")
            pearson = float("nan")
            spearman = float("nan")

            do_eval = step == 1 or step % self.eval_freq == 0 or step == self.meta_steps
            if do_eval:
                self.rater.eval()
                metrics = [self._evaluate_utility(m, select_ds, outer_ds) for m in self.inner_models]
                selection_loss = float(np.mean([x["loss"] for x in metrics]))
                pearson = float(np.nanmean([x["pearson_target"] for x in metrics]))
                spearman = float(np.nanmean([x["spearman_target"] for x in metrics]))

                log(
                    f"[EXAMINE step {step:04d}] train_rater_loss={train_rater_loss:.6f}, "
                    f"inner_CE={mean_inner_ce:.6f}, selection_utility_loss={selection_loss:.6f}, "
                    f"Pearson={pearson:.4f}, Spearman={spearman:.4f}"
                )

                select_scores = self._scores(select_ds)
                for line in self._summary(select_scores):
                    log(f"[EXAMINE/{selection_split}] {line}")
                self._plot_scores(select_scores, output_dir, selection_split, step)

                if test_ds is not None:
                    test_scores = self._scores(test_ds)
                    for line in self._summary(test_scores):
                        log(f"[EXAMINE/test] {line}")
                    self._plot_scores(test_scores, output_dir, "test", step)

                if selection_loss < best_loss:
                    best_loss = selection_loss
                    best_step = step
                    self._save_checkpoint(best_path, step, selection_loss)
                    log(f"[EXAMINE] NEW BEST step={step}, selection_utility_loss={selection_loss:.6f}")

            rows.append({
                "step": step,
                "train_rater_loss": train_rater_loss,
                "mean_inner_ce": mean_inner_ce,
                "mean_raw_utility": mean_utility,
                "selection_utility_loss": selection_loss,
                "pearson_prediction_target": pearson,
                "spearman_prediction_target": spearman,
            })
            with open(history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self._save_checkpoint(latest_path, step, selection_loss)

        if best_step is None:
            raise RuntimeError("No best EXAMINE checkpoint selected.")

        self._load_checkpoint(best_path)
        self.rater.eval()
        log(f"[EXAMINE] Restored BEST step={best_step}, selection_utility_loss={best_loss:.6f}")

        for name, ds in ((selection_split, select_ds), ("test", test_ds)):
            if ds is None:
                continue
            payload = self._scores(ds)
            torch.save(payload, os.path.join(output_dir, f"examine_scores_{name}.pt"))
            self._plot_scores(payload, output_dir, name, best_step)

    # ---------------- test hook ----------------

    def test(self, output_dir, split=("test",), result_path=""):
        cache_dir = self._cache_dir(output_dir)
        if isinstance(split, str):
            split = [split]

        best_path = os.path.join(output_dir, "best_examine_rater.pt")
        if os.path.exists(best_path):
            self._load_checkpoint(best_path)

        if self.rater_input_mode == "class_centered" and self.class_centroids is None:
            self._compute_centroids(self._embedding_dataset("train_no_aug", cache_dir))

        self.rater.eval()
        for sp in split:
            ds = self._embedding_dataset(sp, cache_dir)
            payload = self._scores(ds)
            path = os.path.join(output_dir, f"examine_scores_{sp}.pt")
            torch.save(payload, path)
            log(f"[EXAMINE] {sp} scores saved to {path}")
            summary = self._summary(payload)
            for line in summary:
                log(f"[EXAMINE/{sp}] {line}")
            if result_path:
                with open(result_path, "a") as f:
                    f.write(f"EXAMINE {sp}\n")
                    for line in summary:
                        f.write(line + "\n")
