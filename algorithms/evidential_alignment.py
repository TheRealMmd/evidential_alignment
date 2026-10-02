import torch
from .algorithm import Algorithm
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from .register import register_algorithm
from data.data_utils import IdxDataset
from .utils import init_optimizer
import os
from torch.utils.data import DataLoader
from models.classifier import Classifier
import torch.nn.functional as F
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class AlignmentModel(torch.nn.Module):
    def __init__(self, w0, b0):
        super().__init__()
        self.w0 = w0.detach().clone()
        self.b0 = b0.detach().clone()
        self.linear = torch.nn.Linear(w0.shape[1], w0.shape[0], bias=True)
        
    def forward(self, x):
        y_old = x @ self.w0.t() + self.b0
        y_new = self.linear(x)
        return y_old + y_new

def wxe_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    l = weights * ce
    return l.sum()

@register_algorithm("evidential_alignment")
class EvidentialAlignment(Algorithm):
    def __init__(self, config):
        # Evidential Alignment operates on frozen backbone embeddings /
        # last-layer models during EDL + Alignment.  The current repository
        # Algorithm._get_split() expects config.last_layer to exist, but the
        # public argument parser does not define it.  Set it explicitly here
        # before Algorithm initialization/evaluation.
        #
        # This is a compatibility fix only; it does not change the EA loss,
        # weights, optimizer, data split, or training rule.
        config.last_layer = True

        super(EvidentialAlignment, self).__init__(config)
        self._init_model()
        self._init_training()
        
    def _init_model(self):
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        self.model = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        
        if self.config.check_point:
            log(f"loading the model checkpoint from {self.config.check_point}")
            if len(self.config.erm_model) > 0:
                log(f"ignoring the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.check_point)
            self.model.load_state_dict(saved_dict["model_sd"])
        
        if len(self.config.check_point) == 0 and len(self.config.erm_model) > 0:
            log(f"loading the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.erm_model)
            # Filter out position_ids from state dict
            filtered_sd = {k: v for k, v in saved_dict["model_sd"].items() 
                          if not k.endswith("position_ids")}
            self.model.load_state_dict(filtered_sd, strict=False)

        self.model.to(self.device)
        
    def _init_training(self):
        self.optimizer_cls = None
        self.scheduler_cls = None
        
        if self.config.split_val < 1.0:
            self.sel_metrics = [("embed_val_subset2_acc", True), ("embed_val_subset2_worst_cls_acc", True), 
                              ("embed_val_subset2_worst_group_acc", True), ("embed_val_subset2_avg_cls_diff", False)]
        else:
            self.sel_metrics = [("embed_val_acc", True), ("embed_val_worst_cls_acc", True), 
                              ("embed_val_worst_group_acc", True), ("embed_val_avg_cls_diff", False)]
        self.best_meters = {m:BestMetric(max_val) for m, max_val in self.sel_metrics}

        # Diagnostics only; these do not alter the EA optimization.
        self.ea_diag_every = max(
            1,
            int(os.environ.get("EA_DIAG_EVERY", "1"))
        )
        self.ea_diag_include_epoch0 = (
            os.environ.get("EA_DIAG_INCLUDE_EPOCH0", "1").strip().lower()
            in {"1", "true", "yes", "y", "on"}
        )
        self.ea_diag_save_per_sample = (
            os.environ.get("EA_DIAG_SAVE_PER_SAMPLE", "1").strip().lower()
            in {"1", "true", "yes", "y", "on"}
        )
        self.ea_diag_group_rows = []

        log(
            "[EA compatibility] config.last_layer=True; "
            "validation/test evaluation will use embedding splits."
        )


    # ============================================================
    # Diagnostics helpers
    # ============================================================

    @staticmethod
    def _ea_group_name(gid):
        names = {
            0: "landbird_land",
            1: "landbird_water",
            2: "waterbird_land",
            3: "waterbird_water",
        }
        return names.get(int(gid), f"group_{int(gid)}")

    def _collect_evidence_diagnostics(self, loader):
        """
        Evaluate the CURRENT evidence head on every calibration example.

        This function is diagnostic only. It makes no optimizer step.

        The quantity named repo_uncertainty is exactly the quantity used
        by the supplied repository code:

            evidence = sigmoid(logits)
            alpha = evidence + 1
            S = sum(alpha)
            repo_uncertainty = K / (S + 1)

        We additionally save K/S as a reference diagnostic, but K/S is
        never substituted into the original weighting rule.
        """

        was_training = self.model.fc.training
        self.model.fc.eval()

        n = len(loader.dataset)

        result = {
            "idx": torch.empty(n, dtype=torch.long),
            "label": torch.empty(n, dtype=torch.long),
            "group": torch.empty(n, dtype=torch.long),
            "attr": torch.empty(n, dtype=torch.long),
            "prediction": torch.empty(n, dtype=torch.long),

            "logits": torch.empty(
                n,
                self.n_classes,
                dtype=torch.float32,
            ),
            "evidence": torch.empty(
                n,
                self.n_classes,
                dtype=torch.float32,
            ),
            "alpha": torch.empty(
                n,
                self.n_classes,
                dtype=torch.float32,
            ),
            "probs": torch.empty(
                n,
                self.n_classes,
                dtype=torch.float32,
            ),

            "p_true": torch.empty(n, dtype=torch.float32),
            "evidence_true": torch.empty(n, dtype=torch.float32),
            "evidence_other": torch.empty(n, dtype=torch.float32),
            "evidence_sum": torch.empty(n, dtype=torch.float32),
            "dirichlet_strength": torch.empty(n, dtype=torch.float32),
            "repo_uncertainty": torch.empty(n, dtype=torch.float32),
            "dirichlet_uncertainty_k_over_s": torch.empty(
                n,
                dtype=torch.float32,
            ),
            "predictive_entropy": torch.empty(n, dtype=torch.float32),
            "max_prob": torch.empty(n, dtype=torch.float32),
            "prob_margin": torch.empty(n, dtype=torch.float32),
        }

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="EA evidence diagnostics",
                leave=False,
            ):
                idx, x, y, g, a = batch

                idx_cpu = torch.as_tensor(
                    idx
                ).view(-1).long().cpu()

                x = x.to(self.device)
                y_device = y.to(
                    self.device
                ).view(-1).long()

                output = self.model.fc(x)

                if output.dim() == 1:
                    output = output.unsqueeze(0)

                evidence = torch.sigmoid(output)
                alpha = evidence + 1.0

                S = torch.sum(
                    alpha,
                    dim=-1,
                    keepdim=True,
                )

                probs = alpha / S

                row = torch.arange(
                    y_device.shape[0],
                    device=self.device,
                )

                p_true = probs[
                    row,
                    y_device,
                ]

                evidence_true = evidence[
                    row,
                    y_device,
                ]

                evidence_sum = evidence.sum(
                    dim=-1
                )

                evidence_other = (
                    evidence_sum
                    - evidence_true
                )

                prediction = torch.argmax(
                    output,
                    dim=-1,
                )

                # EXACT repository quantity.
                repo_uncertainty = (
                    self.n_classes
                    / (
                        S.squeeze(-1)
                        + 1.0
                    )
                )

                # Reference only, never used for training here.
                uncertainty_k_over_s = (
                    self.n_classes
                    / S.squeeze(-1)
                )

                predictive_entropy = -(
                    probs
                    * torch.log(
                        probs.clamp_min(1e-12)
                    )
                ).sum(dim=-1)

                sorted_probs, _ = probs.sort(
                    dim=-1,
                    descending=True,
                )

                max_prob = sorted_probs[:, 0]

                if self.n_classes >= 2:
                    prob_margin = (
                        sorted_probs[:, 0]
                        - sorted_probs[:, 1]
                    )
                else:
                    prob_margin = (
                        sorted_probs[:, 0]
                    )

                result["idx"][
                    idx_cpu
                ] = idx_cpu

                result["label"][
                    idx_cpu
                ] = y.view(-1).long().cpu()

                result["group"][
                    idx_cpu
                ] = g.view(-1).long().cpu()

                result["attr"][
                    idx_cpu
                ] = a.view(-1).long().cpu()

                result["prediction"][
                    idx_cpu
                ] = prediction.view(-1).cpu()

                result["logits"][
                    idx_cpu
                ] = output.detach().float().cpu()

                result["evidence"][
                    idx_cpu
                ] = evidence.detach().float().cpu()

                result["alpha"][
                    idx_cpu
                ] = alpha.detach().float().cpu()

                result["probs"][
                    idx_cpu
                ] = probs.detach().float().cpu()

                result["p_true"][
                    idx_cpu
                ] = p_true.detach().float().cpu()

                result["evidence_true"][
                    idx_cpu
                ] = evidence_true.detach().float().cpu()

                result["evidence_other"][
                    idx_cpu
                ] = evidence_other.detach().float().cpu()

                result["evidence_sum"][
                    idx_cpu
                ] = evidence_sum.detach().float().cpu()

                result["dirichlet_strength"][
                    idx_cpu
                ] = S.squeeze(-1).detach().float().cpu()

                result["repo_uncertainty"][
                    idx_cpu
                ] = repo_uncertainty.detach().float().cpu()

                result[
                    "dirichlet_uncertainty_k_over_s"
                ][
                    idx_cpu
                ] = uncertainty_k_over_s.detach().float().cpu()

                result["predictive_entropy"][
                    idx_cpu
                ] = predictive_entropy.detach().float().cpu()

                result["max_prob"][
                    idx_cpu
                ] = max_prob.detach().float().cpu()

                result["prob_margin"][
                    idx_cpu
                ] = prob_margin.detach().float().cpu()

        result["misclassified"] = (
            result["prediction"]
            != result["label"]
        )

        if was_training:
            self.model.fc.train()

        return result

    def _ea_hist_by_group(
        self,
        values,
        groups,
        metric,
        title,
        path,
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        groups = np.asarray(
            groups,
            dtype=np.int64,
        )

        finite = np.isfinite(values)

        if not finite.any():
            return

        clean = values[finite]

        lo = float(
            np.quantile(clean, 0.005)
        )

        hi = float(
            np.quantile(clean, 0.995)
        )

        if abs(hi - lo) < 1e-12:
            lo -= 0.5
            hi += 0.5

        bins = np.linspace(
            lo,
            hi,
            40,
        )

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        for gid in sorted(
            np.unique(groups)
        ):
            mask = (
                (groups == gid)
                & finite
            )

            vals = np.clip(
                values[mask],
                lo,
                hi,
            )

            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.42,
                label=(
                    f"g{int(gid)}: "
                    f"{self._ea_group_name(gid)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(
            True,
            linestyle=":",
            alpha=0.3,
        )

        fig.tight_layout()

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _ea_box_by_group(
        self,
        values,
        groups,
        metric,
        title,
        path,
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        groups = np.asarray(
            groups,
            dtype=np.int64,
        )

        gids = sorted(
            np.unique(groups)
        )

        data = [
            values[
                groups == gid
            ]
            for gid in gids
        ]

        labels = [
            (
                f"g{int(gid)}\n"
                f"{self._ea_group_name(gid)}"
            )
            for gid in gids
        ]

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
        )

        ax.set_title(title)
        ax.set_ylabel(metric)

        ax.grid(
            True,
            axis="y",
            linestyle=":",
            alpha=0.3,
        )

        fig.tight_layout()

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _ea_scatter_by_group(
        self,
        x_values,
        y_values,
        groups,
        x_name,
        y_name,
        title,
        path,
    ):
        x_values = np.asarray(
            x_values,
            dtype=np.float64,
        )
        y_values = np.asarray(
            y_values,
            dtype=np.float64,
        )
        groups = np.asarray(
            groups,
            dtype=np.int64,
        )

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        for gid in sorted(
            np.unique(groups)
        ):
            mask = (
                groups == gid
            )

            ax.scatter(
                x_values[mask],
                y_values[mask],
                s=18,
                alpha=0.55,
                label=(
                    f"g{int(gid)}: "
                    f"{self._ea_group_name(gid)} "
                    f"(n={int(mask.sum())})"
                ),
            )

        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend(fontsize=9)
        ax.grid(
            True,
            linestyle=":",
            alpha=0.3,
        )

        fig.tight_layout()

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)


    def _write_ea_group_summary(
        self,
        output_dir,
    ):
        if not self.ea_diag_group_rows:
            return

        path = os.path.join(
            output_dir,
            "evidential_diagnostics",
            "edl_group_summary.csv",
        )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(
            path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    self.ea_diag_group_rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                self.ea_diag_group_rows
            )

    def _save_evidence_diagnostics(
        self,
        output_dir,
        epoch,
        diagnostics,
    ):
        root = os.path.join(
            output_dir,
            "evidential_diagnostics",
            "edl",
            f"epoch_{int(epoch):03d}",
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        scalar_metrics = [
            "repo_uncertainty",
            "dirichlet_uncertainty_k_over_s",
            "p_true",
            "evidence_true",
            "evidence_other",
            "evidence_sum",
            "dirichlet_strength",
            "predictive_entropy",
            "max_prob",
            "prob_margin",
        ]

        if self.ea_diag_save_per_sample:
            path = os.path.join(
                root,
                "per_sample_evidence.csv",
            )

            with open(
                path,
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)

                header = [
                    "idx",
                    "label",
                    "group",
                    "attr",
                    "prediction",
                    "misclassified",
                ]

                for c in range(
                    self.n_classes
                ):
                    header.append(
                        f"logit_{c}"
                    )

                for c in range(
                    self.n_classes
                ):
                    header.append(
                        f"evidence_{c}"
                    )

                for c in range(
                    self.n_classes
                ):
                    header.append(
                        f"alpha_{c}"
                    )

                for c in range(
                    self.n_classes
                ):
                    header.append(
                        f"prob_{c}"
                    )

                header += scalar_metrics

                writer.writerow(header)

                for i in range(
                    len(
                        diagnostics[
                            "idx"
                        ]
                    )
                ):
                    row = [
                        int(
                            diagnostics[
                                "idx"
                            ][i]
                        ),
                        int(
                            diagnostics[
                                "label"
                            ][i]
                        ),
                        int(
                            diagnostics[
                                "group"
                            ][i]
                        ),
                        int(
                            diagnostics[
                                "attr"
                            ][i]
                        ),
                        int(
                            diagnostics[
                                "prediction"
                            ][i]
                        ),
                        int(
                            diagnostics[
                                "misclassified"
                            ][i]
                        ),
                    ]

                    row += [
                        float(x)
                        for x
                        in diagnostics[
                            "logits"
                        ][i].tolist()
                    ]

                    row += [
                        float(x)
                        for x
                        in diagnostics[
                            "evidence"
                        ][i].tolist()
                    ]

                    row += [
                        float(x)
                        for x
                        in diagnostics[
                            "alpha"
                        ][i].tolist()
                    ]

                    row += [
                        float(x)
                        for x
                        in diagnostics[
                            "probs"
                        ][i].tolist()
                    ]

                    row += [
                        float(
                            diagnostics[
                                name
                            ][i]
                        )
                        for name
                        in scalar_metrics
                    ]

                    writer.writerow(row)

        torch.save(
            diagnostics,
            os.path.join(
                root,
                "per_sample_evidence.pt",
            ),
        )

        groups_np = (
            diagnostics[
                "group"
            ].numpy()
        )

        plot_metrics = [
            "repo_uncertainty",
            "dirichlet_uncertainty_k_over_s",
            "p_true",
            "evidence_sum",
            "evidence_true",
            "predictive_entropy",
            "prob_margin",
        ]

        plot_dir = os.path.join(
            root,
            "plots",
        )

        for metric in plot_metrics:
            self._ea_hist_by_group(
                diagnostics[
                    metric
                ].numpy(),
                groups_np,
                metric,
                (
                    f"{metric} by Waterbirds group\n"
                    f"EDL epoch {int(epoch)}"
                ),
                os.path.join(
                    plot_dir,
                    f"{metric}_hist_by_group.png",
                ),
            )

        for metric in scalar_metrics:
            values = (
                diagnostics[
                    metric
                ].numpy()
            )

            for gid in sorted(
                np.unique(groups_np)
            ):
                mask = (
                    groups_np == gid
                )

                vals = values[mask]

                self.ea_diag_group_rows.append(
                    {
                        "epoch": int(epoch),
                        "group": int(gid),
                        "group_name": (
                            self._ea_group_name(
                                gid
                            )
                        ),
                        "metric": metric,
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
                            np.median(vals)
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

        self._write_ea_group_summary(
            output_dir
        )

        log(
            f"[EA DIAG EDL {int(epoch):03d}] "
            f"mean repo_uncertainty="
            f"{diagnostics['repo_uncertainty'].mean().item():.6f}, "
            f"mean p_true="
            f"{diagnostics['p_true'].mean().item():.6f}"
        )

        for gid in sorted(
            np.unique(groups_np)
        ):
            mask = (
                groups_np == gid
            )

            log(
                f"  g{int(gid)} "
                f"({self._ea_group_name(gid)}): "
                f"uncertainty="
                f"{diagnostics['repo_uncertainty'][mask].mean().item():.6f}, "
                f"p_true="
                f"{diagnostics['p_true'][mask].mean().item():.6f}, "
                f"evidence_sum="
                f"{diagnostics['evidence_sum'][mask].mean().item():.6f}"
            )

    def _save_ea_trajectory_plots(
        self,
        output_dir,
    ):
        if not self.ea_diag_group_rows:
            return

        metrics = [
            "repo_uncertainty",
            "dirichlet_uncertainty_k_over_s",
            "p_true",
            "evidence_sum",
            "evidence_true",
            "predictive_entropy",
            "prob_margin",
        ]

        out_dir = os.path.join(
            output_dir,
            "evidential_diagnostics",
            "trajectory_plots",
        )

        os.makedirs(
            out_dir,
            exist_ok=True,
        )

        for metric in metrics:
            rows = [
                row
                for row
                in self.ea_diag_group_rows
                if row[
                    "metric"
                ] == metric
            ]

            if not rows:
                continue

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            for gid in sorted(
                {
                    row["group"]
                    for row in rows
                }
            ):
                group_rows = sorted(
                    [
                        row
                        for row in rows
                        if row[
                            "group"
                        ] == gid
                    ],
                    key=lambda row: (
                        row["epoch"]
                    ),
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
                    linewidth=1.4,
                    label=(
                        f"g{gid}: "
                        f"{self._ea_group_name(gid)}"
                    ),
                )

            ax.set_title(
                f"Mean {metric} during EDL training"
            )

            ax.set_xlabel(
                "EDL epoch"
            )

            ax.set_ylabel(
                f"group mean {metric}"
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
                os.path.join(
                    out_dir,
                    (
                        f"{metric}_"
                        "group_mean_over_edl_epochs.png"
                    ),
                ),
                dpi=160,
                bbox_inches="tight",
            )

            plt.close(fig)

    def _save_final_weight_diagnostics(
        self,
        output_dir,
        evidence_diagnostics,
        weight_diagnostics,
    ):
        root = os.path.join(
            output_dir,
            "evidential_diagnostics",
            "final_alignment_weights",
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        payload = {
            **evidence_diagnostics,
            **weight_diagnostics,
        }

        torch.save(
            payload,
            os.path.join(
                root,
                "per_sample_final_weights.pt",
            ),
        )

        columns = [
            "idx",
            "label",
            "group",
            "attr",
            "prediction",
            "misclassified",
            "p_true",
            "repo_uncertainty",
            "dirichlet_uncertainty_k_over_s",
            "evidence_true",
            "evidence_other",
            "evidence_sum",
            "dirichlet_strength",
            "predictive_entropy",
            "high_confidence_misclass",
            "low_uncertainty_mask",
            "upweight_mask",
            "weight_before_class_balance",
            "class_balance_multiplier",
            "weight_before_normalization",
            "final_alignment_weight",
        ]

        with open(
            os.path.join(
                root,
                "per_sample_final_weights.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            n = len(
                evidence_diagnostics[
                    "idx"
                ]
            )

            for i in range(n):
                writer.writerow(
                    [
                        int(
                            evidence_diagnostics[
                                "idx"
                            ][i]
                        ),
                        int(
                            evidence_diagnostics[
                                "label"
                            ][i]
                        ),
                        int(
                            evidence_diagnostics[
                                "group"
                            ][i]
                        ),
                        int(
                            evidence_diagnostics[
                                "attr"
                            ][i]
                        ),
                        int(
                            evidence_diagnostics[
                                "prediction"
                            ][i]
                        ),
                        int(
                            evidence_diagnostics[
                                "misclassified"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "p_true"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "repo_uncertainty"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "dirichlet_uncertainty_k_over_s"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "evidence_true"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "evidence_other"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "evidence_sum"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "dirichlet_strength"
                            ][i]
                        ),
                        float(
                            evidence_diagnostics[
                                "predictive_entropy"
                            ][i]
                        ),
                        int(
                            weight_diagnostics[
                                "high_confidence_misclass"
                            ][i]
                        ),
                        int(
                            weight_diagnostics[
                                "low_uncertainty_mask"
                            ][i]
                        ),
                        int(
                            weight_diagnostics[
                                "upweight_mask"
                            ][i]
                        ),
                        float(
                            weight_diagnostics[
                                "weight_before_class_balance"
                            ][i]
                        ),
                        float(
                            weight_diagnostics[
                                "class_balance_multiplier"
                            ][i]
                        ),
                        float(
                            weight_diagnostics[
                                "weight_before_normalization"
                            ][i]
                        ),
                        float(
                            weight_diagnostics[
                                "final_alignment_weight"
                            ][i]
                        ),
                    ]
                )

        summary = {
            "n": int(
                len(
                    evidence_diagnostics[
                        "idx"
                    ]
                )
            ),
            "uncertainty_threshold_quantile_0": float(
                weight_diagnostics[
                    "uncertainty_threshold"
                ]
            ),
            "misclassified_count": int(
                evidence_diagnostics[
                    "misclassified"
                ].sum().item()
            ),
            "high_confidence_misclass_count": int(
                weight_diagnostics[
                    "high_confidence_misclass"
                ].sum().item()
            ),
            "low_uncertainty_count": int(
                weight_diagnostics[
                    "low_uncertainty_mask"
                ].sum().item()
            ),
            "upweight_mask_count": int(
                weight_diagnostics[
                    "upweight_mask"
                ].sum().item()
            ),
            "final_weight_min": float(
                weight_diagnostics[
                    "final_alignment_weight"
                ].min().item()
            ),
            "final_weight_max": float(
                weight_diagnostics[
                    "final_alignment_weight"
                ].max().item()
            ),
            "final_weight_mean": float(
                weight_diagnostics[
                    "final_alignment_weight"
                ].mean().item()
            ),
            "balance_classes": bool(
                self.config.balance_classes
            ),
            "evidence_weighting_active": bool(
                weight_diagnostics[
                    "upweight_mask"
                ].any().item()
            ),
            "weighting_rule_note": (
                "Original repository rule preserved: "
                "uncertainty_threshold = quantile(uncertainties, 0); "
                "low_uncertainty_mask = uncertainties < threshold; "
                "then optional class balancing and max normalization."
            ),
        }

        with open(
            os.path.join(
                root,
                "weighting_summary.json",
            ),
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        groups_np = (
            evidence_diagnostics[
                "group"
            ].numpy()
        )

        plots = {
            "final_alignment_weight": (
                weight_diagnostics[
                    "final_alignment_weight"
                ].numpy()
            ),
            "weight_before_class_balance": (
                weight_diagnostics[
                    "weight_before_class_balance"
                ].numpy()
            ),
            "class_balance_multiplier": (
                weight_diagnostics[
                    "class_balance_multiplier"
                ].numpy()
            ),
            "repo_uncertainty": (
                evidence_diagnostics[
                    "repo_uncertainty"
                ].numpy()
            ),
            "p_true": (
                evidence_diagnostics[
                    "p_true"
                ].numpy()
            ),
            "evidence_sum": (
                evidence_diagnostics[
                    "evidence_sum"
                ].numpy()
            ),
        }

        plot_dir = os.path.join(
            root,
            "plots",
        )

        for metric, values in plots.items():
            self._ea_hist_by_group(
                values,
                groups_np,
                metric,
                (
                    f"{metric} by Waterbirds group\n"
                    "after EDL / before Alignment"
                ),
                os.path.join(
                    plot_dir,
                    f"{metric}_hist_by_group.png",
                ),
            )

        final_weights_np = (
            weight_diagnostics[
                "final_alignment_weight"
            ].numpy()
        )

        self._ea_box_by_group(
            final_weights_np,
            groups_np,
            "final_alignment_weight",
            (
                "ACTUAL fixed Alignment weights "
                "by Waterbirds group"
            ),
            os.path.join(
                plot_dir,
                "final_alignment_weight_boxplot_by_group.png",
            ),
        )


        # ------------------------------------------------------------
        # Group-level final-weight summary
        # ------------------------------------------------------------

        final_weights_np = (
            weight_diagnostics[
                "final_alignment_weight"
            ].numpy()
        )

        uncertainty_np = (
            evidence_diagnostics[
                "repo_uncertainty"
            ].numpy()
        )

        p_true_np = (
            evidence_diagnostics[
                "p_true"
            ].numpy()
        )

        upweight_np = (
            weight_diagnostics[
                "upweight_mask"
            ].numpy()
        )

        labels_np = (
            evidence_diagnostics[
                "label"
            ].numpy()
        )

        final_group_summary_path = os.path.join(
            root,
            "final_weight_group_summary.csv",
        )

        with open(
            final_group_summary_path,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "group",
                    "group_name",
                    "n",
                    "mean_final_weight",
                    "std_final_weight",
                    "min_final_weight",
                    "max_final_weight",
                    "mean_repo_uncertainty",
                    "mean_p_true",
                    "upweight_mask_count",
                ]
            )

            for gid in sorted(
                np.unique(groups_np)
            ):
                mask = (
                    groups_np == gid
                )

                writer.writerow(
                    [
                        int(gid),
                        self._ea_group_name(gid),
                        int(mask.sum()),
                        float(
                            final_weights_np[
                                mask
                            ].mean()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].std()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].min()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].max()
                        ),
                        float(
                            uncertainty_np[
                                mask
                            ].mean()
                        ),
                        float(
                            p_true_np[
                                mask
                            ].mean()
                        ),
                        int(
                            upweight_np[
                                mask
                            ].sum()
                        ),
                    ]
                )

        # ------------------------------------------------------------
        # Class-level final-weight summary
        # ------------------------------------------------------------

        final_class_summary_path = os.path.join(
            root,
            "final_weight_class_summary.csv",
        )

        with open(
            final_class_summary_path,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "label",
                    "n",
                    "mean_final_weight",
                    "std_final_weight",
                    "min_final_weight",
                    "max_final_weight",
                    "mean_repo_uncertainty",
                    "mean_p_true",
                    "upweight_mask_count",
                ]
            )

            for label in sorted(
                np.unique(labels_np)
            ):
                mask = (
                    labels_np == label
                )

                writer.writerow(
                    [
                        int(label),
                        int(mask.sum()),
                        float(
                            final_weights_np[
                                mask
                            ].mean()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].std()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].min()
                        ),
                        float(
                            final_weights_np[
                                mask
                            ].max()
                        ),
                        float(
                            uncertainty_np[
                                mask
                            ].mean()
                        ),
                        float(
                            p_true_np[
                                mask
                            ].mean()
                        ),
                        int(
                            upweight_np[
                                mask
                            ].sum()
                        ),
                    ]
                )

        # ------------------------------------------------------------
        # Sorted per-sample views.
        # These contain the same actual weights as the main CSV, but
        # make the lowest/highest-weight examples easy to inspect.
        # ------------------------------------------------------------

        sample_rows = []

        n = len(
            evidence_diagnostics[
                "idx"
            ]
        )

        for i in range(n):
            sample_rows.append(
                {
                    "idx": int(
                        evidence_diagnostics[
                            "idx"
                        ][i]
                    ),
                    "label": int(
                        evidence_diagnostics[
                            "label"
                        ][i]
                    ),
                    "group": int(
                        evidence_diagnostics[
                            "group"
                        ][i]
                    ),
                    "group_name": self._ea_group_name(
                        evidence_diagnostics[
                            "group"
                        ][i]
                    ),
                    "attr": int(
                        evidence_diagnostics[
                            "attr"
                        ][i]
                    ),
                    "prediction": int(
                        evidence_diagnostics[
                            "prediction"
                        ][i]
                    ),
                    "misclassified": int(
                        evidence_diagnostics[
                            "misclassified"
                        ][i]
                    ),
                    "p_true": float(
                        evidence_diagnostics[
                            "p_true"
                        ][i]
                    ),
                    "repo_uncertainty": float(
                        evidence_diagnostics[
                            "repo_uncertainty"
                        ][i]
                    ),
                    "upweight_mask": int(
                        weight_diagnostics[
                            "upweight_mask"
                        ][i]
                    ),
                    "class_balance_multiplier": float(
                        weight_diagnostics[
                            "class_balance_multiplier"
                        ][i]
                    ),
                    "final_alignment_weight": float(
                        weight_diagnostics[
                            "final_alignment_weight"
                        ][i]
                    ),
                }
            )

        sorted_fields = [
            "idx",
            "label",
            "group",
            "group_name",
            "attr",
            "prediction",
            "misclassified",
            "p_true",
            "repo_uncertainty",
            "upweight_mask",
            "class_balance_multiplier",
            "final_alignment_weight",
        ]

        for filename, reverse in [
            (
                "samples_sorted_low_to_high_weight.csv",
                False,
            ),
            (
                "samples_sorted_high_to_low_weight.csv",
                True,
            ),
        ]:
            with open(
                os.path.join(
                    root,
                    filename,
                ),
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=sorted_fields,
                )

                writer.writeheader()

                writer.writerows(
                    sorted(
                        sample_rows,
                        key=lambda row: (
                            row[
                                "final_alignment_weight"
                            ]
                        ),
                        reverse=reverse,
                    )
                )

        # ------------------------------------------------------------
        # Relationships between evidence quantities and the ACTUAL
        # fixed weights used in Alignment training.
        # ------------------------------------------------------------

        self._ea_scatter_by_group(
            uncertainty_np,
            final_weights_np,
            groups_np,
            "repo_uncertainty",
            "final_alignment_weight",
            (
                "Repository uncertainty vs ACTUAL Alignment weight"
            ),
            os.path.join(
                plot_dir,
                (
                    "repo_uncertainty_vs_"
                    "final_alignment_weight_by_group.png"
                ),
            ),
        )

        self._ea_scatter_by_group(
            p_true_np,
            final_weights_np,
            groups_np,
            "p_true",
            "final_alignment_weight",
            (
                "True-class probability vs ACTUAL Alignment weight"
            ),
            os.path.join(
                plot_dir,
                (
                    "p_true_vs_"
                    "final_alignment_weight_by_group.png"
                ),
            ),
        )

        log("=" * 100)
        log("[EA FINAL WEIGHT DIAGNOSTICS]")

        log(
            f"quantile-0 uncertainty threshold="
            f"{summary['uncertainty_threshold_quantile_0']:.8f}"
        )

        log(
            f"misclassified="
            f"{summary['misclassified_count']}, "
            f"high_confidence_misclass="
            f"{summary['high_confidence_misclass_count']}, "
            f"low_uncertainty="
            f"{summary['low_uncertainty_count']}, "
            f"upweight_mask="
            f"{summary['upweight_mask_count']}"
        )

        log(
            f"final weight min="
            f"{summary['final_weight_min']:.6f}, "
            f"max="
            f"{summary['final_weight_max']:.6f}, "
            f"mean="
            f"{summary['final_weight_mean']:.6f}"
        )


        if not summary[
            "evidence_weighting_active"
        ]:
            log(
                "[EA DIAGNOSTIC NOTE] "
                "upweight_mask is empty under the exact released "
                "quantile-0 + strict-'<' rule. In this run, the "
                "final weight variation therefore comes from the "
                "subsequent class-balancing / max-normalization "
                "steps rather than per-sample uncertainty."
            )

        for gid in sorted(
            np.unique(groups_np)
        ):
            mask = (
                groups_np == gid
            )

            log(
                f"  g{int(gid)} "
                f"({self._ea_group_name(gid)}): "
                f"weight="
                f"{final_weights_np[mask].mean():.6f}, "
                f"repo_uncertainty="
                f"{evidence_diagnostics['repo_uncertainty'][mask].mean().item():.6f}, "
                f"p_true="
                f"{evidence_diagnostics['p_true'][mask].mean().item():.6f}"
            )

        log(
            f"Saved diagnostics to: {root}"
        )

        log("=" * 100)

    # ============================================================
    # Original repository weighting rule + audit outputs
    # ============================================================

    def compute_uncertainty_weights(
        self,
        loader,
        balance_classes,
        return_diagnostics=False,
    ):
        diagnostics = (
            self._collect_evidence_diagnostics(
                loader
            )
        )

        uncertainties = (
            diagnostics[
                "repo_uncertainty"
            ].clone()
        )

        predictions = (
            diagnostics[
                "prediction"
            ].clone()
        )

        class_labels = (
            diagnostics[
                "label"
            ].clone()
        )

        p_true = (
            diagnostics[
                "p_true"
            ].clone()
        )

        # Enhanced misclassification detection.
        misclassified = (
            predictions
            != class_labels
        )

        # ORIGINAL repository logic preserved.
        uncertainty_threshold = (
            torch.quantile(
                uncertainties,
                0,
            )
        )

        low_uncertainty_mask = (
            uncertainties
            < uncertainty_threshold
        )

        # Focused upweighting: Only high-confidence errors.
        high_confidence_misclass = (
            misclassified
            & (p_true < 0.3)
        )

        upweight_mask = (
            high_confidence_misclass
            & low_uncertainty_mask
        )

        weights = torch.ones_like(
            uncertainties
        )

        weights[
            upweight_mask
        ] = uncertainties[
            upweight_mask
        ]

        weight_before_class_balance = (
            weights.clone()
        )

        class_balance_multiplier = (
            torch.ones_like(
                weights
            )
        )

        if balance_classes:
            class_counts = torch.bincount(
                class_labels
            ).float()

            class_weights = (
                class_counts.max()
                / class_counts
            )

            for y in range(
                len(class_counts)
            ):
                mask = (
                    class_labels == y
                )

                weights[
                    mask
                ] *= class_weights[y]

                class_balance_multiplier[
                    mask
                ] = class_weights[y]

        weight_before_normalization = (
            weights.clone()
        )

        # ORIGINAL normalization.
        weights = (
            weights
            / weights.max()
        )

        weight_diagnostics = {
            "uncertainty_threshold": float(
                uncertainty_threshold.item()
            ),
            "high_confidence_misclass": (
                high_confidence_misclass.cpu()
            ),
            "low_uncertainty_mask": (
                low_uncertainty_mask.cpu()
            ),
            "upweight_mask": (
                upweight_mask.cpu()
            ),
            "weight_before_class_balance": (
                weight_before_class_balance.cpu()
            ),
            "class_balance_multiplier": (
                class_balance_multiplier.cpu()
            ),
            "weight_before_normalization": (
                weight_before_normalization.cpu()
            ),
            "final_alignment_weight": (
                weights.cpu()
            ),
        }

        if return_diagnostics:
            return (
                weights,
                diagnostics,
                weight_diagnostics,
            )

        return weights
    def train(self, output_dir, split="train"):
        timer = Timer()
        criterion = wxe_fn
        initial_weight = self.model.fc.weight.detach().clone()
        initial_bias = self.model.fc.bias.detach().clone()
        embed_loaders = self.get_embed_loaders(split)
        train_loader = embed_loaders["embed_"+split]
        idxdataset = IdxDataset(train_loader.dataset)
        idx_loader = DataLoader(
            idxdataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
        )

        # Epoch-0 diagnostic snapshot of the ERM head before EDL.
        if self.ea_diag_include_epoch0:
            epoch0_diagnostics = (
                self._collect_evidence_diagnostics(
                    idx_loader
                )
            )

            self._save_evidence_diagnostics(
                output_dir,
                0,
                epoch0_diagnostics,
            )
        
        # EDL training phase
        log("Starting EDL training phase...")
        edl_optimizer = init_optimizer(
            self.model.fc,
            self.config.optimizer_cls,
            {"lr":0.01, "weight_decay":0.0001, "momentum":0.9}
        )
        
        for epoch in range(1, self.config.edl_epochs + 1):
            self.model.train()
            epoch_meters = {k: AverageMeter() for k in ["edl_loss", "edl_acc"]}
            
            for batch in tqdm(idx_loader, desc=f"EDL Training {split}", leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)
                
                edl_optimizer.zero_grad()
                output = self.model.fc(x)
                evidence = F.sigmoid(output)
                alpha = evidence + 1.0
            
                S = torch.sum(alpha, dim=-1, keepdim=True)
                probs = alpha / S
                y_oh = F.one_hot(y, self.n_classes)
                edl_loss = (torch.pow(y_oh - probs, 2.0) + alpha * (S - alpha) / (torch.pow(S,2.0)*(S+1))).sum()
                
                # Fix dimension in KL divergence calculation
                alpha_0 = torch.ones_like(alpha)
                alpha = y_oh + (1 - y_oh) * alpha
                kl_div = torch.sum(
                    torch.lgamma(torch.sum(alpha, dim=-1)) - torch.lgamma(torch.sum(alpha_0, dim=-1)) -
                    torch.sum(torch.lgamma(alpha), dim=-1) + torch.sum(torch.lgamma(alpha_0), dim=-1) +
                    torch.sum((alpha - alpha_0) * (torch.digamma(alpha) - 
                    torch.digamma(torch.sum(alpha, dim=-1, keepdim=True))), dim=-1)
                )
                
                total_loss = edl_loss + min(epoch/self.config.annealing_step, 1) * kl_div.sum()
                # total_loss = edl_loss + self.config.kl_reg * kl_div.sum()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fc.parameters(), 100.)
                edl_optimizer.step()
                
                # Fix dimension in prediction calculation
                pred = torch.argmax(output, dim=-1)
                acc = (pred == y).float().mean()
                
                epoch_meters["edl_loss"].update(total_loss.item(), x.size(0))
                epoch_meters["edl_acc"].update(acc.item(), x.size(0))
            
            train_state = {k: epoch_meters[k].avg for k in epoch_meters}
            msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
            log(f"[EDL Epoch {epoch}] {msg}")

            # Diagnostics only: no optimizer step occurs here.
            if (
                epoch == 1
                or epoch % self.ea_diag_every == 0
                or epoch == self.config.edl_epochs
            ):
                edl_diagnostics = (
                    self._collect_evidence_diagnostics(
                        idx_loader
                    )
                )

                self._save_evidence_diagnostics(
                    output_dir,
                    epoch,
                    edl_diagnostics,
                )


       
        (
            weights,
            final_evidence_diagnostics,
            final_weight_diagnostics,
        ) = self.compute_uncertainty_weights(
            idx_loader,
            self.config.balance_classes,
            return_diagnostics=True,
        )

        self._save_final_weight_diagnostics(
            output_dir,
            final_evidence_diagnostics,
            final_weight_diagnostics,
        )

        self._save_ea_trajectory_plots(
            output_dir
        )
        
        alignment_model = AlignmentModel(initial_weight, initial_bias)
        alignment_model.to(self.device)
        self.model.fc = alignment_model
        
        
        self.optimizer_cls = init_optimizer(
            self.model.fc.linear,
            self.config.optimizer_cls,
            self.config.optimizer_cls_kwargs 
        )
        
        # Alignment training phase
        for epoch in range(1, self.config.alignment_epochs+1):
            self.model.fc.train()
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc"]}

            for batch in tqdm(idx_loader, desc=f"Alignment {split}", leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer_cls.zero_grad()
                logits = self.model.fc(x)
                
                # Weighted cross entropy loss
                loss = criterion(logits, y, weights[idx].to(self.device))
                
                # Add regularization
                reg = self.model.fc.linear.weight.pow(2).sum() + self.model.fc.linear.bias.pow(2).sum()
                loss += self.config.reg_weight * reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fc.linear.parameters(), 10.)
                self.optimizer_cls.step()
                
                pred = torch.argmax(logits, dim=1)
                acc = (pred == y).float().mean()
                
                epoch_meters["loss"].update(loss.item(), x.size(0))
                epoch_meters["acc"].update(acc.item(), x.size(0))

            train_state = {k:epoch_meters[k].avg for k in epoch_meters}
            
            if epoch % self.config.eval_freq == 0:
                result_dict = self.evaluate(self._get_split("val_subset2") if self.config.split_val < 1.0 else self._get_split("val"))
                result_dict_test = self.evaluate(self._get_split("test"))
                result_dict.update(result_dict_test)
                for metric, _ in self.sel_metrics:
                    if self.best_meters[metric].add(result_dict[metric]):
                        self.save(epoch, self.best_meters[metric].get(), 
                                os.path.join(output_dir, f"best_{metric}_model.pt"))
                
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
                msg += ', ' + ', '.join([f"{k}:{v:.6f}" for k,v in result_dict.items()])
                elapsed_time = timer.t()
                est_all_time = elapsed_time / epoch * self.config.epoch
                log(f"[Epoch {epoch}] {msg}, lr:{self.optimizer_cls.param_groups[0]['lr']:.6f} ({time_str(elapsed_time)}/{time_str(est_all_time)})")

            if self.config.save_freq > 0 and epoch % self.config.save_freq == 0:
                self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), 
                         os.path.join(output_dir, f"model_epoch{epoch}.pt"))

        self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), 
                 os.path.join(output_dir, "latest_model.pt"))

    def save(self, epoch, sel_metric, file_path):
        save_dict = {}
        save_dict["model_sd"] = self.model.state_dict()
        save_dict["sel_metric"] = sel_metric
        save_dict["config"] = self.config
        save_dict["optimizer"] = self.optimizer_cls.state_dict()
        save_dict["scheduler"] = self.scheduler_cls.state_dict() if self.scheduler_cls else None
        save_dict["epoch"] = epoch
        torch.save(save_dict, file_path)

    def test(self, output_dir, split=["test"], result_path=""):
        model_info = f"evidential_alignment {self.config.dataset} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f} class_balanced:{self.config.balance_classes} temperature:{self.config.temperature} kl_reg:{self.config.kl_reg} reg_weight:{self.config.reg_weight} seed:{self.config.seed}"
        if len(result_path) > 0:
            with open(result_path, "a") as fout:
                fout.write(model_info)
                fout.write('\n')
        model_paths = []
        for metric, _ in self.sel_metrics:
            model_path = os.path.join(output_dir, f"best_{metric}_model.pt")
            model_paths.append((model_path,metric))
        model_paths.append((os.path.join(output_dir, "latest_model.pt"),"latest"))
        for model_path, metric in model_paths:
            saved_dict = self.load_check_point(model_path)
            model_dict = saved_dict["model_sd"]
            sel_metric_val = saved_dict["sel_metric"]
            self.model.load_state_dict(model_dict)
            for sp in split:
                results = self.evaluate("embed_"+sp)
                result_str = f"[{sp} ({metric}:{sel_metric_val:.6f})]: " + ', '.join([f"{k}:{results[k]:.6f}" for k in results])
                log(result_str) 
                if len(result_path) > 0:
                    with open(result_path, "a") as fout:
                        fout.write(result_str)
                        fout.write('\n')
    
