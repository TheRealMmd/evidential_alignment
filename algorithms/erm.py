import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from .algorithm import Algorithm
from .utils import init_optimizer, init_scheduler
from .register import register_algorithm

from models.classifier import Classifier

from utils import (
    AverageMeter,
    BestMetric,
    Timer,
    time_str,
    log,
)


@register_algorithm("erm")
class ERM(Algorithm):
    """
    Standard repository ERM, extended only for diagnostics.

    Training behavior is unchanged:
        * Classifier(backbone, n_classes, pretrained)
        * all model parameters optimized by the configured optimizer
        * ordinary unweighted cross-entropy

    Added diagnostics:
        * post-epoch TRAIN accuracy/WGA on train_no_aug when available
        * validation accuracy/WGA
        * test accuracy/WGA

    IMPORTANT:
        TEST metrics are diagnostic only. They are NOT used for checkpoint
        selection. Checkpoints are still selected exclusively by validation
        metrics, preserving the original ERM selection behavior.
    """

    def __init__(self, config):
        super(ERM, self).__init__(config)

        self._init_model()
        self._init_training()

    # ========================================================
    # Initialization
    # ========================================================

    def _init_training(self):

        self._optimizer = init_optimizer(
            self.model,
            self.config.optimizer,
            self.config.optimizer_kwargs,
        )

        self._scheduler = init_scheduler(
            self._optimizer,
            self.config.scheduler,
            self.config.scheduler_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.sel_metrics = [
            ("val_acc", True),
            ("val_worst_cls_acc", True),
            ("val_worst_group_acc", True),
            ("val_avg_cls_diff", False),
        ]

        self.best_meters = {
            metric: BestMetric(max_val)
            for metric, max_val in self.sel_metrics
        }

    def _init_model(self):

        self.n_classes = self.datasets["train"].n_classes

        self.device = f"cuda:{self.config.gpu}"

        self.model = Classifier(
            self.config.backbone,
            self.n_classes,
            self.config.pretrained,
        )

        if self.config.check_point:

            saved_dict = self.load_check_point(
                self.config.check_point
            )

            model_sd = saved_dict["model_sd"]

            self.model.load_state_dict(
                model_sd
            )

        self.model.to(
            self.device
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def _train_eval_split(self, requested_train_split):
        """
        Prefer the augmentation-free training split for evaluation.

        This is important because evaluating accuracy through the normal
        shuffled/augmented training loader is not the clean question:

            "How accurately does the final model classify the training
             examples themselves?"

        train_no_aug answers that question directly.
        """

        if "train_no_aug" in self.dataloaders:
            return "train_no_aug"

        return requested_train_split

    def _evaluate_available_split(self, split):
        """
        Evaluate one split if it exists.
        """

        if split not in self.dataloaders:
            return None

        return self.evaluate(
            split
        )

    @staticmethod
    def _format_metrics(result_dict):
        """
        Compact readable metric string.
        """

        if result_dict is None:
            return "N/A"

        return ", ".join(
            f"{key}:{value:.6f}"
            for key, value in result_dict.items()
        )

    # ========================================================
    # Training
    # ========================================================

    def train(self, output_dir, split="train"):

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        timer = Timer()

        train_loader = self.dataloaders[
            split
        ]

        # Clean post-epoch training evaluation.
        train_eval_split = self._train_eval_split(
            split
        )

        history_path = os.path.join(
            output_dir,
            "erm_train_val_test_history.csv",
        )

        history_rows = []

        for epoch in range(
            1,
            self.config.epoch + 1,
        ):

            # ------------------------------------------------
            # Normal ERM optimization
            # ------------------------------------------------

            self.model.train()

            epoch_meters = {
                key: AverageMeter()
                for key in [
                    "loss",
                    "acc",
                ]
            }

            if self._scheduler:
                curr_lr = (
                    self._scheduler
                    .get_last_lr()[0]
                )
            else:
                curr_lr = (
                    self.config
                    .optimizer_kwargs["lr"]
                )

            for batch in tqdm(
                train_loader,
                desc=split,
                leave=False,
            ):

                x, y, _, _ = batch

                x = x.to(
                    self.device
                )

                y = y.to(
                    self.device
                )

                self._optimizer.zero_grad()

                logits = self.model(
                    x
                )

                loss = self.criterion(
                    logits,
                    y,
                )

                loss.backward()

                self._optimizer.step()

                epoch_meters["loss"].update(
                    loss.item(),
                    x.size(0),
                )

                batch_acc = (
                    torch.argmax(
                        logits,
                        dim=-1,
                    )
                    == y
                ).float().mean()

                epoch_meters["acc"].update(
                    batch_acc.item(),
                    len(y),
                )

            if self._scheduler:
                self._scheduler.step()

            # Online accuracy is the accuracy observed during SGD,
            # before/while parameters are changing throughout the epoch.
            train_state = {
                key: epoch_meters[key].avg
                for key in epoch_meters
            }

            # ------------------------------------------------
            # POST-EPOCH evaluation on fixed model parameters
            # ------------------------------------------------

            if (
                epoch % self.config.eval_freq == 0
                or epoch == self.config.epoch
            ):

                clean_train_results = (
                    self._evaluate_available_split(
                        train_eval_split
                    )
                )

                val_results = (
                    self._evaluate_available_split(
                        "val"
                    )
                )

                test_results = (
                    self._evaluate_available_split(
                        "test"
                    )
                )

                if val_results is None:
                    raise RuntimeError(
                        "Validation split is required "
                        "for ERM checkpoint selection."
                    )

                # --------------------------------------------
                # ORIGINAL checkpoint selection:
                # validation metrics only.
                # --------------------------------------------

                for metric, _ in self.sel_metrics:

                    if self.best_meters[
                        metric
                    ].add(
                        val_results[metric]
                    ):

                        self.save(
                            epoch,
                            self.best_meters[
                                metric
                            ].get(),
                            os.path.join(
                                output_dir,
                                f"best_{metric}_model.pt",
                            ),
                        )

                # --------------------------------------------
                # Logging
                # --------------------------------------------

                elapsed_time = timer.t()

                est_all_time = (
                    elapsed_time
                    / epoch
                    * self.config.epoch
                )

                log(
                    f"[ERM Epoch {epoch}] "
                    f"online_train_loss={train_state['loss']:.6f}, "
                    f"online_train_acc={100.0 * train_state['acc']:.2f}%"
                )

                log(
                    f"[ERM CLEAN TRAIN = {train_eval_split}] "
                    + self._format_metrics(
                        clean_train_results
                    )
                )

                log(
                    "[ERM VAL] "
                    + self._format_metrics(
                        val_results
                    )
                )

                log(
                    "[ERM TEST - diagnostic only] "
                    + self._format_metrics(
                        test_results
                    )
                )

                log(
                    f"[ERM Epoch {epoch}] "
                    f"lr={curr_lr:.6f} "
                    f"({time_str(elapsed_time)}/"
                    f"{time_str(est_all_time)})"
                )

                # --------------------------------------------
                # CSV history
                # --------------------------------------------

                def get_metric(
                    result,
                    suffix,
                ):
                    if result is None:
                        return np.nan

                    # evaluate(split) prefixes keys with split name.
                    exact_candidates = [
                        key
                        for key in result
                        if key.endswith(
                            suffix
                        )
                    ]

                    if not exact_candidates:
                        return np.nan

                    return result[
                        exact_candidates[0]
                    ]

                history_rows.append(
                    [
                        epoch,
                        train_state["loss"],
                        train_state["acc"],

                        get_metric(
                            clean_train_results,
                            "_acc",
                        ),
                        get_metric(
                            clean_train_results,
                            "_worst_group_acc",
                        ),

                        get_metric(
                            val_results,
                            "_acc",
                        ),
                        get_metric(
                            val_results,
                            "_worst_group_acc",
                        ),

                        get_metric(
                            test_results,
                            "_acc",
                        ),
                        get_metric(
                            test_results,
                            "_worst_group_acc",
                        ),
                    ]
                )

                with open(
                    history_path,
                    "w",
                    newline="",
                ) as f:

                    writer = csv.writer(
                        f
                    )

                    writer.writerow(
                        [
                            "epoch",
                            "online_train_loss",
                            "online_train_accuracy",
                            "clean_train_accuracy",
                            "clean_train_wga",
                            "val_accuracy",
                            "val_wga",
                            "test_accuracy",
                            "test_wga",
                        ]
                    )

                    writer.writerows(
                        history_rows
                    )

            # ------------------------------------------------
            # Optional intermediate checkpoint
            # ------------------------------------------------

            if (
                self.config.save_freq > 0
                and epoch
                % self.config.save_freq
                == 0
            ):

                self.save(
                    epoch,
                    self.best_meters[
                        "val_acc"
                    ].get(),
                    os.path.join(
                        output_dir,
                        f"model_epoch{epoch}.pt",
                    ),
                )

        # Latest model, identical conceptually to original code.
        self.save(
            epoch,
            self.best_meters[
                "val_acc"
            ].get(),
            os.path.join(
                output_dir,
                "latest_model.pt",
            ),
        )

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        epoch,
        sel_metric,
        file_path,
    ):

        save_dict = {}

        save_dict[
            "model_sd"
        ] = self.model.state_dict()

        save_dict[
            "sel_metric"
        ] = sel_metric

        save_dict[
            "config"
        ] = self.config

        save_dict[
            "optimizer"
        ] = self._optimizer.state_dict()

        save_dict[
            "scheduler"
        ] = (
            self._scheduler.state_dict()
            if self._scheduler
            else None
        )

        save_dict[
            "epoch"
        ] = epoch

        torch.save(
            save_dict,
            file_path,
        )

    # ========================================================
    # Test / checkpoint report
    # ========================================================

    def test(
        self,
        output_dir,
        split=("test",),
        result_path="",
    ):

        model_info = (
            f"erm "
            f"{self.config.dataset} "
            f"{self.config.backbone} "
            f"{self.config.train_split} "
            f"train_ratio:{self.config.split_train:.2f} "
            f"val_ratio:{self.config.split_val:.2f} "
            f"seed:{self.config.seed}"
        )

        if len(result_path) > 0:

            with open(
                result_path,
                "a",
            ) as fout:

                fout.write(
                    model_info
                )

                fout.write(
                    "\n"
                )

        # Report train + val + requested test splits.
        report_splits = []

        train_eval_split = self._train_eval_split(
            self.config.train_split
        )

        for sp in [
            train_eval_split,
            "val",
            *(
                [split]
                if isinstance(split, str)
                else list(split)
            ),
        ]:
            if (
                sp in self.dataloaders
                and sp not in report_splits
            ):
                report_splits.append(
                    sp
                )

        model_paths = []

        for metric, _ in self.sel_metrics:

            model_path = os.path.join(
                output_dir,
                f"best_{metric}_model.pt",
            )

            model_paths.append(
                (
                    model_path,
                    metric,
                )
            )

        model_paths.append(
            (
                os.path.join(
                    output_dir,
                    "latest_model.pt",
                ),
                "latest",
            )
        )

        for model_path, metric in model_paths:

            if not os.path.exists(
                model_path
            ):
                continue

            saved_dict = self.load_check_point(
                model_path
            )

            model_dict = saved_dict[
                "model_sd"
            ]

            sel_metric_val = saved_dict[
                "sel_metric"
            ]

            self.model.load_state_dict(
                model_dict
            )

            for sp in report_splits:

                results = self.evaluate(
                    sp
                )

                result_str = (
                    f"[{sp} "
                    f"({metric}:"
                    f"{sel_metric_val:.6f})]: "
                    + ", ".join(
                        f"{key}:"
                        f"{results[key]:.6f}"
                        for key in results
                    )
                )

                log(
                    result_str
                )

                if len(
                    result_path
                ) > 0:

                    with open(
                        result_path,
                        "a",
                    ) as fout:

                        fout.write(
                            result_str
                        )

                        fout.write(
                            "\n"
                        )
