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


    # ============================================================
    # Diagnostics helpers
    # ============================================================
