# src/callbacks/confusion_confidence.py
import io
import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Callback, Trainer, LightningModule
from torchmetrics.classification import MulticlassConfusionMatrix

@dataclass
class ConfusionConfidenceConfig:
    num_classes: int
    class_names: Optional[List[str]] = None
    cm_normalize: bool = True
    reliability_bins: int = 15
    log_images: bool = True
    max_per_class_hist: int = 8  # how many per-class histograms to log

class ConfusionAndConfidenceCallback(Callback):
    """
    Collects validation logits/targets and logs:
    - confusion matrix (counts + normalized)
    - confidence histograms (overall + per-class)
    - reliability diagram + ECE
    Works with TensorBoardLogger.
    """
    def __init__(self, cfg: ConfusionConfidenceConfig):
        super().__init__()
        self.cfg = cfg
        self.cm_metric = MulticlassConfusionMatrix(num_classes=cfg.num_classes)
        self._val_logits = []
        self._val_targets = []

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        # Expect model's step to return logits in outputs or recompute here
        # Safer to recompute to avoid assumptions:
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)  # [B, C]
        self._val_logits.append(logits.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits, dim=0)  # [N, C]
        targets = torch.cat(self._val_targets, dim=0)  # [N]
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # --- Confusion matrix ---
        cm_counts = self.cm_metric(preds, targets).cpu().numpy()  # counts
        cm_norm = cm_counts / cm_counts.sum(axis=1, keepdims=True).clip(min=1e-9)

        # --- ECE / reliability ---
        ece, bin_conf, bin_acc, bin_count = self._compute_ece(probs, targets, self.cfg.reliability_bins)

        # --- Log to TensorBoard ---
        tb = getattr(trainer.logger, "experiment", None)
        global_step = trainer.global_step
        tag_prefix = "val"

        if tb is not None and self.cfg.log_images:
            # Confusion matrices
            tb.add_figure(f"{tag_prefix}/confusion_matrix_counts", self._plot_cm(cm_counts, normalize=False), global_step)
            tb.add_figure(f"{tag_prefix}/confusion_matrix_normalized", self._plot_cm(cm_norm, normalize=True), global_step)

            # Confidence histograms
            tb.add_figure(f"{tag_prefix}/confidence_hist_overall", self._plot_conf_hist(probs, preds, targets), global_step)

            # Per-class histograms (cap how many to avoid spamming)
            K = min(self.cfg.num_classes, self.cfg.max_per_class_hist)
            for c in range(K):
                tb.add_figure(f"{tag_prefix}/confidence_hist_class_{c}", self._plot_conf_hist(probs, preds, targets, cls=c), global_step)

            # Reliability diagram
            tb.add_figure(f"{tag_prefix}/reliability_diagram", self._plot_reliability(bin_conf, bin_acc, bin_count, ece), global_step)

        # Scalars
        trainer.logger.log_metrics({f"{tag_prefix}/ece": float(ece)}, step=global_step)

        # reset state
        self.cm_metric.reset()
        self._val_logits.clear()
        self._val_targets.clear()

    # ---------- helpers ----------

    def _compute_ece(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15):
        """
        ECE with max-prob calibration (top-1). Returns tuple:
        (ece, bin_confidence, bin_accuracy, bin_count)
        """
        conf, pred = probs.max(dim=1)
        correct = (pred == targets).to(torch.float32)

        bins = torch.linspace(0, 1, steps=n_bins + 1)
        bin_ids = torch.bucketize(conf, bins) - 1  # [0..n_bins-1]
        bin_conf, bin_acc, bin_count = [], [], []

        ece = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            cnt = mask.sum().item()
            if cnt == 0:
                bin_conf.append(0.0); bin_acc.append(0.0); bin_count.append(0)
                continue
            conf_b = conf[mask].mean().item()
            acc_b = correct[mask].mean().item()
            bin_conf.append(conf_b)
            bin_acc.append(acc_b)
            bin_count.append(cnt)
            ece += (cnt / len(conf)) * abs(acc_b - conf_b)

        return float(ece), np.array(bin_conf), np.array(bin_acc), np.array(bin_count)

    def _plot_cm(self, cm: np.ndarray, normalize: bool):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(f"Confusion Matrix ({'norm' if normalize else 'counts'})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        ticks = np.arange(self.cfg.num_classes)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        labels = self.cfg.class_names if self.cfg.class_names else [str(i) for i in range(self.cfg.num_classes)]
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)

        # annotate
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        fig.tight_layout()
        return fig

    def _plot_conf_hist(self, probs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, cls: Optional[int] = None):
        """
        Hist of max prob for correct vs incorrect; optionally filter by true class.
        """
        with torch.no_grad():
            conf, pred = probs.max(dim=1)
            mask = torch.ones_like(conf, dtype=torch.bool) if cls is None else (targets == cls)

            conf = conf[mask]
            correct = (pred[mask] == targets[mask]).cpu().numpy()
            conf = conf.cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(conf[correct.astype(bool)], bins=20, alpha=0.7, label="correct")
        ax.hist(conf[~correct.astype(bool)], bins=20, alpha=0.7, label="incorrect")
        title = "Confidence (overall)" if cls is None else f"Confidence (true class={cls})"
        ax.set_title(title)
        ax.set_xlabel("max softmax probability"); ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_reliability(self, bin_conf, bin_acc, bin_count, ece: float):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.bar(bin_conf, bin_acc, width=1.0 / max(len(bin_conf), 1), align='center', alpha=0.7, label="accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability Diagram (ECE={ece:.4f})")
        fig.tight_layout()
        return fig
