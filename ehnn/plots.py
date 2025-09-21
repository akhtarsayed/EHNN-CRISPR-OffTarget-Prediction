from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, matthews_corrcoef, precision_recall_curve, roc_curve

def _ensure_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot")

def plot_rocs(results: Dict[str, dict], save_dir: Path) -> None:
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, res in results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax[0].plot(fpr, tpr, label=f"{name} (AUROC={auc(fpr, tpr):.3f})")
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        ax[1].plot(rec, pre, label=f"{name} (AUPR={auc(rec, pre):.3f})")
        thr = np.linspace(0, 1, 100)
        sens = [((y_true == 1) & (y_pred >= t)).sum() / (y_true == 1).sum() for t in thr]
        spec = [((y_true == 0) & (y_pred < t)).sum() / (y_true == 0).sum() for t in thr]
        ax[2].plot(1 - np.array(spec), sens, label=f"{name}")
    ax[0].legend(); ax[0].set_title("ROC")
    ax[1].legend(); ax[1].set_title("Precision-Recall")
    ax[2].legend(); ax[2].set_title("Sensitivity-Specificity")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "CRISPR_ROC_SS_PR.png", dpi=300); plt.close()

def plot_mcc_curve(results: Dict[str, dict], save_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        thr = np.linspace(0, 1, 100)
        mcc = [matthews_corrcoef(y_true, y_pred >= t) for t in thr]
        plt.plot(thr, mcc, label=f"{name} (maxMCC={max(mcc):.3f})")
    plt.xlabel("Threshold"); plt.ylabel("MCC"); plt.title("MCC vs Threshold")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(save_dir) / "CRISPR_MCC_curve.png", dpi=300); plt.close()

def plot_calibration_belt(results: Dict[str, dict], save_dir: Path) -> None:
    plt.figure(figsize=(6, 6))
    for name, res in results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("Mean predicted"); plt.ylabel("Observed")
    plt.title("Calibration Belt"); plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "CRISPR_calibration.png", dpi=300); plt.close()

def plot_individual_plots(results: Dict[str, dict], save_dir: Path) -> None:
    save_dir = Path(save_dir)
    _ensure_dir(save_dir)
    for name, res in results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(); plt.plot(fpr, tpr)
        plt.title(f"{name} ROC (AUROC={auc(fpr,tpr):.3f})")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(save_dir / f"{name}_roc.png", dpi=300); plt.close()
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(); plt.plot(rec, pre)
        plt.title(f"{name} PR (AUPR={auc(rec,pre):.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(save_dir / f"{name}_pr.png", dpi=300); plt.close()
        thr = np.linspace(0,1,100)
        mcc = [matthews_corrcoef(y_true, y_pred>=t) for t in thr]
        plt.figure(); plt.plot(thr, mcc)
        plt.title(f"{name} MCC (max={max(mcc):.3f})")
        plt.xlabel("Threshold"); plt.ylabel("MCC")
        plt.savefig(save_dir / f"{name}_mcc.png", dpi=300); plt.close()
