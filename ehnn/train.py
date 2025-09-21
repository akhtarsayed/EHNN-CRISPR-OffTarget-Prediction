from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers

from .model import EHNNHybrid
from .plots import plot_calibration_belt, plot_individual_plots, plot_mcc_curve, plot_rocs
from .utils import get_logger, set_seed

logger = get_logger()

def tpr_at_fpr(y_true: np.ndarray, y_pred: np.ndarray, fpr_level: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    idx = np.where(fpr <= fpr_level)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0

def _prepare_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_xor = np.array([np.array(x) for x in df["xor_sequence"]], dtype="float32")[..., None]
    X_kmer = np.array([np.array(x) for x in df["kmer_sequence"]], dtype="float32")[..., None]
    X_num = df.drop(columns=["xor_sequence", "kmer_sequence", "encoded_on_pam", "encoded_off_pam", "label"]).values.astype("float32")
    y = df["label"].values.astype("float32")
    return X_xor, X_kmer, X_num, y

def train_single(pkl_path: Path, *, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3,
                 val_split: float = 0.2, test_size: float = 0.2, random_state: int = 42,
                 threshold: float = 0.3) -> Dict[str, float | np.ndarray]:
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    X_xor, X_kmer, X_num, y = _prepare_arrays(df)

    # handle edge case: stratify requires >= 2 classes
    stratify = y if len(np.unique(y)) > 1 else None
    (x_xor_tr,  x_xor_te,
     x_kmer_tr, x_kmer_te,
     x_num_tr,  x_num_te,
     y_tr,      y_te) = train_test_split(
        X_xor, X_kmer, X_num, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    norm = layers.Normalization()
    norm.adapt(x_num_tr)
    model = EHNNHybrid(norm)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    model.fit([x_xor_tr, x_kmer_tr, x_num_tr], y_tr, validation_split=val_split,
              epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=1)

    y_pred = model.predict([x_xor_te, x_kmer_te, x_num_te], verbose=0).flatten()
    y_pred_cls = (y_pred >= threshold).astype(int)
    return {
        "y_true": y_te,
        "y_pred": y_pred,
        "roc_auc": roc_auc_score(y_te, y_pred) if len(np.unique(y_te)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred_cls, zero_division=0),
        "recall": recall_score(y_te, y_pred_cls, zero_division=0),
        "precision": precision_score(y_te, y_pred_cls, zero_division=0),
        "accuracy": accuracy_score(y_te, y_pred_cls),
        "mcc": matthews_corrcoef(y_te, y_pred_cls) if len(np.unique(y_te)) > 1 else float("nan"),
        "brier": brier_score_loss(y_te, y_pred),
        "tpr@1%fpr": tpr_at_fpr(y_te, y_pred, 0.01),
    }

def train_on_dir(in_dir: Path, out_dir: Path, **train_kwargs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results, metrics = {}, []
    for pkl in sorted(in_dir.glob("*.pkl")):
        try:
            logger.info(f"[train] Training on {pkl.name}")
            m = train_single(pkl, **train_kwargs)
            name = pkl.stem
            results[name] = {"y_true": m["y_true"], "y_pred": m["y_pred"]}
            metrics.append({k: v for k, v in m.items() if k not in ("y_true", "y_pred")} | {"dataset": name})
        except Exception as e:
            logger.error(f"[train] Failed on {pkl.name}: {e}")

    pd.DataFrame(metrics).to_csv(out_dir / "full_results_EHNN.csv", index=False)
    if results:
        plot_rocs(results, out_dir)
        plot_mcc_curve(results, out_dir)
        plot_calibration_belt(results, out_dir)
        plot_individual_plots(results, out_dir / "individual")
    logger.info("[train] Completed.")
