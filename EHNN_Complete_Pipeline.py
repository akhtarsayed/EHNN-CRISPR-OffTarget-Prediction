# ======================================================================
# EHNN – End-to-End CRISPR Off-Target Predictor
# Pipeline: Feature Extraction → Normalization → Hybrid Encoding → Model Training
# 2025-08-02 – fixes: single-row CSV, numeric legends, individual plots, extra metrics
# ======================================================================

import os
import re
import glob
import pickle
from pathlib import Path
from itertools import product
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, accuracy_score,
    roc_curve, precision_recall_curve, auc, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

from Bio.SeqUtils import gc_fraction
from Bio.Seq import Seq
from tensorflow.keras import layers, callbacks
from tensorflow.keras import Model as KModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================================================================
# 1. Feature Extraction
# ======================================================================

REGULATORY = {'TATA_box': 'TATAAA', 'CAAT_box': 'CCAAT', 'GC_box': 'GGGCGG'}
RESTRICTION = {'EcoRI': 'GAATTC', 'BamHI': 'GGATCC', 'HindIII': 'AAGCTT'}

def calculate_features(seq: str):
    seq_obj = Seq(seq)
    gc = gc_fraction(seq_obj)
    length = len(seq)
    pam = seq[-3:] if length >= 3 else seq
    self_comp = sum(seq[i] == seq[-(i+1)] for i in range(length // 2))
    at = (seq.count('A') + seq.count('T')) / max(length, 1)
    di = {pair: seq.count(pair) for pair in
          ['AA','AT','AG','AC','TT','TA','TG','TC','GG','GA','GT','GC','CC','CA','CT','CG']}
    return {
        "GC_Content": gc,
        "AT_Content": at,
        "Length": length,
        "PAM_Sequence": pam,
        "Self_Complementarity": self_comp,
        **di
    }

def process_file(csv_path):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        on = r['on_seq']
        off = r['off_seq']
        label = r['label']
        on_f = calculate_features(on)
        off_f = calculate_features(off)
        pair = {
            "on_seq": on, "off_seq": off, "label": label,
            **{f"on_{k}": v for k, v in on_f.items()},
            **{f"off_{k}": v for k, v in off_f.items()},
            "Hamming_Distance": sum(a != b for a, b in zip(on, off)) if len(on) == len(off) else np.nan,
            "Mismatch_Count": sum(a != b for a, b in zip(on, off)) if len(on) == len(off) else np.nan
        }
        rows.append(pair)
    return pd.DataFrame(rows)

def feature_main():
    in_dir = Path("original_dataset")
    out_dir = Path("feature_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in in_dir.glob("*.csv"):
        logging.info(f"Processing {f}")
        df = process_file(f)
        df.to_csv(out_dir / f"{f.stem}_feature.csv", index=False)
    logging.info("Feature extraction completed.")

# ======================================================================
# 2. Normalisation
# ======================================================================

def normalize_main():
    in_dir, out_dir = Path("feature_dataset"), Path("normalize_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler()
    for f in in_dir.glob("*.csv"):
        df = pd.read_csv(f)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'label' in num_cols:
            num_cols.remove('label')
        if num_cols:
            df[num_cols] = scaler.fit_transform(df[num_cols])
        df.to_csv(out_dir / f"{f.stem}_normalized.csv", index=False)
    logging.info("Normalization completed.")

# ======================================================================
# 3. Hybrid Encoding (XOR + k-mer + PAM)
# ======================================================================

NT_BITS = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
K = 3
KMER_VOCAB = {''.join(p): idx for idx, p in enumerate(product('ACGT', repeat=K))}
PAM_MAP = {p: i for i, p in enumerate(['TGG', 'GAG', 'GGG', 'Other'])}

def encode_xor(off: str, on: str):
    return [NT_BITS.get(o, 0) ^ NT_BITS.get(i, 0) for o, i in zip(off.upper(), on.upper())]

def encode_kmer(seq: str):
    vec = [KMER_VOCAB.get(seq[i:i+K].upper(), len(KMER_VOCAB)) for i in range(len(seq) - K + 1)]
    return vec if vec else [len(KMER_VOCAB)]

def encode_pam(pam: str):
    return PAM_MAP.get(pam.upper(), PAM_MAP['Other'])

def hybrid_encoding_main():
    in_dir, out_dir = Path("normalize_dataset"), Path("encoded_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ['on_seq', 'off_seq', 'on_PAM_Sequence', 'off_PAM_Sequence', 'label']
    for file in in_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if not set(required).issubset(df.columns):
            logging.warning(f"Skipping {file} (missing columns)")
            continue
        df.fillna(0, inplace=True)

        xor = [encode_xor(r.off_seq, r.on_seq) for _, r in df.iterrows()]
        kmer = [encode_kmer(r.off_seq) for _, r in df.iterrows()]
        pam_on = df['on_PAM_Sequence'].apply(encode_pam).values.astype('int32')
        pam_off = df['off_PAM_Sequence'].apply(encode_pam).values.astype('int32')

        max_xor = max(map(len, xor))
        max_kmer = max(map(len, kmer))
        xor = [v + [4] * (max_xor - len(v)) for v in xor]
        kmer = [v + [len(KMER_VOCAB)] * (max_kmer - len(v)) for v in kmer]

        num_cols = [c for c in df.columns if c not in required]
        X_num = df[num_cols].values.astype('float32')
        y = df['label'].values.astype('float32')

        final = pd.DataFrame(X_num, columns=num_cols)
        final['xor_sequence'] = xor
        final['kmer_sequence'] = kmer
        final['encoded_on_pam'] = pam_on
        final['encoded_off_pam'] = pam_off
        final['label'] = y

        base = file.stem
        final.to_pickle(out_dir / f"hybrid_{base}.pkl")
    logging.info("Hybrid encoding completed.")

# ======================================================================
# 4. Model Architecture
# ======================================================================

class TimeAttention(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attn = layers.Dense(1, activation='tanh')

    def call(self, inputs, **kwargs):
        w = tf.nn.softmax(self.attn(inputs), axis=1)
        return tf.reduce_sum(w * inputs, axis=1)

class EHNNHybrid(KModel):
    def __init__(self, num_norm):
        super().__init__()
        self.cnn_xor = tf.keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            TimeAttention()
        ])
        self.cnn_kmer = tf.keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            TimeAttention()
        ])
        self.lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPooling1D()
        ])
        self.num_mlp = tf.keras.Sequential([
            num_norm,
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ])
        self.concat = layers.Concatenate()
        self.head = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, **kwargs):
        x_xor, x_kmer, x_num = inputs
        c1 = self.cnn_xor(x_xor)
        c2 = self.cnn_kmer(x_kmer)
        c  = self.concat([c1, c2])
        lstm_out = self.lstm(tf.expand_dims(c, axis=1))
        num_out  = self.num_mlp(x_num)
        fused = self.concat([c1, c2, lstm_out, num_out])
        return self.head(fused)

# ======================================================================
# 5. Training Utilities
# ======================================================================

class LRTracker(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()

def tpr_at_fpr(y_true, y_pred, fpr_level=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    idx = np.where(fpr <= fpr_level)[0]
    return tpr[idx[-1]] if len(idx) else 0.0

def train_model(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)

    X_xor  = np.array([np.array(x) for x in df['xor_sequence']])
    X_kmer = np.array([np.array(x) for x in df['kmer_sequence']])
    X_num  = df.drop(columns=['xor_sequence', 'kmer_sequence',
                              'encoded_on_pam', 'encoded_off_pam', 'label']).values.astype('float32')
    y = df['label'].values.astype('float32')

    X_xor  = np.expand_dims(X_xor,  axis=-1)
    X_kmer = np.expand_dims(X_kmer, axis=-1)

    (x_xor_tr,  x_xor_te,
     x_kmer_tr, x_kmer_te,
     x_num_tr,  x_num_te,
     y_tr,      y_te) = train_test_split(
        X_xor, X_kmer, X_num, y,
        test_size=0.2, random_state=42, stratify=y)

    norm = layers.Normalization()
    norm.adapt(x_num_tr)

    model = EHNNHybrid(norm)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    model.fit(
        [x_xor_tr, x_kmer_tr, x_num_tr], y_tr,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            LRTracker()
        ],
        verbose=1
    )

    y_pred = model.predict([x_xor_te, x_kmer_te, x_num_te]).flatten()
    y_pred_cls = (y_pred >= 0.3).astype(int)
    return {
        'y_true': y_te,
        'y_pred': y_pred,
        'roc_auc': roc_auc_score(y_te, y_pred),
        'pr_auc': average_precision_score(y_te, y_pred),
        'f1': f1_score(y_te, y_pred_cls, zero_division=0),
        'recall': recall_score(y_te, y_pred_cls, zero_division=0),
        'precision': precision_score(y_te, y_pred_cls, zero_division=0),
        'accuracy': accuracy_score(y_te, y_pred_cls),
        'mcc': matthews_corrcoef(y_te, y_pred_cls),
        'brier': brier_score_loss(y_te, y_pred),
        'tpr@1%fpr': tpr_at_fpr(y_te, y_pred, 0.01)
    }

# ======================================================================
# 6. Plotting
# ======================================================================

plt.style.use("ggplot")
sns.set_palette("Set2")

def plot_rocs(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1,3,figsize=(18,5))
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax[0].plot(fpr, tpr, label=f"{name} (AUROC={auc(fpr,tpr):.3f})")
        # PR
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        ax[1].plot(rec, pre, label=f"{name} (AUPR={auc(rec,pre):.3f})")
        # SS-curve
        thr = np.linspace(0,1,100)
        sens = [((y_true==1)&(y_pred>=t)).sum()/(y_true==1).sum() for t in thr]
        spec = [((y_true==0)&(y_pred<t)).sum()/(y_true==0).sum() for t in thr]
        ax[2].plot(1-np.array(spec), sens, label=f"{name}")
    ax[0].legend(); ax[0].set_title("ROC")
    ax[1].legend(); ax[1].set_title("Precision-Recall")
    ax[2].legend(); ax[2].set_title("Sensitivity-Specificity")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/CRISPR_ROC_SS_PR.png", dpi=300); plt.close()

def plot_mcc_curve(results, save_dir):
    plt.figure(figsize=(6,5))
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        thr = np.linspace(0,1,100)
        mcc = [matthews_corrcoef(y_true, y_pred>=t) for t in thr]
        plt.plot(thr, mcc, label=f"{name} (maxMCC={max(mcc):.3f})")
    plt.xlabel("Threshold"); plt.ylabel("MCC"); plt.title("MCC vs Threshold")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/CRISPR_MCC_curve.png", dpi=300); plt.close()

def plot_calibration_belt(results, save_dir):
    plt.figure(figsize=(6,6))
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        brier = brier_score_loss(y_true, y_pred)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier={brier:.3f})")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("Mean predicted"); plt.ylabel("Observed")
    plt.title("Calibration Belt"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/CRISPR_calibration.png", dpi=300); plt.close()

def plot_individual_plots(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(); plt.plot(fpr, tpr)
        plt.title(f"{name} ROC (AUROC={auc(fpr,tpr):.3f})")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(f"{save_dir}/{name}_roc.png", dpi=300); plt.close()
        # PR
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(); plt.plot(rec, pre)
        plt.title(f"{name} PR (AUPR={auc(rec,pre):.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(f"{save_dir}/{name}_pr.png", dpi=300); plt.close()
        # MCC
        thr = np.linspace(0,1,100)
        mcc = [matthews_corrcoef(y_true, y_pred>=t) for t in thr]
        plt.figure(); plt.plot(thr, mcc)
        plt.title(f"{name} MCC (max={max(mcc):.3f})")
        plt.xlabel("Threshold"); plt.ylabel("MCC")
        plt.savefig(f"{save_dir}/{name}_mcc.png", dpi=300); plt.close()

# ======================================================================
# 7. Orchestrator
# ======================================================================

def train_main():
    in_dir = Path("encoded_dataset")
    out_dir = Path("Final_Results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results, metrics = {}, []
    for f in in_dir.glob("*.pkl"):
        try:
            m = train_model(f)
            name = f.stem
            results[name] = {'y_true': m['y_true'], 'y_pred': m['y_pred']}
            # drop arrays for CSV
            metrics.append({
                'dataset': name,
                'roc_auc': m['roc_auc'],
                'pr_auc': m['pr_auc'],
                'f1': m['f1'],
                'recall': m['recall'],
                'precision': m['precision'],
                'accuracy': m['accuracy'],
                'mcc': m['mcc'],
                'brier': m['brier'],
                'tpr@1%fpr': m['tpr@1%fpr']
            })
        except Exception as e:
            logging.error(f"Model failed on {f}: {e}")

    pd.DataFrame(metrics).to_csv(out_dir / "full_results_EHNN.csv", index=False)
    plot_rocs(results, out_dir)
    plot_mcc_curve(results, out_dir)
    plot_calibration_belt(results, out_dir)
    plot_individual_plots(results, out_dir / "individual")
    logging.info("EHNN pipeline completed.")

# ======================================================================
# 8. Entry Point
# ======================================================================

if __name__ == "__main__":
    # feature_main()
    # normalize_main()
    # hybrid_encoding_main()
    train_main()