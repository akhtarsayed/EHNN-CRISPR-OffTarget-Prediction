# ======================================================================
# EHNN – Enhanced Pipeline with Evaluation Methods + Standard Deviations
# Added: Cross-validation, Ablation study, Cross-dataset testing
# Enhanced: Standard deviations for all metrics
# ======================================================================

import os
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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
# 1-3. Feature Extraction, Normalization, Hybrid Encoding (unchanged)
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
# 4. Model Architecture (unchanged)
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
# 5. EVALUATION METHODS (ENHANCED WITH STD)
# ======================================================================

def tpr_at_fpr(y_true, y_pred, fpr_level=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    idx = np.where(fpr <= fpr_level)[0]
    return tpr[idx[-1]] if len(idx) else 0.0

def compute_metrics(y_true, y_pred, threshold=0.3):
    """Compute all evaluation metrics"""
    y_pred_cls = (y_pred >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred_cls, zero_division=0),
        'recall': recall_score(y_true, y_pred_cls, zero_division=0),
        'precision': precision_score(y_true, y_pred_cls, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred_cls),
        'mcc': matthews_corrcoef(y_true, y_pred_cls),
        'brier': brier_score_loss(y_true, y_pred),
        'tpr@1%fpr': tpr_at_fpr(y_true, y_pred, 0.01)
    }

# ======================================================================
# EVALUATION METHOD 1: K-Fold Cross-Validation (WITH STD)
# ======================================================================

def cross_validation_evaluation(X_xor, X_kmer, X_num, y, k_folds=5):
    """
    Perform stratified k-fold cross-validation to assess model robustness
    Returns: cv_df, mean_metrics, std_metrics
    """
    logging.info(f"Starting {k_folds}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_xor, y)):
        logging.info(f"Training fold {fold + 1}/{k_folds}")
        
        # Split data
        x_xor_tr, x_xor_val = X_xor[train_idx], X_xor[val_idx]
        x_kmer_tr, x_kmer_val = X_kmer[train_idx], X_kmer[val_idx]
        x_num_tr, x_num_val = X_num[train_idx], X_num[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Normalize numeric features
        norm = layers.Normalization()
        norm.adapt(x_num_tr)
        
        # Build and train model
        model = EHNNHybrid(norm)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        model.fit(
            [x_xor_tr, x_kmer_tr, x_num_tr], y_tr,
            validation_data=([x_xor_val, x_kmer_val, x_num_val], y_val),
            epochs=20,
            batch_size=128,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict([x_xor_val, x_kmer_val, x_num_val], verbose=0).flatten()
        metrics = compute_metrics(y_val, y_pred)
        metrics['fold'] = fold + 1
        cv_results.append(metrics)
    
    # Aggregate results
    cv_df = pd.DataFrame(cv_results)
    mean_metrics = cv_df.drop('fold', axis=1).mean().to_dict()
    std_metrics = cv_df.drop('fold', axis=1).std().to_dict()
    
    logging.info(f"Cross-validation completed. Mean ROC-AUC: {mean_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")
    
    return cv_df, mean_metrics, std_metrics

# ======================================================================
# EVALUATION METHOD 2: Ablation Study (WITH STD FROM MULTIPLE RUNS)
# ======================================================================

def ablation_study(X_xor, X_kmer, X_num, y, n_runs=3):
    """
    Evaluate contribution of each component by removing it
    Runs multiple times to compute std
    """
    logging.info(f"Starting ablation study with {n_runs} runs...")
    
    ablation_results = {
        'Full Model': [],
        'No XOR': [],
        'No k-mer': [],
        'No BiLSTM': [],
        'No Numeric': []
    }
    
    for run in range(n_runs):
        logging.info(f"Ablation run {run + 1}/{n_runs}")
        
        # Split data with different seed for each run
        (x_xor_tr, x_xor_te, x_kmer_tr, x_kmer_te,
         x_num_tr, x_num_te, y_tr, y_te) = train_test_split(
            X_xor, X_kmer, X_num, y,
            test_size=0.2, random_state=42 + run, stratify=y)
        
        norm = layers.Normalization()
        norm.adapt(x_num_tr)
        num_features = x_num_tr.shape[1]
        
        # 1. Full model
        model_full = EHNNHybrid(norm)
        model_full.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss='binary_crossentropy', metrics=['accuracy'])
        model_full.fit([x_xor_tr, x_kmer_tr, x_num_tr], y_tr,
                      validation_split=0.2, epochs=20, batch_size=128,
                      callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                      verbose=0)
        y_pred = model_full.predict([x_xor_te, x_kmer_te, x_num_te], verbose=0).flatten()
        ablation_results['Full Model'].append(compute_metrics(y_te, y_pred))
        
        # 2. Without XOR
        model_no_xor = build_ablated_model(norm, remove='xor', num_features=num_features)
        model_no_xor.fit([x_kmer_tr, x_num_tr], y_tr,
                        validation_split=0.2, epochs=20, batch_size=128,
                        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                        verbose=0)
        y_pred = model_no_xor.predict([x_kmer_te, x_num_te], verbose=0).flatten()
        ablation_results['No XOR'].append(compute_metrics(y_te, y_pred))
        
        # 3. Without k-mer
        model_no_kmer = build_ablated_model(norm, remove='kmer', num_features=num_features)
        model_no_kmer.fit([x_xor_tr, x_num_tr], y_tr,
                         validation_split=0.2, epochs=20, batch_size=128,
                         callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                         verbose=0)
        y_pred = model_no_kmer.predict([x_xor_te, x_num_te], verbose=0).flatten()
        ablation_results['No k-mer'].append(compute_metrics(y_te, y_pred))
        
        # 4. Without BiLSTM
        model_no_lstm = build_ablated_model(norm, remove='lstm', num_features=num_features)
        model_no_lstm.fit([x_xor_tr, x_kmer_tr, x_num_tr], y_tr,
                         validation_split=0.2, epochs=20, batch_size=128,
                         callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                         verbose=0)
        y_pred = model_no_lstm.predict([x_xor_te, x_kmer_te, x_num_te], verbose=0).flatten()
        ablation_results['No BiLSTM'].append(compute_metrics(y_te, y_pred))
        
        # 5. Without numeric
        model_no_num = build_ablated_model(norm, remove='numeric', num_features=num_features)
        model_no_num.fit([x_xor_tr, x_kmer_tr], y_tr,
                        validation_split=0.2, epochs=20, batch_size=128,
                        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                        verbose=0)
        y_pred = model_no_num.predict([x_xor_te, x_kmer_te], verbose=0).flatten()
        ablation_results['No Numeric'].append(compute_metrics(y_te, y_pred))
    
    # Compute mean and std across runs
    ablation_mean = {}
    ablation_std = {}
    
    for config, runs in ablation_results.items():
        metrics_df = pd.DataFrame(runs)
        ablation_mean[config] = metrics_df.mean().to_dict()
        ablation_std[config] = metrics_df.std().to_dict()
    
    ablation_mean_df = pd.DataFrame(ablation_mean).T
    ablation_std_df = pd.DataFrame(ablation_std).T
    
    logging.info("Ablation study completed")
    
    return ablation_mean_df, ablation_std_df

def build_ablated_model(norm, remove='xor', num_features=None):
    """Build model with specific component removed"""
    if remove == 'xor':
        kmer_input = layers.Input(shape=(None, 1))
        num_input = layers.Input(shape=(num_features,))
        
        c_kmer = layers.Conv1D(64, 3, activation='relu', padding='same')(kmer_input)
        c_kmer = layers.MaxPooling1D(2)(c_kmer)
        c_kmer = layers.Conv1D(128, 3, activation='relu', padding='same')(c_kmer)
        c_kmer = TimeAttention()(c_kmer)
        
        num_out = norm(num_input)
        num_out = layers.Dense(128, activation='relu')(num_out)
        num_out = layers.Dense(64, activation='relu')(num_out)
        
        fused = layers.Concatenate()([c_kmer, num_out])
        output = layers.Dense(256, activation='relu')(fused)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(1, activation='sigmoid')(output)
        
        model = KModel(inputs=[kmer_input, num_input], outputs=output)
        
    elif remove == 'kmer':
        xor_input = layers.Input(shape=(None, 1))
        num_input = layers.Input(shape=(num_features,))
        
        c_xor = layers.Conv1D(64, 3, activation='relu', padding='same')(xor_input)
        c_xor = layers.MaxPooling1D(2)(c_xor)
        c_xor = layers.Conv1D(128, 3, activation='relu', padding='same')(c_xor)
        c_xor = TimeAttention()(c_xor)
        
        num_out = norm(num_input)
        num_out = layers.Dense(128, activation='relu')(num_out)
        num_out = layers.Dense(64, activation='relu')(num_out)
        
        fused = layers.Concatenate()([c_xor, num_out])
        output = layers.Dense(256, activation='relu')(fused)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(1, activation='sigmoid')(output)
        
        model = KModel(inputs=[xor_input, num_input], outputs=output)
        
    elif remove == 'lstm':
        xor_input = layers.Input(shape=(None, 1))
        kmer_input = layers.Input(shape=(None, 1))
        num_input = layers.Input(shape=(num_features,))
        
        c_xor = layers.Conv1D(64, 3, activation='relu', padding='same')(xor_input)
        c_xor = layers.MaxPooling1D(2)(c_xor)
        c_xor = layers.Conv1D(128, 3, activation='relu', padding='same')(c_xor)
        c_xor = TimeAttention()(c_xor)
        
        c_kmer = layers.Conv1D(64, 3, activation='relu', padding='same')(kmer_input)
        c_kmer = layers.MaxPooling1D(2)(c_kmer)
        c_kmer = layers.Conv1D(128, 3, activation='relu', padding='same')(c_kmer)
        c_kmer = TimeAttention()(c_kmer)
        
        num_out = norm(num_input)
        num_out = layers.Dense(128, activation='relu')(num_out)
        num_out = layers.Dense(64, activation='relu')(num_out)
        
        fused = layers.Concatenate()([c_xor, c_kmer, num_out])
        output = layers.Dense(256, activation='relu')(fused)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(1, activation='sigmoid')(output)
        
        model = KModel(inputs=[xor_input, kmer_input, num_input], outputs=output)
        
    elif remove == 'numeric':
        xor_input = layers.Input(shape=(None, 1))
        kmer_input = layers.Input(shape=(None, 1))
        
        c_xor = layers.Conv1D(64, 3, activation='relu', padding='same')(xor_input)
        c_xor = layers.MaxPooling1D(2)(c_xor)
        c_xor = layers.Conv1D(128, 3, activation='relu', padding='same')(c_xor)
        c_xor = TimeAttention()(c_xor)
        
        c_kmer = layers.Conv1D(64, 3, activation='relu', padding='same')(kmer_input)
        c_kmer = layers.MaxPooling1D(2)(c_kmer)
        c_kmer = layers.Conv1D(128, 3, activation='relu', padding='same')(c_kmer)
        c_kmer = TimeAttention()(c_kmer)
        
        c = layers.Concatenate()([c_xor, c_kmer])
        c_expanded = layers.Reshape((1, -1))(c)
        lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(c_expanded)
        lstm_out = layers.GlobalMaxPooling1D()(lstm_out)
        
        fused = layers.Concatenate()([c_xor, c_kmer, lstm_out])
        output = layers.Dense(256, activation='relu')(fused)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(1, activation='sigmoid')(output)
        
        model = KModel(inputs=[xor_input, kmer_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ======================================================================
# EVALUATION METHOD 3: Cross-Dataset Testing (WITH STD)
# ======================================================================

def cross_dataset_evaluation(encoded_dir):
    """
    Train on one dataset, test on all others to assess generalization
    Returns: cross_df with results, mean_summary, std_summary
    """
    logging.info("Starting cross-dataset evaluation...")
    encoded_dir = Path(encoded_dir)
    dataset_files = list(encoded_dir.glob("*.pkl"))
    
    if len(dataset_files) < 2:
        logging.warning("Need at least 2 datasets for cross-dataset testing")
        return None, None, None
    
    cross_results = []
    
    for train_file in dataset_files:
        train_name = train_file.stem
        logging.info(f"Training on: {train_name}")
        
        # Load training data
        with open(train_file, 'rb') as f:
            df_train = pickle.load(f)
        
        X_xor_tr = np.array([np.array(x) for x in df_train['xor_sequence']])
        X_kmer_tr = np.array([np.array(x) for x in df_train['kmer_sequence']])
        X_num_tr = df_train.drop(columns=['xor_sequence', 'kmer_sequence',
                                          'encoded_on_pam', 'encoded_off_pam', 'label']).values.astype('float32')
        y_tr = df_train['label'].values.astype('float32')
        
        X_xor_tr = np.expand_dims(X_xor_tr, axis=-1)
        X_kmer_tr = np.expand_dims(X_kmer_tr, axis=-1)
        
        # Normalize
        norm = layers.Normalization()
        norm.adapt(X_num_tr)
        
        # Train model
        model = EHNNHybrid(norm)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            [X_xor_tr, X_kmer_tr, X_num_tr], y_tr,
            validation_split=0.2, epochs=20, batch_size=128,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
            verbose=0
        )
        
        # Test on all other datasets
        for test_file in dataset_files:
            test_name = test_file.stem
            
            with open(test_file, 'rb') as f:
                df_test = pickle.load(f)
            
            X_xor_te = np.array([np.array(x) for x in df_test['xor_sequence']])
            X_kmer_te = np.array([np.array(x) for x in df_test['kmer_sequence']])
            X_num_te = df_test.drop(columns=['xor_sequence', 'kmer_sequence',
                                            'encoded_on_pam', 'encoded_off_pam', 'label']).values.astype('float32')
            y_te = df_test['label'].values.astype('float32')
            
            X_xor_te = np.expand_dims(X_xor_te, axis=-1)
            X_kmer_te = np.expand_dims(X_kmer_te, axis=-1)
            
            # Predict
            y_pred = model.predict([X_xor_te, X_kmer_te, X_num_te], verbose=0).flatten()
            metrics = compute_metrics(y_te, y_pred)
            metrics['train_dataset'] = train_name
            metrics['test_dataset'] = test_name
            cross_results.append(metrics)
    
    cross_df = pd.DataFrame(cross_results)
    
    # Compute summaries with std
    train_summary_mean = cross_df.groupby('train_dataset')[['roc_auc', 'f1', 'mcc']].mean()
    train_summary_std = cross_df.groupby('train_dataset')[['roc_auc', 'f1', 'mcc']].std()
    
    test_summary_mean = cross_df.groupby('test_dataset')[['roc_auc', 'f1', 'mcc']].mean()
    test_summary_std = cross_df.groupby('test_dataset')[['roc_auc', 'f1', 'mcc']].std()
    
    logging.info("Cross-dataset evaluation completed")
    
    return cross_df, (train_summary_mean, train_summary_std), (test_summary_mean, test_summary_std)

# ======================================================================
# 6. Plotting (updated with std error bars)
# ======================================================================

plt.style.use("ggplot")
sns.set_palette("Set2")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_roc_only(results, save_dir, fname="roc.png"):
    _ensure_dir(save_dir)
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc(fpr, tpr):.3f})")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()

def plot_pr_only(results, save_dir, fname="pr.png"):
    _ensure_dir(save_dir)
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(rec, pre, label=f"{name} (AUPR={auc(rec, pre):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()

def plot_sens_spec_only(results, save_dir, fname="sens_spec.png"):
    _ensure_dir(save_dir)
    plt.figure(figsize=(6, 5))
    thr = np.linspace(0, 1, 100)
    for name, res in results.items():
        y_true, y_pred = res['y_true'], res['y_pred']
        sens = [((y_true == 1) & (y_pred >= t)).sum() / (y_true == 1).sum() for t in thr]
        spec = [((y_true == 0) & (y_pred < t)).sum() / (y_true == 0).sum() for t in thr]
        plt.plot(1 - np.array(spec), sens, label=name)
    plt.xlabel("1 – Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs 1 – Specificity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()

def plot_cross_validation_results(cv_df, save_dir):
    """Plot k-fold CV results with error bars"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision', 'mcc']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        axes[idx].bar(cv_df['fold'], cv_df[metric])
        mean_val = cv_df[metric].mean()
        std_val = cv_df[metric].std()
        axes[idx].axhline(mean_val, color='r', linestyle='--', 
                         label=f'Mean={mean_val:.3f}±{std_val:.3f}')
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} across folds')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cross_validation_results.png", dpi=300)
    plt.close()

def plot_ablation_results(ablation_mean_df, ablation_std_df, save_dir):
    """Plot ablation study results with error bars"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision', 'mcc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ablation_mean_df.index))
    width = 0.12
    
    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics)/2)
        ax.bar(x + offset, ablation_mean_df[metric], width, 
               yerr=ablation_std_df[metric], capsize=3,
               label=metric.upper())
    
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Component Contribution (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_mean_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_study.png", dpi=300)
    plt.close()
    
    # Performance drop with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    full_model_mean = ablation_mean_df.loc['Full Model']
    full_model_std = ablation_std_df.loc['Full Model']
    drop_mean_df = ablation_mean_df.drop('Full Model')
    drop_std_df = ablation_std_df.drop('Full Model')
    
    for metric in metrics:
        drops = (full_model_mean[metric] - drop_mean_df[metric]) * 100
        # Propagate error: std = sqrt(std1^2 + std2^2)
        drops_std = np.sqrt(full_model_std[metric]**2 + drop_std_df[metric]**2) * 100
        ax.errorbar(range(len(drop_mean_df.index)), drops, yerr=drops_std, 
                   marker='o', label=metric.upper(), capsize=4)
    
    ax.set_xlabel('Component Removed')
    ax.set_ylabel('Performance Drop (%)')
    ax.set_title('Performance Impact of Removing Each Component (Mean ± Std)')
    ax.set_xticks(range(len(drop_mean_df.index)))
    ax.set_xticklabels(drop_mean_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_performance_drop.png", dpi=300)
    plt.close()

def plot_cross_dataset_heatmap(cross_df, save_dir):
    """Plot cross-dataset generalization heatmap"""
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in ['roc_auc', 'f1', 'mcc']:
        pivot = cross_df.pivot(index='train_dataset', 
                               columns='test_dataset', 
                               values=metric)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': metric.upper()})
        ax.set_title(f'Cross-Dataset {metric.upper()}: Train → Test')
        ax.set_xlabel('Test Dataset')
        ax.set_ylabel('Train Dataset')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cross_dataset_{metric}.png", dpi=300)
        plt.close()

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
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(); plt.plot(fpr, tpr)
        plt.title(f"{name} ROC (AUROC={auc(fpr,tpr):.3f})")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(f"{save_dir}/{name}_roc.png", dpi=300); plt.close()
        
        pre, rec, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(); plt.plot(rec, pre)
        plt.title(f"{name} PR (AUPR={auc(rec,pre):.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(f"{save_dir}/{name}_pr.png", dpi=300); plt.close()
        
        thr = np.linspace(0,1,100)
        mcc = [matthews_corrcoef(y_true, y_pred>=t) for t in thr]
        plt.figure(); plt.plot(thr, mcc)
        plt.title(f"{name} MCC (max={max(mcc):.3f})")
        plt.xlabel("Threshold"); plt.ylabel("MCC")
        plt.savefig(f"{save_dir}/{name}_mcc.png", dpi=300); plt.close()

# ======================================================================
# 7. Main Training with ALL Evaluation Methods + STD
# ======================================================================

def train_model_single(path):
    """Train model on single dataset (for standard evaluation)"""
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
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )

    y_pred = model.predict([x_xor_te, x_kmer_te, x_num_te]).flatten()
    metrics = compute_metrics(y_te, y_pred)
    metrics['y_true'] = y_te
    metrics['y_pred'] = y_pred
    
    return metrics

def train_main():
    """Main training orchestrator with ALL evaluation methods + STD"""
    in_dir = Path("encoded_dataset")
    out_dir = Path("Final_Results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists and has files
    if not in_dir.exists():
        logging.error(f"Input directory '{in_dir}' does not exist!")
        logging.error("Please run feature_main(), normalize_main(), and hybrid_encoding_main() first.")
        return
    
    pkl_files = list(in_dir.glob("*.pkl"))
    if not pkl_files:
        logging.error(f"No .pkl files found in '{in_dir}'!")
        logging.error("Please run feature_main(), normalize_main(), and hybrid_encoding_main() first.")
        return
    
    logging.info(f"Found {len(pkl_files)} dataset files to process")
    
    # ===== STANDARD HOLDOUT EVALUATION =====
    logging.info("=" * 70)
    logging.info("EVALUATION METHOD 1: Standard Holdout (80/20 split)")
    logging.info("=" * 70)
    
    results, metrics = {}, []
    for f in in_dir.glob("*.pkl"):
        try:
            m = train_model_single(f)
            name = f.stem
            results[name] = {'y_true': m['y_true'], 'y_pred': m['y_pred']}
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

    holdout_df = pd.DataFrame(metrics)
    
    if len(holdout_df) > 0:
        # Calculate mean and std across datasets
        numeric_cols = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision', 'accuracy', 'mcc', 'brier', 'tpr@1%fpr']
        holdout_summary = pd.DataFrame({
            'mean': holdout_df[numeric_cols].mean(),
            'std': holdout_df[numeric_cols].std()
        })
        
        holdout_df.to_csv(out_dir / "holdout_evaluation_results.csv", index=False)
        holdout_summary.to_csv(out_dir / "holdout_summary_mean_std.csv")
        
        logging.info("\nHoldout Evaluation Summary (Mean ± Std):")
        for metric in numeric_cols:
            mean_val = holdout_summary.loc[metric, 'mean']
            std_val = holdout_summary.loc[metric, 'std']
            logging.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        logging.warning("No holdout evaluation results to summarize")
    
    # Only plot if we have results
    if results:
        plot_roc_only(results, out_dir)
        plot_pr_only(results, out_dir)
        plot_sens_spec_only(results, out_dir)
        plot_mcc_curve(results, out_dir)
        plot_calibration_belt(results, out_dir)
        plot_individual_plots(results, out_dir / "individual")
    
    # ===== CROSS-VALIDATION EVALUATION =====
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION METHOD 2: K-Fold Cross-Validation")
    logging.info("=" * 70)
    
    cv_results_all = []
    cv_summary_all = []
    
    for f in in_dir.glob("*.pkl"):
        try:
            logging.info(f"\nPerforming 5-fold CV on {f.stem}")
            with open(f, 'rb') as file:
                df = pickle.load(file)
            
            X_xor = np.expand_dims(np.array([np.array(x) for x in df['xor_sequence']]), axis=-1)
            X_kmer = np.expand_dims(np.array([np.array(x) for x in df['kmer_sequence']]), axis=-1)
            X_num = df.drop(columns=['xor_sequence', 'kmer_sequence',
                                    'encoded_on_pam', 'encoded_off_pam', 'label']).values.astype('float32')
            y = df['label'].values.astype('float32')
            
            cv_df, mean_metrics, std_metrics = cross_validation_evaluation(X_xor, X_kmer, X_num, y, k_folds=5)
            cv_df['dataset'] = f.stem
            cv_results_all.append(cv_df)
            
            # Store summary with std
            summary = {'dataset': f.stem}
            for key in mean_metrics.keys():
                summary[f'{key}_mean'] = mean_metrics[key]
                summary[f'{key}_std'] = std_metrics[key]
            cv_summary_all.append(summary)
            
            # Save individual dataset CV results
            cv_df.to_csv(out_dir / f"cv_{f.stem}.csv", index=False)
            
            logging.info(f"  {f.stem} CV Results (Mean ± Std):")
            for key in ['roc_auc', 'pr_auc', 'f1', 'mcc']:
                logging.info(f"    {key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
            
        except Exception as e:
            logging.error(f"CV failed on {f}: {e}")
    
    if cv_results_all:
        cv_combined = pd.concat(cv_results_all, ignore_index=True)
        cv_combined.to_csv(out_dir / "cross_validation_all_datasets.csv", index=False)
        
        # Save summary with mean and std
        cv_summary_df = pd.DataFrame(cv_summary_all)
        cv_summary_df.to_csv(out_dir / "cross_validation_summary_mean_std.csv", index=False)
        
        # Plot CV results for each dataset
        for dataset_name in cv_combined['dataset'].unique():
            dataset_cv = cv_combined[cv_combined['dataset'] == dataset_name]
            plot_cross_validation_results(dataset_cv, out_dir / f"cv_plots_{dataset_name}")
    
    # ===== ABLATION STUDY =====
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION METHOD 3: Ablation Study (with multiple runs)")
    logging.info("=" * 70)
    
    # Perform ablation on the first/largest dataset
    ablation_file = list(in_dir.glob("*.pkl"))[0]
    logging.info(f"Performing ablation study on: {ablation_file.stem}")
    
    try:
        with open(ablation_file, 'rb') as f:
            df = pickle.load(f)
        
        X_xor = np.expand_dims(np.array([np.array(x) for x in df['xor_sequence']]), axis=-1)
        X_kmer = np.expand_dims(np.array([np.array(x) for x in df['kmer_sequence']]), axis=-1)
        X_num = df.drop(columns=['xor_sequence', 'kmer_sequence',
                                'encoded_on_pam', 'encoded_off_pam', 'label']).values.astype('float32')
        y = df['label'].values.astype('float32')
        
        ablation_mean_df, ablation_std_df = ablation_study(X_xor, X_kmer, X_num, y, n_runs=3)
        
        # Combine mean and std into single dataframe for saving
        ablation_combined = ablation_mean_df.copy()
        for col in ablation_mean_df.columns:
            ablation_combined[f'{col}_std'] = ablation_std_df[col]
        
        ablation_combined.to_csv(out_dir / "ablation_study_results_mean_std.csv")
        ablation_mean_df.to_csv(out_dir / "ablation_study_mean.csv")
        ablation_std_df.to_csv(out_dir / "ablation_study_std.csv")
        
        plot_ablation_results(ablation_mean_df, ablation_std_df, out_dir)
        
        logging.info("\nAblation Study Results (Mean ± Std):")
        for config in ablation_mean_df.index:
            logging.info(f"  {config}:")
            for metric in ['roc_auc', 'f1', 'mcc']:
                mean_val = ablation_mean_df.loc[config, metric]
                std_val = ablation_std_df.loc[config, metric]
                logging.info(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
    except Exception as e:
        logging.error(f"Ablation study failed: {e}")
    
    # ===== CROSS-DATASET EVALUATION =====
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION METHOD 4: Cross-Dataset Generalization Testing")
    logging.info("=" * 70)
    
    try:
        cross_df, train_summaries, test_summaries = cross_dataset_evaluation(in_dir)
        
        if cross_df is not None:
            cross_df.to_csv(out_dir / "cross_dataset_evaluation.csv", index=False)
            plot_cross_dataset_heatmap(cross_df, out_dir)
            
            # Unpack summaries
            train_summary_mean, train_summary_std = train_summaries
            test_summary_mean, test_summary_std = test_summaries
            
            # Combine mean and std for saving
            train_combined = train_summary_mean.copy()
            for col in train_summary_mean.columns:
                train_combined[f'{col}_std'] = train_summary_std[col]
            train_combined.to_csv(out_dir / "cross_dataset_train_summary_mean_std.csv")
            
            test_combined = test_summary_mean.copy()
            for col in test_summary_mean.columns:
                test_combined[f'{col}_std'] = test_summary_std[col]
            test_combined.to_csv(out_dir / "cross_dataset_test_summary_mean_std.csv")
            
            logging.info("\nCross-Dataset Generalization - Training on each dataset (Mean ± Std):")
            for dataset in train_summary_mean.index:
                logging.info(f"  {dataset}:")
                for metric in ['roc_auc', 'f1', 'mcc']:
                    mean_val = train_summary_mean.loc[dataset, metric]
                    std_val = train_summary_std.loc[dataset, metric]
                    logging.info(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
            
    except Exception as e:
        logging.error(f"Cross-dataset evaluation failed: {e}")
    
    # ===== FINAL SUMMARY =====
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION COMPLETE - All Results Include Standard Deviations")
    logging.info("=" * 70)
    logging.info(f"Results saved to: {out_dir}")
    logging.info("\nGenerated outputs (all with Mean ± Std):")
    logging.info("  1. holdout_evaluation_results.csv - Standard 80/20 split metrics")
    logging.info("     holdout_summary_mean_std.csv - Summary across datasets")
    logging.info("  2. cross_validation_all_datasets.csv - K-fold CV results")
    logging.info("     cross_validation_summary_mean_std.csv - CV summary with std")
    logging.info("  3. ablation_study_results_mean_std.csv - Component contribution")
    logging.info("     ablation_study_mean.csv & ablation_study_std.csv - Detailed results")
    logging.info("  4. cross_dataset_evaluation.csv - Generalization across datasets")
    logging.info("     cross_dataset_*_summary_mean_std.csv - Summaries with std")
    logging.info("  5. Multiple visualization plots with error bars in Final_Results/")

# ======================================================================
# 8. Entry Point
# ======================================================================

if __name__ == "__main__":
    # PIPELINE STEPS:
    # Step 1: Extract features from raw CSV files
    # feature_main()
    
    # Step 2: Normalize the features
    # normalize_main()
    
    # Step 3: Create hybrid encodings
    # hybrid_encoding_main()
    
    # Step 4: Train models and run all evaluations (WITH STANDARD DEVIATIONS)
    # Uncomment the above three functions first if you haven't processed the data yet
    train_main()