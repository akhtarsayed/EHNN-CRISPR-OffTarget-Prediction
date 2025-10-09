# EHNN: Enhanced Hybrid Neural Network for CRISPR–Cas9 Off-Target Prediction


## Overview
EHNN is a hybrid deep learning pipeline that combines XOR-based sequence encoding, k-mer context, PAM classification, and 51 extrected engineered  features to predict CRISPR–Cas9 off-target cleavage events.
• EHNN is introduced as the first architecture that jointly integrates XOR-mismatch maps, k-mer
  85 contexts, and PAM categories within a unified CNN–LSTM–MLP framework.
• A 51-dimensional feature vector is extracted from each guide–target pair to comprehensively
  quantify sequence-level determinants of off-target activity.
• State-of-the-art performance in both accuracy and calibration is achieved across eight diverse
  CRISPR datasets

## Project Structure
```bash
original_dataset/        # Raw input CSVs (on_seq, off_seq, label)
feature_dataset/         # Extracted biochemical and sequence features
normalize_dataset/       # Min-Max scaled feature datasets
encoded_dataset/         # XOR/k-mer/PAM encoded datasets
Final_Results/           # Model metrics, calibration plots, AUC curves
EHNN_Complete_Pipeline.py  # Full pipeline script



## Dataset Information
- Public datasets used: CIRCLE-seq, GUIDE-seq, Doench 2016, HEK293T, K562, SITE-seq.
- See paper Section 3.1 for detailed descriptions.

## Code Information
- Language: Python
- Frameworks: TensorFlow/Keras, Scikit-learn
- Scripts:
  - `EHNN_Complete_Pipeline.py` – Complete pipeline 


## Usage Instructions
```bash
git clone https://github.com/akhtarsayed/EHNN-CRISPR-OffTarget-Prediction.git
cd EHNN-CRISPR-OffTarget-Prediction
pip install -r requirements.txt
python train_model.py
