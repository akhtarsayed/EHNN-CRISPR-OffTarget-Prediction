# EHNN: Enhanced Hybrid Neural Network for CRISPR–Cas9 Off-Target Prediction


## Overview
EHNN is a hybrid deep learning pipeline that combines XOR-based sequence encoding, k-mer context, PAM classification, and 51 extrected engineered  features to predict CRISPR–Cas9 off-target cleavage events.

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
  - `train_model.py` – Model training
  - `evaluate.py` – Model evaluation
  - `feature_extraction.py` – Sequence feature processing
  - `model_architecture.py` – EHNN architecture
  - `data_loader.py` – Loads and processes datasets

## Usage Instructions
```bash
git clone https://github.com/akhtarsayed/EHNN-CRISPR-OffTarget-Prediction.git
cd EHNN-CRISPR-OffTarget-Prediction
pip install -r requirements.txt
python train_model.py
