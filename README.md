# EHNN: Enhanced Hybrid Neural Network for CRISPR–Cas9 Off-Target Prediction

## 📌 Description
EHNN is a hybrid deep learning pipeline that combines XOR-based sequence encoding, k-mer context, PAM classification, and 51 extrected engineered  features to predict CRISPR–Cas9 off-target cleavage events.
• EHNN is introduced as the first architecture that jointly integrates XOR-mismatch maps, k-mer
  85 contexts, and PAM categories within a unified CNN–LSTM–MLP framework.
• A 51-dimensional feature vector is extracted from each guide–target pair to comprehensively
  quantify sequence-level determinants of off-target activity.
• State-of-the-art performance in both accuracy and calibration is achieved across eight diverse
  CRISPR datasets.
  
EHNN is a hybrid deep learning pipeline for **predicting CRISPR–Cas9 off-target cleavage events**.  
It is the **first architecture** that jointly integrates:
- **XOR mismatch maps**
- **k-mer contexts**
- **PAM categories**
- **51 engineered biochemical sequence features**



The model combines **CNN–LSTM–MLP** modules to achieve state-of-the-art prediction accuracy and calibration across **eight benchmark CRISPR datasets**.  

---

## 📂 Project Structure
```
original_dataset/        # Raw input CSVs (on_seq, off_seq, label)
feature_dataset/         # Extracted biochemical and sequence features
normalize_dataset/       # Min-Max scaled feature datasets
encoded_dataset/         # XOR/k-mer/PAM encoded datasets
Final_Results/           # Model metrics, calibration plots, AUC curves
EHNN_Complete_Pipeline.py  # Full pipeline script
```

---

## 📊 Dataset Information
The pipeline uses well-established **public CRISPR off-target datasets**, including:  
- **CIRCLE-seq** (Tsai et al. 2017) – https://doi.org/10.1038/nmeth.4278  
- **GUIDE-seq** (Tsai et al. 2015) – https://doi.org/10.1038/nbt.3117  
- **Doench 2016** – https://doi.org/10.1038/nbt.3437  
- **SITE-seq** – https://doi.org/10.1038/nmeth.4281  
- **HEK293T / K562 cell lines** (Kleinstiver et al.)  
- **CHANGE-seq** and other benchmark sets  

All datasets contain **on-target and off-target gRNA–DNA pairs with binary cleavage labels (0/1)**.  

---

## 💻 Code Information
- **Language**: Python 3.8+  
- **Frameworks**: TensorFlow/Keras, scikit-learn, Biopython, Matplotlib/Seaborn  

**Pipeline Components:**
1. **Feature Extraction** → biochemical + sequence descriptors  
2. **Normalization** → Min-Max scaling  
3. **Hybrid Encoding** → XOR maps, k-mers, PAM classification  
4. **Model Training** → CNN + BiLSTM + MLP fusion  
5. **Evaluation & Plotting** → ROC, PR, MCC, calibration  

---

## ⚙️ Requirements
Install dependencies via:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
tensorflow>=2.6
scikit-learn>=0.24
numpy>=1.19
pandas>=1.1
matplotlib>=3.3
seaborn>=0.11
biopython>=1.78
gitpython>=3.1
```

---

## 🚀 Usage Instructions
Clone the repo and run the pipeline:

```bash
git clone https://github.com/akhtarsayed/EHNN-CRISPR-OffTarget-Prediction.git
cd EHNN-CRISPR-OffTarget-Prediction
pip install -r requirements.txt
python EHNN_Complete_Pipeline.py
```

**Outputs:**
- `Final_Results/full_results_EHNN.csv` → summary metrics  
- `Final_Results/*.png` → combined ROC/PR/SS plots  
- `Final_Results/individual/*.png` → per-dataset ROC/PR/MCC curves  

---

## 🧪 Methodology (Summary)
1. **Input:** gRNA–DNA sequence pairs from CRISPR datasets  
2. **Feature extraction:** 51-dimensional feature vector (GC content, mismatches, dinucleotide counts, PAM context, etc.)  
3. **Encoding:** XOR mismatch mapping, k-mer encoding (k=3), PAM classification  
4. **Model architecture:**  
   - CNN branches for XOR/k-mer sequences  
   - BiLSTM for sequential dependencies  
   - MLP for biochemical features  
   - Fused via concatenation → dense layers → sigmoid output  
5. **Evaluation Metrics:** ROC–AUC, PR–AUC, F1, Precision, Recall, MCC, Brier score, TPR@1%FPR  

---

## 📚 Citations
If you use EHNN in your research, please cite:

- **This repository/manuscript** (will be updated).  
- Tsai et al., *Nat Methods* 2017 (CIRCLE-seq)  
- Tsai et al., *Nat Biotechnol* 2015 (GUIDE-seq)  
- Doench et al., *Nat Biotechnol* 2016 (Doench dataset)  
- Kleinstiver et al., *Nat Biotechnol* 2016 (HEK293T/K562)  

---

## 📜 License
This repository is distributed under the **MIT License**. See `LICENSE` for details.

---

## 🤝 Contribution Guidelines
Contributions are welcome!  
- Fork the repo, create a feature branch, and submit a pull request.  
- Please ensure code is tested and documented.  
