# EHNN: Enhanced Hybrid Neural Network for CRISPR–Cas9 Off-Target Prediction

## Description

EHNN is a hybrid deep learning pipeline that combines XOR-based sequence encoding, k-mer context, PAM classification, and engineered sequence features to predict CRISPR–Cas9 off-target cleavage events.

It jointly integrates:

* **XOR mismatch maps**
* **k-mer contexts**
* **PAM categories**
* **51 engineered biochemical sequence features**

The model fuses **CNN–LSTM–MLP** modules and achieves robust accuracy and calibration across multiple public CRISPR datasets.

---

## Project structure

```
EHNN_Complete_Pipeline.py  # Main pipeline: preprocessing, encoding, training/evaluation
README.md
requirements.txt
LICENSE
.gitignore
original_dataset/           # Raw input CSVs (on_seq, off_seq, label)
Final_Results/              # Consolidated outputs and figures
  full_results_EHNN.csv     # Primary summary of final results
  figures/                  # Generated plots
  tables/                   # CSV/TSV result tables for manuscript
```

The following folders are generated when the pipeline runs and are not committed to the repository:

```
feature_dataset/    # Extracted biochemical and sequence features
normalize_dataset/  # Min-Max scaled features
encoded_dataset/    # XOR/k-mer/PAM encoded datasets
```

**Files in `original_dataset/`**

- `CIRCLE_seq.csv`
- `Doench.csv`
- `GUIDE-Seq_Kleinstiver.csv`
- `GUIDE-Seq_Listgarten.csv`
- `Hek293t_K562.csv`
- `Hek293t.csv`
- `K562.csv`
- `SITE-Seq.csv`

---

## Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the pipeline (this runs preprocessing, encoding and evaluation):

```powershell
python EHNN_Complete_Pipeline.py
```

Outputs will be written into `Final_Results/` and encoded inputs into `encoded_dataset/`.

---

## Notes

- If you add new raw CSVs to `original_dataset/`, the pipeline will detect and process them when run.
- Use `Final_Results/full_results_EHNN.csv` for aggregated metrics used in manuscript tables and figures.

---

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

---

## License

MIT — see `LICENSE`.
