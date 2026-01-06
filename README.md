# EHNN: Enhanced Hybrid Neural Network for CRISPR–Cas9 Off-Target Prediction

## 📌 Description

EHNN is a hybrid deep learning pipeline that combines XOR-based sequence encoding, k-mer context, PAM classification, and engineered sequence features to predict CRISPR–Cas9 off-target cleavage events.

It is the **first architecture** that jointly integrates:

* **XOR mismatch maps**
* **k-mer contexts**
* **PAM categories**
* **51 engineered biochemical sequence features**

The model fuses **CNN–LSTM–MLP** modules and achieves robust accuracy and calibration across multiple public CRISPR datasets.

---

## 📂 Project Structure

```
original_dataset/        # Raw input CSVs (on_seq, off_seq, label)
feature_dataset/         # Extracted biochemical and sequence features
normalize_dataset/       # Min-Max scaled features
encoded_dataset/         # XOR/k-mer/PAM encoded datasets
# EHNN Pipeline — current workspace

This repository contains the EHNN pipeline and supporting datasets/results used for CRISPR–Cas9 off-target prediction experiments.

**Quick summary:** the workspace includes raw datasets, encoded inputs, and final evaluation outputs (CSV + figures).

---

**Current repository layout**

- `ehnn_pipline.py` — main pipeline script (run preprocessing, encoding, training/evaluation).
- `upload_pipeline_to_github.py` — helper to prepare/upload artifacts.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT license.
- `README.md` — this file.
- `original_dataset/` — raw CSV files used as inputs (see list below).
- `encoded_dataset/` — encoded inputs produced by the pipeline.
- `Final_Results/` — consolidated outputs and figures.
   - `full_results_EHNN.csv` — primary summary of final results.
   - `figures/` — generated plots.
   - `tables/` — CSV/TSV result tables for manuscript.

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

**Quick start**

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the pipeline (this runs preprocessing, encoding and evaluation):

```powershell
python ehnn_pipline.py
```

Outputs will be written into `Final_Results/` and encoded inputs into `encoded_dataset/`.

---

**Notes & pointers**

- If you add new raw CSVs to `original_dataset/`, the pipeline will detect and process them when run.
- Use `Final_Results/full_results_EHNN.csv` for aggregated metrics used in manuscript tables and figures.
- For debugging or custom runs, inspect `ehnn_pipline.py` to find options/flags (if present).

---

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

---

## License

MIT — see `LICENSE`.

---

If you'd like, I can also:

- add a brief usage snippet at the top of `ehnn_pipline.py` to document CLI flags, or
- generate a short `run_example.bat` / `run_example.sh` for easy runs on Windows/Linux.
