# EHNN – End‑to‑End CRISPR Off‑Target Predictor

A clean, production‑ready Python package and CLI that reproduces and modularizes the
EHNN pipeline: **Feature Extraction → Normalization → Hybrid Encoding → Model Training**.

> This repository refactors the original single‑file script into a standard, testable,
> and documented project, with CLI commands, configuration, CI, and examples.

## Quickstart

```bash
# 1) Clone your GitHub repo (or unzip the archive from ChatGPT)
cd ehnn-crispr

# 2) Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install package
pip install -e .
# For training (TensorFlow):
pip install -e ".[ml]"
```

### CLI usage

```bash
# Show help
ehnn --help

# Run full pipeline on the default folders
ehnn all

# Or run stage by stage
ehnn features --in-dir original_dataset --out-dir feature_dataset
ehnn normalize --in-dir feature_dataset --out-dir normalize_dataset
ehnn encode --in-dir normalize_dataset --out-dir encoded_dataset
ehnn train  --in-dir encoded_dataset --out-dir Final_Results
```

### Expected input format

CSV files inside `original_dataset/` with the following columns:

- `on_seq` – on‑target sequence (DNA letters A/C/G/T)
- `off_seq` – off‑target sequence (DNA letters A/C/G/T)
- `label`   – 1 or 0 (positive/negative)

A minimal example is provided at `examples/data/original_dataset/sample.csv`.

## Project layout

```text
ehnn-crispr/
├─ ehnn/                     # Python package
│  ├─ __init__.py
│  ├─ cli.py                 # Typer-based CLI
│  ├─ config.py              # YAML/CLI configuration helpers
│  ├─ utils.py               # seeds, logging
│  ├─ features.py            # feature extraction
│  ├─ normalize.py           # normalization
│  ├─ encoding.py            # hybrid encodings (XOR, k-mer, PAM)
│  ├─ model.py               # EHNN Hybrid model
│  ├─ train.py               # training and evaluation
│  └─ plots.py               # ROC/PR/MCC/Calibration plots
├─ examples/data/original_dataset/sample.csv
├─ tests/                    # light unit tests (no TF required)
├─ .github/workflows/ci.yml  # CI for linting & tests
├─ pyproject.toml
├─ LICENSE
├─ CONTRIBUTING.md
├─ CITATION.cff
├─ .gitignore
└─ README.md
```

## Reproducibility & outputs

- All intermediate datasets are written to `feature_dataset/`, `normalize_dataset/`,
  and `encoded_dataset/` by default.
- Training metrics and figures are saved under `Final_Results/`.
- Random seeds are set for reproducibility.

## Notes

- TensorFlow is an optional dependency (`.[ml]`) so that CI and users who
  only need the feature/encoding stages don't pull the entire ML stack.
- The refactor preserves the core logic of your original code while improving
  modularity and maintainability.

## License

MIT License © 2025 Akhtar Sayed
