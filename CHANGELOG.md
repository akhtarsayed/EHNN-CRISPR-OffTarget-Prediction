# Changelog

## Unreleased

- Added standard repository scaffolding: `pyproject.toml`, `run.py`
  command-line entry point, `configs/default.yaml`,
  `src/ehnn/encoding.py` with unit tests, GitHub Actions CI,
  `CONTRIBUTING.md` and `CITATION.cff`.
- Added the missing MIT `LICENSE` file.
- Fixed `README.md` to name the actual pipeline script,
  `EHNN_Complete_Pipeline.py`, and to list only committed files;
  generated folders are documented separately.
- Added generated folders (`feature_dataset/`, `normalize_dataset/`,
  `encoded_dataset/`) to `.gitignore`.

## 0.1.0

- Initial public EHNN pipeline for CRISPR-Cas9 off-target prediction,
  with feature extraction, hybrid XOR/k-mer/PAM encoding, a
  CNN-LSTM-MLP model, and holdout, cross-validation, ablation and
  cross-dataset evaluation.
