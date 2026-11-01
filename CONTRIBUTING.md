# Contributing

Thank you for considering a contribution to EHNN.

## Workflow

1. Fork the repository and create a branch for your change.
2. Run the unit tests before opening a pull request:

   ```bash
   pip install pytest
   pytest
   ```

3. Keep changes focused, and do not commit generated data or results
   (`feature_dataset/`, `normalize_dataset/`, `encoded_dataset/`).
   Those folders are created when the pipeline runs and are ignored
   by `.gitignore`.
4. Update `README.md` or `CHANGELOG.md` when a change affects how
   the pipeline is run or what it produces.

## Reporting issues

Please include the dataset name, the stage that failed
(`features`, `normalize`, `encode` or `train`), and the full error
message.
