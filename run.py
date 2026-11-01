#!/usr/bin/env python3
"""Command-line entry point for the EHNN pipeline.

Run from the repository root, because the pipeline stages use relative
paths (``original_dataset/``, ``feature_dataset/``, ``normalize_dataset/``,
``encoded_dataset/`` and ``Final_Results/``).

Examples:
    python run.py --stage all
    python run.py --stage features
    python run.py --stage train

The pipeline module is imported lazily inside each stage so that
``python run.py --help`` works without TensorFlow installed.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the EHNN pipeline.")
    parser.add_argument(
        "--stage",
        choices=["features", "normalize", "encode", "train", "all"],
        required=True,
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    # Imported here, not at module level, so --help does not need TensorFlow.
    from EHNN_Complete_Pipeline import (
        feature_main,
        hybrid_encoding_main,
        normalize_main,
        train_main,
    )

    stages = {
        "features": [feature_main],
        "normalize": [normalize_main],
        "encode": [hybrid_encoding_main],
        "train": [train_main],
        "all": [feature_main, normalize_main, hybrid_encoding_main, train_main],
    }
    for stage_fn in stages[args.stage]:
        stage_fn()


if __name__ == "__main__":
    main()
