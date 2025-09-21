from __future__ import annotations

import logging
import os
import random

import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # optional
        tf.random.set_seed(seed)
    except Exception:
        pass

def get_logger(name: str = "ehnn") -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger
