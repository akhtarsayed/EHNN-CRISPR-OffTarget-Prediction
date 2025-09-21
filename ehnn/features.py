from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

from .utils import get_logger

logger = get_logger()

REGULATORY = {"TATA_box": "TATAAA", "CAAT_box": "CCAAT", "GC_box": "GGGCGG"}
RESTRICTION = {"EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT"}

def calculate_features(seq: str) -> Dict[str, float | int]:
    """Compute sequence-level features for a DNA string."""
    seq = (seq or "").upper().replace("U", "T")
    seq_obj = Seq(seq)
    gc = gc_fraction(seq_obj)
    length = len(seq)
    pam = seq[-3:] if length >= 3 else seq
    self_comp = sum(seq[i] == seq[-(i + 1)] for i in range(length // 2))
    at = (seq.count("A") + seq.count("T")) / max(length, 1)
    di = {pair: seq.count(pair) for pair in
          ["AA","AT","AG","AC","TT","TA","TG","TC","GG","GA","GT","GC","CC","CA","CT","CG"]}
    return {
        "GC_Content": float(gc),
        "AT_Content": float(at),
        "Length": int(length),
        "PAM_Sequence": pam,
        "Self_Complementarity": int(self_comp),
        **di,
    }

def _process_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        on = str(r["on_seq"])
        off = str(r["off_seq"])
        label = int(r["label"])
        on_f = calculate_features(on)
        off_f = calculate_features(off)
        pair = {
            "on_seq": on,
            "off_seq": off,
            "label": label,
            **{f"on_{k}": v for k, v in on_f.items()},
            **{f"off_{k}": v for k, v in off_f.items()},
            "Hamming_Distance": sum(a != b for a, b in zip(on, off)) if len(on) == len(off) else np.nan,
            "Mismatch_Count": sum(a != b for a, b in zip(on, off)) if len(on) == len(off) else np.nan,
        }
        rows.append(pair)
    return pd.DataFrame(rows)

def run(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in in_dir.glob("*.csv"):
        logger.info(f"[features] Processing {f}")
        df = _process_file(f)
        df.to_csv(out_dir / f"{f.stem}_feature.csv", index=False)
    logger.info("[features] Completed.")
