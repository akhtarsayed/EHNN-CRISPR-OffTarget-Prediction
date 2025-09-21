from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger()

NT_BITS: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
K = 3
KMER_VOCAB = {"".join(p): idx for idx, p in enumerate(product("ACGT", repeat=K))}
PAM_MAP = {p: i for i, p in enumerate(["TGG", "GAG", "GGG", "Other"])}

def encode_xor(off: str, on: str) -> List[int]:
    return [NT_BITS.get(o, 0) ^ NT_BITS.get(i, 0) for o, i in zip(off.upper(), on.upper())]

def encode_kmer(seq: str) -> List[int]:
    seq = (seq or "").upper()
    vec = [KMER_VOCAB.get(seq[i : i + K].upper(), len(KMER_VOCAB)) for i in range(len(seq) - K + 1)]
    return vec if vec else [len(KMER_VOCAB)]

def encode_pam(pam: str) -> int:
    return PAM_MAP.get((pam or "").upper(), PAM_MAP["Other"])

def run(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ["on_seq", "off_seq", "on_PAM_Sequence", "off_PAM_Sequence", "label"]
    for file in in_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if not set(required).issubset(df.columns):
            logger.warning(f"[encode] Skipping {file} (missing columns)")
            continue
        df.fillna(0, inplace=True)

        xor = [encode_xor(r.off_seq, r.on_seq) for _, r in df.iterrows()]
        kmer = [encode_kmer(r.off_seq) for _, r in df.iterrows()]
        pam_on = df["on_PAM_Sequence"].apply(encode_pam).values.astype("int32")
        pam_off = df["off_PAM_Sequence"].apply(encode_pam).values.astype("int32")

        max_xor = max(map(len, xor))
        max_kmer = max(map(len, kmer))
        xor = [v + [4] * (max_xor - len(v)) for v in xor]
        kmer = [v + [len(KMER_VOCAB)] * (max_kmer - len(v)) for v in kmer]

        num_cols = [c for c in df.columns if c not in required]
        X_num = df[num_cols].values.astype("float32")
        y = df["label"].values.astype("float32")

        final = pd.DataFrame(X_num, columns=num_cols)
        final["xor_sequence"] = xor
        final["kmer_sequence"] = kmer
        final["encoded_on_pam"] = pam_on
        final["encoded_off_pam"] = pam_off
        final["label"] = y

        base = file.stem
        out_path = out_dir / f"hybrid_{base}.pkl"
        final.to_pickle(out_path)
        logger.info(f"[encode] Wrote {out_path}")
    logger.info("[encode] Completed.")
