from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import get_logger

logger = get_logger()

def run(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler()
    for f in in_dir.glob("*.csv"):
        df = pd.read_csv(f)
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if "label" in num_cols:
            num_cols.remove("label")
        if num_cols:
            df[num_cols] = scaler.fit_transform(df[num_cols])
        df.to_csv(out_dir / f"{f.stem}_normalized.csv", index=False)
        logger.info(f"[normalize] Wrote {out_dir / (f.stem + '_normalized.csv')}")
    logger.info("[normalize] Completed.")
