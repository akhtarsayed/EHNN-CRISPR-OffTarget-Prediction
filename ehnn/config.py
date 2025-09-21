from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

@dataclass
class Paths:
    original: Path = Path("original_dataset")
    features: Path = Path("feature_dataset")
    normalized: Path = Path("normalize_dataset")
    encoded: Path = Path("encoded_dataset")
    results: Path = Path("Final_Results")

    def ensure(self) -> None:
        self.features.mkdir(parents=True, exist_ok=True)
        self.normalized.mkdir(parents=True, exist_ok=True)
        self.encoded.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    val_split: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    threshold: float = 0.3

@dataclass
class EHNNConfig:
    paths: Paths = field(default_factory=Paths)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_yaml(path: Optional[str | Path]) -> "EHNNConfig":
        if path is None:
            return EHNNConfig()
        data = yaml.safe_load(Path(path).read_text())
        # minimal merging
        p = data.get("paths", {})
        t = data.get("train", {})
        cfg = EHNNConfig(
            paths=Paths(
                original=Path(p.get("original", "original_dataset")),
                features=Path(p.get("features", "feature_dataset")),
                normalized=Path(p.get("normalized", "normalize_dataset")),
                encoded=Path(p.get("encoded", "encoded_dataset")),
                results=Path(p.get("results", "Final_Results")),
            ),
            train=TrainConfig(
                epochs=int(t.get("epochs", 20)),
                batch_size=int(t.get("batch_size", 128)),
                lr=float(t.get("lr", 1e-3)),
                val_split=float(t.get("val_split", 0.2)),
                test_size=float(t.get("test_size", 0.2)),
                random_state=int(t.get("random_state", 42)),
                threshold=float(t.get("threshold", 0.3)),
            ),
        )
        return cfg