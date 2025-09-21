from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import EHNNConfig
from .features import run as run_features
from .normalize import run as run_normalize
from .encoding import run as run_encode
from .train import train_on_dir
from .utils import get_logger, set_seed

app = typer.Typer(help="EHNN – End-to-End CRISPR Off-Target Predictor")

logger = get_logger()

@app.callback()
def _common(ctx: typer.Context, config: Optional[Path] = typer.Option(None, "--config", "-c",
                help="Optional YAML config file."),
            seed: int = typer.Option(42, help="Random seed")):
    ctx.obj = EHNNConfig.from_yaml(config)
    set_seed(seed)

@app.command()
def features(ctx: typer.Context,
             in_dir: Path = typer.Option(None, help="Input CSV dir (on_seq, off_seq, label)"),
             out_dir: Path = typer.Option(None, help="Output dir for feature CSVs")):
    cfg = ctx.obj
    in_dir = in_dir or cfg.paths.original
    out_dir = out_dir or cfg.paths.features
    run_features(in_dir, out_dir)

@app.command()
def normalize(ctx: typer.Context,
              in_dir: Path = typer.Option(None, help="Input feature CSV dir"),
              out_dir: Path = typer.Option(None, help="Output normalized CSV dir")):
    cfg = ctx.obj
    in_dir = in_dir or cfg.paths.features
    out_dir = out_dir or cfg.paths.normalized
    run_normalize(in_dir, out_dir)

@app.command()
def encode(ctx: typer.Context,
           in_dir: Path = typer.Option(None, help="Input normalized CSV dir"),
           out_dir: Path = typer.Option(None, help="Output encoded PKL dir")):
    cfg = ctx.obj
    in_dir = in_dir or cfg.paths.normalized
    out_dir = out_dir or cfg.paths.encoded
    run_encode(in_dir, out_dir)

@app.command()
def train(ctx: typer.Context,
          in_dir: Path = typer.Option(None, help="Input encoded PKL dir"),
          out_dir: Path = typer.Option(None, help="Output results dir"),
          epochs: int = typer.Option(None, help="Epochs"),
          batch_size: int = typer.Option(None, help="Batch size"),
          lr: float = typer.Option(None, help="Learning rate"),
          val_split: float = typer.Option(None, help="Validation split"),
          test_size: float = typer.Option(None, help="Test split"),
          threshold: float = typer.Option(None, help="Classification threshold")):
    cfg = ctx.obj
    in_dir = in_dir or cfg.paths.encoded
    out_dir = out_dir or cfg.paths.results
    train_on_dir(
        in_dir, out_dir,
        epochs=epochs or cfg.train.epochs,
        batch_size=batch_size or cfg.train.batch_size,
        lr=lr or cfg.train.lr,
        val_split=val_split or cfg.train.val_split,
        test_size=test_size or cfg.train.test_size,
        random_state=cfg.train.random_state,
        threshold=threshold or cfg.train.threshold,
    )

@app.command()
def all(ctx: typer.Context):
    cfg = ctx.obj
    cfg.paths.ensure()
    run_features(cfg.paths.original, cfg.paths.features)
    run_normalize(cfg.paths.features, cfg.paths.normalized)
    run_encode(cfg.paths.normalized, cfg.paths.encoded)
    train_on_dir(cfg.paths.encoded, cfg.paths.results,
                 epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
                 lr=cfg.train.lr, val_split=cfg.train.val_split,
                 test_size=cfg.train.test_size, random_state=cfg.train.random_state,
                 threshold=cfg.train.threshold)

def main():
    app()
