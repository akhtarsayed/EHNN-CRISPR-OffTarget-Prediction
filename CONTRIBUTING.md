# Contributing

Thank you for considering a contribution!

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## Tests & style

```bash
pytest -q
black . && isort . && flake8
```
