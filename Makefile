.PHONY: install dev test lint features normalize encode train all

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	black . && isort . && flake8 .

features:
	ehnn features

normalize:
	ehnn normalize

encode:
	ehnn encode

train:
	ehnn train

all:
	ehnn all
