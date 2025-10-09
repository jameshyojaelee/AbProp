PYTHON ?= python
PIP ?= pip

.PHONY: env install dev lint format typecheck test clean train dist-train

env:
	@echo "Configure your environment here (e.g., conda/mamba)."

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e .[dev]

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

typecheck:
	mypy src

test:
	PYTHONPATH=src pytest -q

train:
	abprop-train --distributed none

dist-train:
	abprop-launch --nodes 1 --gpus-per-node 4 --config configs/train.yaml

clean:
	rm -rf build dist .pytest_cache *.egg-info
