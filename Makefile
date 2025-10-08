PYTHON ?= python
PIP ?= pip

.PHONY: install dev lint format typecheck test clean

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
	pytest -q

clean:
	rm -rf build dist .pytest_cache *.egg-info

