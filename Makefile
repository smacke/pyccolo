# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy black blackcheck imports lint typecheck check_no_typing check test tests coverage xmlcov check_ci deps devdeps

clean:
	rm -rf __pycache__ build/ dist/ *.egg-info/ .coverage htmlcov

build: clean
	python -m build

bump:
	./scripts/bump-version.py

deploy: build
	./scripts/deploy.sh

black:
	isort .
	./scripts/blacken.sh

blackcheck:
	isort . --check-only
	./scripts/blacken.sh --check

imports:
	pycln .
	isort .

lint:
	ruff check

typecheck:
	mypy pyccolo

check_no_typing:
	rm -f .coverage
	rm -rf htmlcov
	PYCCOLO_DEV_MODE=1 pytest --cov-config=pyproject.toml --cov=pyccolo

check: blackcheck lint typecheck check_no_typing

test: check
tests: check

coverage: check_no_typing
	coverage html

xmlcov: check_no_typing
	coverage xml

check_ci: typecheck xmlcov

deps:
	pip install -r requirements.txt

devdeps:
	pip install -e .[dev]

