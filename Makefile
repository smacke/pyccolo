# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy blackcheck lint typecheck check_no_typing check test tests coverage xmlcov check_ci deps devdeps

clean:
	rm -rf __pycache__ build/ dist/ *.egg-info/ .coverage htmlcov

build: clean
	python -m build

bump:
	./scripts/bump-version.py

deploy: build
	./scripts/deploy.sh

black:
	./scripts/blacken.sh

blackcheck:
	./scripts/blacken.sh --check

lint:
	flake8

typecheck:
	mypy pyccolo

check_no_typing:
	rm -f .coverage
	rm -rf htmlcov
	pytest --cov-config=pyproject.toml --cov=pyccolo

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

