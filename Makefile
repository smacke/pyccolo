# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy black blackcheck imports lint typecheck check_no_typing check test tests coverage xmlcov check_ci deps devdeps jupyterlite jupyterlite-serve jupyterlite-dev docs

# Port for the local JupyterLite demo server; override with `make ... LITE_PORT=8999`.
LITE_PORT ?= 8000

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

jupyterlite:
	bash jupyterlite/build.sh dist

jupyterlite-serve:
	@test -f dist/lab/index.html || $(MAKE) jupyterlite
	@echo "Serving JupyterLite demo at http://127.0.0.1:$(LITE_PORT)/lab/index.html?path=demo.ipynb (Ctrl-C to stop)"
	python -m http.server -d dist $(LITE_PORT)

jupyterlite-dev: jupyterlite jupyterlite-serve

docs:
	$(MAKE) -C docs html

