# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy typecheck check_no_typing check test tests coverage xmlcov check_ci deps devdeps

clean:
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov

build: clean
	python setup.py sdist bdist_wheel --universal

bump:
	./scripts/bump-version.py

deploy: build
	./scripts/deploy.sh

typecheck:
	mypy pyccolo

check_no_typing:
	rm -f .coverage
	rm -rf htmlcov
	pytest --cov-config=.coveragerc --cov=pyccolo

check: typecheck check_no_typing

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
	pip install -e .
	pip install -r requirements-dev.txt

