# Check Python version (>=3.9 required)
PYTHON_VERSION_OK := $(shell python3 -c "import sys; print(int(sys.version_info >= (3, 9)))")
ifeq ($(PYTHON_VERSION_OK),0)
$(error "Python >= 3.9 is required")
endif

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

.PHONY: install
install:
	pip install uv
	uv pip install -e .

.PHONY: dev
dev: install
	uv pip install -e ".[dev]"

.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format: 
	uv run ruff format
	uv run ruff check --fix

.PHONY: check
check:
	uv run ruff format --check
	uv run ruff check
	uv run mypy .

.PHONY: lint
lint: 
	uv run ruff check

.PHONY: mypy
mypy: 
	uv run mypy .

.PHONY: tests
tests: 
	uv run pytest 

.PHONY: coverage
coverage:
	uv run coverage run -m pytest
	uv run coverage xml -o coverage.xml
	uv run coverage report -m --fail-under=95

.PHONY: snapshots-fix
snapshots-fix: 
	uv run pytest --inline-snapshot=fix 

.PHONY: snapshots-create 
snapshots-create: 
	uv run pytest --inline-snapshot=create 

.PHONY: old_version_tests
old_version_tests: 
	UV_PROJECT_ENVIRONMENT=.venv_39 uv run --python 3.9 -m pytest

.PHONY: build-docs
build-docs:
	uv run mkdocs build

.PHONY: build-full-docs
build-full-docs:
	uv run docs/scripts/translate_docs.py
	uv run mkdocs build

.PHONY: serve-docs
serve-docs:
	uv run mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	uv run mkdocs gh-deploy --force --verbose

.PHONY: dist
dist: clean
	uv pip install build
	python -m build

.PHONY: check-all
check-all: check tests coverage

.PHONY: all
all: format lint mypy tests build-docs


