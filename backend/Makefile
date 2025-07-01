.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests dev init 

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	uv run --with-editable . pytest $(TEST_FILE)

test_watch:
	uv run --with-editable . ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	uv run --with-editable . pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	uv run --with-editable . pytest --only-extended $(TEST_FILE)

dev:
	@echo "Starting backend development server..."
	@set BG_JOB_ISOLATED_LOOPS=true
	@langgraph dev --debug-port 2025

init:
	@echo "Preparing backend uv packages"
	@pip install -e '.[dev]'

build:
	@echo "Building backend"
	@uv build

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'make format                       - run code formatters'
	@echo 'make lint                         - run linters'
	@echo 'make test                         - run unit tests'
	@echo 'make tests                        - run unit tests'
	@echo 'make test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'make test_watch                   - run unit tests in watch mode'
	@echo 'make init   						 - Initializes the project runtime'
	@echo 'make build  						 - Builds the project'
	@echo 'make dev    						 - Starts the backend server (fastAPI)''
