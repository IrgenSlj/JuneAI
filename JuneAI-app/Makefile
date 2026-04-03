.PHONY: all bootstrap bootstrap-force format lint test tests test_watch integration_tests docker_tests help extended_tests verify-env run smoke

# Default target executed when no arguments are given to make.
all: help

# Define the interpreters used by make targets.
SYSTEM_PYTHON ?= python3
VENV_DIR ?= .venv
ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
CODESPELL := $(VENV_DIR)/Scripts/codespell.exe
else
VENV_PYTHON := $(VENV_DIR)/bin/python3.9
CODESPELL := $(VENV_DIR)/bin/codespell
endif

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
SMOKE_PORT ?= 8501
SMOKE_TIMEOUT ?= 45

bootstrap:
	$(SYSTEM_PYTHON) scripts/bootstrap_env.py

bootstrap-force:
	$(SYSTEM_PYTHON) scripts/bootstrap_env.py --force

verify-env:
	$(VENV_PYTHON) scripts/verify_env.py

check-ollama:
	$(VENV_PYTHON) scripts/check_ollama.py

run:
	$(VENV_PYTHON) -m streamlit run app.py --server.headless true --server.port 8501

smoke:
	$(VENV_PYTHON) scripts/smoke_app.py --port $(SMOKE_PORT) --timeout $(SMOKE_TIMEOUT)

test:
	$(VENV_PYTHON) -m pytest $(TEST_FILE)

integration_tests:
	$(VENV_PYTHON) -m pytest tests/integration_tests

test_watch:
	$(VENV_PYTHON) -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	$(VENV_PYTHON) -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	$(VENV_PYTHON) -m pytest --only-extended $(TEST_FILE)


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
	$(VENV_PYTHON) -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || $(VENV_PYTHON) -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || $(VENV_PYTHON) -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(VENV_PYTHON) -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(VENV_PYTHON) -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	$(VENV_PYTHON) -m ruff format $(PYTHON_FILES)
	$(VENV_PYTHON) -m ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	$(CODESPELL) --toml pyproject.toml README.md src/

spell_fix:
	$(CODESPELL) --toml pyproject.toml README.md src/ -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'bootstrap                   - create .venv, install dependencies, and verify the environment'
	@echo 'bootstrap-force             - recreate .venv before bootstrapping'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'run                          - start Streamlit locally from the venv'
	@echo 'smoke                        - start Streamlit and verify HTTP 200'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'verify-env                   - run the environment verification script'
