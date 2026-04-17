.PHONY: all bootstrap bootstrap-force format lint lint_repo lint_tests typecheck_src test tests test_watch integration_tests docker_tests help extended_tests verify-env run smoke export-memory docker-build docker-up docker-down

# Default target executed when no arguments are given to make.
all: help

# Define the interpreters used by make targets.
SYSTEM_PYTHON ?= python3
VENV_DIR ?= .venv
ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
CODESPELL := $(VENV_DIR)/Scripts/codespell.exe
else
VENV_PYTHON := $(VENV_DIR)/bin/python
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

export-memory:
	$(VENV_PYTHON) scripts/export_memory.py --user $(USER_ID)

docker-build:
	docker build -t juneai .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

test tests:
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
SRC_FILES=src
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint:
	$(VENV_PYTHON) -m ruff check $(SRC_FILES)
	mkdir -p $(MYPY_CACHE) && $(VENV_PYTHON) -m mypy $(SRC_FILES) --cache-dir $(MYPY_CACHE)

typecheck_src:
	mkdir -p $(MYPY_CACHE) && $(VENV_PYTHON) -m mypy $(SRC_FILES) --cache-dir $(MYPY_CACHE)

lint_repo:
	$(VENV_PYTHON) -m ruff check .

lint_tests:
	$(VENV_PYTHON) -m ruff check tests scripts app.py

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
	@echo 'lint                         - run enforced source checks (ruff + mypy on src/)'
	@echo 'lint_repo                    - run ruff across the whole repository'
	@echo 'lint_tests                   - run ruff on tests, scripts, and app.py'
	@echo 'typecheck_src                - run mypy on src/'
	@echo 'run                          - start Streamlit locally from the venv'
	@echo 'smoke                        - start Streamlit and verify HTTP 200'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'verify-env                   - run the environment verification script'
	@echo 'export-memory                - export memory to JSON (USER_ID=admin)'
	@echo 'docker-build                 - build the Docker image'
	@echo 'docker-up                    - start the stack with docker compose (detached)'
	@echo 'docker-down                  - stop the docker compose stack'
