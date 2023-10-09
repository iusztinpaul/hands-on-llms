### Install ###

install:
	@echo "Installing financial bot..."
	
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install && \
	poetry run pip install torch==2.0.1

install_dev: install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --only dev

install_only_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --only dev

add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add $(package)

add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add --group dev $(package)


# === Run ===

run:
	@echo "Running financial_bot..."

	@echo "LD_LIBRARY_PATH: $(LD_LIBRARY_PATH)"

	poetry run python -m tools.run_chain


### PEP 8 ###
# Be sure to install the dev dependencies first #

lint_check:
	@echo "Checking for linting issues..."

	poetry run ruff check .

lint_fix:
	@echo "Fixing linting issues..."

	poetry run ruff check --fix .

format_check:
	@echo "Checking for formatting issues..."

	poetry run black --check .

format_fix:
	@echo "Formatting code..."

	poetry run black .
