# === Install ===

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

	poetry run python -m tools.bot


# === Beam ===

export_requirements:
	@echo "Exporting requirements..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes

run_beam: export_requirements
	@echo "Running financial_bot on Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/bot.py:run -d '{"env_file_path": "env", "model_cache_dir": "./model_cache"}'

deploy_beam: export_requirements
	@echo "Deploying financial_bot on Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam deploy ./tools/bot.py:run

call_restful_api:
	curl -X POST  \
	--compressed 'https://apps.beam.cloud/${DEPLOYMENT_ID}' \
	-H 'Accept: */*' \
	-H 'Accept-Encoding: gzip, deflate' \
	-H 'Authorization: Basic ${TOKEN}' \
	-H 'Connection: keep-alive' \
	-H 'Content-Type: application/json' \
	-d '{"about_me": "I am a student and I have some money that I want to invest.", "question": "Should I consider investing in stocks from the Tech Sector?"}'

# === Formatting & Linting ===
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
