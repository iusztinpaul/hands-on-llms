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

	poetry run python -m tools.bot \
		--about_me "I am a student and I have some money that I want to invest." \
		--question "Should I consider investing in stocks from the Tech Sector?" \
		--debug False

run_dev:
	@echo "Running financial_bot [Dev Mode]..."

	poetry run python -m tools.bot \
		--about_me "I am a student and I have some money that I want to invest." \
		--question "Should I consider investing in stocks from the Tech Sector?" \
		--debug True

run_ui:
	@echo "Running financial_bot UI..."

	poetry run python -m tools.ui

run_ui_dev:
	@echo "Running financial_bot UI [Dev Mode]..."

	poetry run gradio tools/ui.py --debug


# === Beam ===

export_requirements:
	@echo "Exporting requirements..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes

deploy_beam: export_requirements
	@echo "Deploying financial_bot on Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam deploy ./tools/bot.py:run

deploy_beam_dev: export_requirements
	@echo "Deploying financial_bot on Beam [Dev Mode]..."

	BEAM_IGNORE_IMPORTS_OFF=true beam deploy ./tools/bot.py:run_dev

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
