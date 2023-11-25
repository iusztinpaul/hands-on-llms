.PHONY: help install install_dev add add_dev export_requirements dev_train_local train_local dev_train_beam train_beam dev_infer_local infer_local dev_infer_beam infer_beam lint_check lint_fix format_check format_fix

# === Install ===

install:
	@echo "Installing training pipeline..."
	
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install
	# poetry run pip install torch==2.0.1

install_dev: install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --with dev

install_only_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --only dev

add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add $(package)

add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add --group dev $(package)


# === Beam ===

export_requirements:
	@echo "Exporting requirements..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes

upload_dataset_to_beam:
	@echo "Pushing data to the qa_dataset volume on Beam..."
	
	beam volume upload qa_dataset dataset


# === Training ===

dev_train_local:
	@echo "Running training pipeline locally using the development config..."
	
	poetry run python -m tools.train_run --config_file configs/dev_training_config.yaml --output_dir ./output --dataset_dir ./dataset

train_local:
	@echo "Running training pipeline locally using the production config..."
	
	poetry run python -m tools.train_run --config_file configs/training_config.yaml --output_dir ./output --dataset_dir ./dataset

dev_train_beam: export_requirements
	@echo "Running training pipeline on Beam using the development config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_run.py:train -d '{"config_file": "configs/dev_training_config.yaml", "output_dir": "./output", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'

train_beam: export_requirements
	@echo "Running training pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_run.py:train -d '{"config_file": "configs/training_config.yaml", "output_dir": "./output", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'


# === Inference ===

dev_infer_local:
	@echo "Running inference pipeline locally using the production config..."

	poetry run python -m tools.inference_run --config_file configs/dev_inference_config.yaml --dataset_dir ./dataset

infer_local:
	@echo "Running inference pipeline locally using the production config..."

	poetry run python -m tools.inference_run --config_file configs/inference_config.yaml --dataset_dir ./dataset

dev_infer_beam: export_requirements
	@echo "Running inference pipeline on Beam using the development config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_run.py:infer -d '{"config_file": "configs/dev_inference_config.yaml", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'

infer_beam: export_requirements
	@echo "Running inference pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_run.py:infer -d '{"config_file": "configs/inference_config.yaml", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'


# === PEP8 ===
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
