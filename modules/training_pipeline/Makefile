### Setup ###

install:
	@echo "Installing training pipeline..."
	
	poetry env use $(shell which python3.10) && \
	poetry install && \
	poetry run pip install torch==2.0.1

add:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add $(package)

add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add --group dev $(package)

export_requirements:
	@echo "Exporting requirements..."

	rm requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

upload_dataset_to_beam:
	beam volume upload finqa_dataset dataset


### Training ###

train_help:
	poetry run python -m tools.train_finqa --help

dev_train_local:
	@echo "Running training pipeline locally using the development config..."
	
	poetry run python -m tools.train_finqa --config_file configs/dev_training_config.yaml --output_dir ./output --dataset_dir ./dataset

train_local:
	@echo "Running training pipeline locally using the production config..."
	
	poetry run python -m tools.train_finqa --config_file configs/training_config.yaml --output_dir ./output --dataset_dir ./dataset

dev_train_beam:
	@echo "Running training pipeline on Beam using the development config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_finqa.py:train -d '{"config_file": "configs/dev_training_config.yaml", "output_dir": "./output", "dataset_dir": "./dataset/dataset", "env_file_path": "env", "model_cache_dir": "./model_cache"}'

train_beam:
	@echo "Running training pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_finqa.py:train -d '{"config_file": "configs/training_config.yaml", "output_dir": "./output", "dataset_dir": "./dataset/dataset", "env_file_path": "env", "model_cache_dir": "./model_cache"}'


### Inference ###

dev_infer_local:
	@echo "Running inference pipeline locally using the production config..."

	poetry run python -m tools.inference_finqa --config_file configs/dev_inference_config.yaml --dataset_dir ./dataset

infer_local:
	@echo "Running inference pipeline locally using the production config..."

	poetry run python -m tools.inference_finqa --config_file configs/inference_config.yaml --dataset_dir ./dataset

dev_infer_beam:
	@echo "Running inference pipeline on Beam using the development config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_finqa.py:infer -d '{"config_file": "configs/dev_inference_config.yaml", "dataset_dir": "./dataset/dataset", "env_file_path": "env", "model_cache_dir": "./model_cache"}'

infer_beam:
	@echo "Running inference pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_finqa.py:infer -d '{"config_file": "configs/inference_config.yaml", "dataset_dir": "./dataset/dataset", "env_file_path": "env", "model_cache_dir": "./model_cache"}'
