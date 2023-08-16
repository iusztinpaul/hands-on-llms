install:
	@echo "Installing training pipeline..."
	
	poetry env use $(shell which python3.10) && \
	poetry install && \
	poetry run pip install torch==2.0.1

add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add $(package)

add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add --group dev $(package)

export_requirements:
	@echo "Exporting requirements..."

	rm requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

upload_dataset_to_beam:
	beam volume upload train_finqa_dataset dataset

help:
	poetry run python -m tools.train_finqa --help

dev_train_locally:
	@echo "Running training pipeline locally using the development config..."
	
	poetry run python -m tools.train_finqa --config_file configs/dev_training_config.yaml --output_dir ../results --dataset_dir ./dataset

train_locally:
	@echo "Running training pipeline locally using the production config..."
	
	poetry run python -m tools.train_finqa --config_file configs/training_config.yaml --output_dir ../results --dataset_dir ./dataset

dev_train_on_beam:
	@echo "Running training pipeline on Beam using the development config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_finqa.py:train -d '{"config_file": "configs/dev_training_config.yaml", "output_dir": "../results", "dataset_dir": "./dataset/dataset", "env_file_path": "env"}'

train_on_beam:
	@echo "Running training pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_finqa.py:train -d '{"config_file": "configs/training_config.yaml", "output_dir": "../results", "dataset_dir": "./dataset/dataset", "env_file_path": "env"}'
