install_training_pipeline:
	@echo "Installing training pipeline..."
	
	poetry env use $(shell which python3.10) && \
	poetry install && \
	poetry run pip install torch==2.0.1 && \

export_requirements:
	@echo "Exporting requirements..."

	rm requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

run_training_pipeline_locally:
	@echo "Running training pipeline locally..."
	
	poetry run python -m ../tools/train_finqa

run_training_pipeline_on_beam:
	@echo "Running training pipeline on beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_finqa.py:train -d '{"config_file": "configs/dev_config.yaml", "output_dir": "./results", "dataset_dir": "./dataset/dataset"}'

upload_dataset_to_beam:
	beam volume upload train_finqa_dataset dataset
