# Export env variables from .env file if it exists.
ifneq (,$(wildcard .env))
include .env
export
endif

### Install ###

install-debian:
	sudo apt-get install jq

install:
	@echo "Installing streaming pipeline..."
	
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install

install_dev: install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --with dev

install_only_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --only dev

add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add $(package)

add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add --group dev $(package)


### Run ###

run_real_time:
	RUST_BACKTRACE=full poetry run python -m bytewax.run tools.run_real_time:build_flow

run_real_time_dev:
	RUST_BACKTRACE=full poetry run python -m bytewax.run "tools.run_real_time:build_flow(debug=True)"

run_batch:
	RUST_BACKTRACE=full poetry run python -m bytewax.run -p4 "tools.run_batch:build_flow(latest_n_days=8)"

run_batch_dev:
	RUST_BACKTRACE=full poetry run python -m bytewax.run "tools.run_batch:build_flow(latest_n_days=2, debug=True)"

search:
	poetry run python -m tools.search ${PARAMS}


### Run Docker ###

build:
	@echo "Build docker image"

	docker build -t streaming_pipeline:latest -f deploy/Dockerfile .

run_real_time_docker: 
	@echo "Run docker image"
	
	docker run --rm \
		-e BYTEWAX_PYTHON_FILE_PATH=tools.run_real_time:build_flow \
		-e ALPACA_API_KEY=${ALPACA_API_KEY} \
		-e ALPACA_API_SECRET=${ALPACA_API_SECRET} \
		-e QDRANT_API_KEY=${QDRANT_API_KEY} \
		-e QDRANT_URL=${QDRANT_URL} \
		--name streaming_pipeline \
		streaming_pipeline:latest

run_docker_dev:
	@echo "Run docker image"

	docker run -it --rm \
		--env-file .env \
		-e DEBUG=true \
		--name streaming_pipeline \
		streaming_pipeline:latest


### Deploy AWS ###

deploy_aws:
	chmod +x deploy/create_user_data.sh
	chmod +x deploy/launch_ec2.sh 

	bash deploy/launch_ec2.sh

info_aws:
	aws ec2 describe-instances --filters "Name=tag:Name,Values=streaming-pipeline-server"

undeploy_aws:
	bash deploy/terminate_ec2.sh


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
