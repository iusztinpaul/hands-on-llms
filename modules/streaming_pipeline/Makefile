### Install ###

install:
	@echo "Installing streaming pipeline..."
	
	poetry env use $(shell which python3.10) && \
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install

	# Other dependencies
	# TODO: Extract to a separate script
	sudo apt-get install jq

install_dev: install
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install --only dev

install_only_dev:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install --only dev

add:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add $(package)

add_dev:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add --group dev $(package)


### Run ###

run:
	export RUST_BACKTRACE=full; poetry run python -m bytewax.run tools.run:build_flow

run_dev:
	export RUST_BACKTRACE=full; poetry run python -m tools.run_dev

search:
	poetry run python -m tools.search ${PARAMS}


### Run Docker ###

build:
	@echo "Build docker image"

	docker build -t streaming_pipeline -f deploy/Dockerfile .

run_docker: 
	@echo "Run docker image"
	
	docker run --rm \
		-e BYTEWAX_PYTHON_FILE_PATH=tools.run:build_flow \
		-e ALPACA_API_KEY=${ALPACA_API_KEY} \
		-e ALPACA_API_SECRET=${ALPACA_API_SECRET} \
		-e QDRANT_API_KEY=${QDRANT_API_KEY} \
		-e QDRANT_URL=${QDRANT_URL} \
		--name streaming_pipeline \
		streaming_pipeline

run_docker_dev:
	@echo "Run docker image"

	docker run -it --rm \
		--env-file .env \
		-e DEBUG=true \
		--name streaming_pipeline \
		streaming_pipeline


### Deploy AWS ###

deploy-aws:
	chmod +x deploy/create_user_data.sh
	chmod +x deploy/launch_ec2.sh 

	bash deploy/launch_ec2.sh

undeploy-aws:
	bash deploy/terminate_ec2.sh

info-aws:
	aws ec2 describe-instances --filters "Name=tag:Name,Values=streaming-pipeline-server"


### Deploy Waxctl ###

NAME := streaming_pipeline_test

# generate a tar file with project files to send to AWS EC2 instance
deployment-files:
	if [ -d test ]; then rm -rf test; fi

	tar \
	--exclude __pycache__ \
	--exclude .ruff_cache \
	--exclude logs \
	--exclude .beamignore \
	--exclude .env.example \
	--exclude .env \
	--exclude Makefile \
	--exclude *.tar \
	--exclude .DS_Store \
	--exclude setup_ec2.sh \
	--exclude README.md \
	--exclude requirements.txt \
	--exclude poetry.lock \
	--exclude user-data \
	--exclude tools \
	--warning=no-file-changed \
	-cvzf project-files.tar -C . .

	mkdir test; cp project-files.tar test; cd test; tar -xvzf project-files.tar 


# deploy the feature-pipeline to an EC2 instance on AWS
deploy-waxctl: deployment-files
	# --requirements-file-name requirements.txt \
	# --python-package \

	# Export env vars
	export $(cat .env | xargs)

	waxctl aws deploy project-files.tar \
		--python-file-name streaming_pipeline/run.py \
		--requirements-file-name requirements.txt \
		--name ${NAME} \
		--system-setup-file-name ./setup_ec2.sh \
		--instance-type t2.small \
		--region eu-central-1 \
		--save-cloud-config \
		--debug \
		-E ALPACA_API_KEY=${ALPACA_API_KEY},ALPACA_API_SECRET=${ALPACA_API_SECRET},QDRANT_API_KEY=${QDRANT_API_KEY},QDRANT_URL=${QDRANT_URL}

clean_aws:
	# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage_delete.html
	aws iam remove-role-from-instance-profile --instance-profile-name Waxctl-EC2-eu-central-1-streaming_pipeline-InstanceProfile --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role
	aws iam detach-role-policy --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role --policy-arn arn:aws:iam::994231256807:policy/Waxctl-EC2-eu-central-1-streaming_pipeline-Policy
	aws iam delete-role --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role
	aws iam delete-policy --policy-arn arn:aws:iam::994231256807:policy/Waxctl-EC2-eu-central-1-streaming_pipeline-Policy
	aws iam delete-instance-profile --instance-profile-name Waxctl-EC2-eu-central-1-streaming_pipeline-InstanceProfile

# list all bytewax deployments
info:
	waxctl aws ls --verbose --name ${NAME}

# stop the feature-pipeline and delete the EC2 instance
undeploy-waxctl:
	waxctl aws delete --name ${NAME} --yes


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
