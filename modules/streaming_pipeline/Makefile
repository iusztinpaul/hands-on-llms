### Install ###

install:
	@echo "Installing streaming pipeline..."
	
	poetry env use $(shell which python3.10) && \
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install

install_dev: install
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install --only dev

install_only_dev:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry install --only dev

add:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add $(package)

add_dev:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add --group dev $(package)


### Beam ###

export_requirements:
	@echo "Exporting requirements..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes


### Run ###
run:
	export RUST_BACKTRACE=full; poetry run python -m bytewax.run tools.run:build_flow

run_dev:
	export RUST_BACKTRACE=full; poetry run python -m tools.run_dev

search:
	poetry run python -m tools.search ${PARAMS}

# generate a tar file with project files to send to AWS EC2 instance
deployment-files:
	tar \
	--exclude={'__pycache__','.ruff_cache','logs','.beamignore','.env.example','Makefile','poetry.lock','poetry.toml','*.tar','.DS_Store','README.md','setup_ec2.sh'} \
	-vzcf project-files.tar -C . .

# deploy the feature-pipeline to an EC2 instance on AWS
deploy: deployment-files
	waxctl aws deploy project-files.tar \
		--python-file-name tools/run.py \
		--requirements-file-name requirements.txt \
		--name "streaming_pipeline" \
		--instance-type t2.small \
		--region eu-central-1

# list all bytewax deployments
info:
	waxctl aws ls --verbose --name "streaming_pipeline"


clean_aws:
	# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage_delete.html
	aws iam remove-role-from-instance-profile --instance-profile-name Waxctl-EC2-eu-central-1-streaming_pipeline-InstanceProfile --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role
	aws iam detach-role-policy --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role --policy-arn arn:aws:iam::994231256807:policy/Waxctl-EC2-eu-central-1-streaming_pipeline-Policy
	aws iam delete-role --role-name Waxctl-EC2-eu-central-1-streaming_pipeline-Role
	aws iam delete-policy --policy-arn arn:aws:iam::994231256807:policy/Waxctl-EC2-eu-central-1-streaming_pipeline-Policy
	aws iam delete-instance-profile --instance-profile-name Waxctl-EC2-eu-central-1-streaming_pipeline-InstanceProfile

# stop the feature-pipeline and delete the EC2 instance
undeploy:
	waxctl aws delete --name "streaming_pipeline" --yes


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
