### Install ###

install:
	@echo "Installing training pipeline..."
	
	poetry env use $(shell which python3.10) && \
	poetry install

install_dev: install
	poetry install --only dev

install_only_dev:
	poetry install --only dev

add:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add $(package)

add_dev:
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; poetry add --group dev $(package)