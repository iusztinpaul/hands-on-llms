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


### Run ###
run:
	export RUST_BACKTRACE=1; poetry run python -m bytewax.run streaming_pipeline.alpaca:flow