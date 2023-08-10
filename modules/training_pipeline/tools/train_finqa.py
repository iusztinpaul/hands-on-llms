from pathlib import Path
from training_pipeline import utils

from beam import App, Runtime, Image, Volume


requirements = utils.read_requirements("requirements.txt")
training_app = App(
    name="train_finqa",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        # TODO: Install requirements using Poetry & custom commands.
        image=Image(python_version="python3.10", python_packages=requirements),
    ),
    volumes=[
        Volume(path="./dataset", name="train_finqa_dataset"),
        Volume(path="./results", name="train_finqa_results"),
        Volume(path="./model_cache", name="model_cache"),
    ],
)


@training_app.run()
def train(config_file: str, output_dir: str, dataset_dir: str):
    import logging

    from training_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(env_file_path="env")

    from training_pipeline import utils, constants
    from training_pipeline.api import FinQATrainingAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    root_dataset_dir = Path(dataset_dir)

    config = constants.load_config(config_file)
    training_arguments = constants.build_training_arguments(config=config, output_dir=output_dir)

    training_api = FinQATrainingAPI(
        root_dataset_dir=root_dataset_dir,
        model_id=config["model"]["id"],
        training_arguments=training_arguments,
        max_seq_length=config["model"]["max_seq_length"],
        debug=config["setup"]["debug"],
    )
    training_api.train()


if __name__ == "__main__":
    train()
