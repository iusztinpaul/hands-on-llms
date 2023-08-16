from pathlib import Path

import fire

from beam import App, Runtime, Image, Volume

from training_pipeline import configs, utils


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
        Volume(path="../results", name="train_finqa_results"),
        Volume(path="../model_cache", name="model_cache"),
    ],
)


@training_app.run()
def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
):
    import logging

    from training_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from training_pipeline import utils
    from training_pipeline.api import FinQATrainingAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    root_dataset_dir = Path(dataset_dir)

    training_config = configs.TrainingConfig.from_yaml(config_file, output_dir)
    training_api = FinQATrainingAPI.from_config(
        config=training_config, root_dataset_dir=root_dataset_dir
    )
    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
