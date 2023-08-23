import fire

from pathlib import Path

from beam import App, Runtime, Image, Volume, VolumeType

from training_pipeline import configs, utils


requirements = utils.read_requirements("requirements.txt")
training_app = App(
    name="train_finqa",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        # TODO: Install requirements using Poetry & custom commands.
        image=Image(python_version="python3.10", python_packages=requirements),
    ),
    volumes=[
        Volume(path="./dataset", name="finqa_dataset"),
        Volume(
            path="./output",
            name="train_finqa_output",
            volume_type=VolumeType.Persistent,
        ),
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


@training_app.run()
def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
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
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None

    training_config = configs.TrainingConfig.from_yaml(config_file, output_dir)
    training_api = FinQATrainingAPI.from_config(
        config=training_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )
    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
