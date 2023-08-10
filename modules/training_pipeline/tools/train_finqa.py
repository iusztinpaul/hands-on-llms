from training_pipeline import utils

from beam import App, Runtime, Image, Output, Volume, VolumeType


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
def train():
    import logging

    from training_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(env_file_path="env")

    from training_pipeline import utils
    from training_pipeline.api import FinQATrainingAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    training_api = FinQATrainingAPI(debug=True)
    training_api.train()


if __name__ == "__main__":
    train()
