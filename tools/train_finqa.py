import sys

sys.path.append("../modules")

from beam import App, Runtime, Image, Output, Volume

from training.api import FinQATrainingAPI
from training.constants import ROOT_DIR


def read_requirements(file_path):
    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


requirements = read_requirements("modules/training/requirements.txt")
training_app = App(
    name="train_finqa",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        image=Image(python_version="python3.10", python_packages=requirements),
    ),
    volumes=[Volume(name="root_dir", path=str(ROOT_DIR))],
)


@training_app.run()
def train():
    training_api = FinQATrainingAPI(debug=True)
    training_api.train()


if __name__ == "__main__":
    train()
