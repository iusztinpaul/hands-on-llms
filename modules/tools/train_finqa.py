import os
import sys
sys.path.append(".")

from beam import App, Runtime, Image, Output, Volume, VolumeType


def print_files_and_subdirs(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for dirpath, dirnames, filenames in os.walk(directory_path):
            print(f"Directory: {dirpath}")
            for filename in filenames:
                print(f"File: {os.path.join(dirpath, filename)}")
            for dirname in dirnames:
                print(f"Sub-directory: {os.path.join(dirpath, dirname)}")
    else:
        print(f"The directory '{directory_path}' does not exist")


def print_available_gpu_memory():
    import torch
    import subprocess

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info = subprocess.check_output(f'nvidia-smi -i {i} --query-gpu=memory.free --format=csv,nounits,noheader', shell=True)
            memory_info = str(memory_info).split("\\")[0][2:]
            print(f"GPU {i} memory available: {memory_info} MiB")
    else:
        print("No GPUs available")


def print_available_ram():
    import psutil

    memory_info = psutil.virtual_memory()
    print(f"Available RAM: {memory_info.available / (1024.0 ** 3)} GB")  # convert bytes to GB


print("#" * 100)
print_files_and_subdirs(os.getcwd())
print("#" * 100)


def read_requirements(file_path):
    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    requirements += ["psutil"]

    return requirements


requirements = read_requirements("training/requirements.txt")
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
        Volume(name="train_finqa_dataset", path="dataset", volume_type=VolumeType.Persistent),
        Volume(name="train_finqa_results", path="results")
        ],
)



@training_app.run()
def train():
    import torch

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()

        # Set device map dynamically for multi-gpu training
        device_map = {i: [i] for i in range(device_count)}
        print(f"Using {device_count} GPUs: {device_map}")

    else:
        device_map = None
        print("No GPUs available, using CPU.")

    print("#" * 100)
    print_available_gpu_memory()
    print_available_ram()
    print("#" * 100)
    print()

    print("#" * 100)
    print_files_and_subdirs("/workspace/dataset")
    print("#" * 100)

    from training.api import FinQATrainingAPI
    training_api = FinQATrainingAPI(debug=True)
    training_api.train()


if __name__ == "__main__":
    train()
