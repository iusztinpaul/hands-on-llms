import logging
import os
import subprocess

import psutil
import torch

logger = logging.getLogger(__name__)


def log_available_gpu_memory():
    """
    Logs the available GPU memory for each available GPU device.

    If no GPUs are available, logs "No GPUs available".

    Returns:
        None
    """

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info = subprocess.check_output(
                f"nvidia-smi -i {i} --query-gpu=memory.free --format=csv,nounits,noheader",
                shell=True,
            )
            memory_info = str(memory_info).split("\\")[0][2:]

            logger.info(f"GPU {i} memory available: {memory_info} MiB")
    else:
        logger.info("No GPUs available")


def log_available_ram():
    """
    Logs the amount of available RAM in gigabytes.

    Returns:
        None
    """

    memory_info = psutil.virtual_memory()

    logger.info(
        f"Available RAM: {memory_info.available / (1024.0 ** 3):.2f} GB"
    )  # convert bytes to GB


def read_requirements(file_path):
    """
    Reads a file containing a list of requirements and returns a list of stripped requirements.

    Args:
        file_path (str): The path to the file containing the requirements.

    Returns:
        list: A list of stripped requirements.
    """

    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


def log_files_and_subdirs(directory_path: str):
    """
    Logs all files and subdirectories in the specified directory.

    Args:
        directory_path (str): The path to the directory to log.

    Returns:
        None
    """

    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for dirpath, dirnames, filenames in os.walk(directory_path):
            logger.info(f"Directory: {dirpath}")
            for filename in filenames:
                logger.info(f"File: {os.path.join(dirpath, filename)}")
            for dirname in dirnames:
                logger.info(f"Sub-directory: {os.path.join(dirpath, dirname)}")
    else:
        logger.info(f"The directory '{directory_path}' does not exist")
