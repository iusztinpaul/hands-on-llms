import json
from pathlib import Path
from typing import List, Union

import yaml


def load_json(path: Path) -> dict:
    """
    Load JSON data from a file.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        dict: The JSON data as a dictionary.
    """

    with path.open("r") as f:
        data = json.load(f)

    return data


def write_json(data: Union[dict, List[dict]], path: Path) -> None:
    """
    Write a dictionary or a list of dictionaries to a JSON file.

    Args:
        data (Union[dict, List[dict]]): The data to be written to the file.
        path (Path): The path to the file.

    Returns:
        None
    """

    with path.open("w") as f:
        json.dump(data, f, indent=4)


def load_yaml(path: Path) -> dict:
    """
    Load a YAML file from the given path and return its contents as a dictionary.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
