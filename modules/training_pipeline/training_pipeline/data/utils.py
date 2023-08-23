import json
from pathlib import Path
from typing import List, Union

import yaml


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    return data


def write_json(data: Union[dict, List[dict]], path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
