import json
from pathlib import Path

import yaml


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    return data


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        config = yaml.safe_load(f)
    
    return config
