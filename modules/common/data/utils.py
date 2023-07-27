import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    return data
