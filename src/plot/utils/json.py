import json
import os
from typing import Any


def load_json_result(file: str) -> dict[str, Any]:
    with open(file) as f:
        res = json.load(f)

    return res


def get_result_file(filename: str) -> str:
    if not filename.endswith(".json"):
        filename += ".json"

    return os.path.join("results", filename)


if __name__ == "__main__":
    load_json_result("results/eesm19_bitfit_usleep_usleep.json")
