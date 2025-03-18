import os
from typing import Any

import yaml


def include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Custom constructor to handle !include directives in YAML files."""
    filename = node.value
    base_path = os.path.dirname(loader.name)
    file_path = os.path.join(base_path, filename)

    with open(file_path, "r") as f:
        return yaml.safe_load(f)


yaml.SafeLoader.add_constructor("!include", include_constructor)


def load_yaml_content(file: str) -> dict[str, Any]:
    yaml_file = os.path.join("src", "config", "yaml", file)

    if not yaml_file.endswith(".yaml"):
        yaml_file += ".yaml"

    with open(yaml_file, "r") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    default = raw_config.get("default", {})
    if default != {}:
        del raw_config["default"]
    config = update_default_config(default, raw_config)

    return config


def update_default_config(
    default: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    config = default.copy()

    for key, value in updates.items():
        if key in config and isinstance(value, dict) and isinstance(config[key], dict):
            config[key] = update_default_config(config[key], value)

        else:
            config[key] = value

    return config
