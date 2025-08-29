from pathlib import Path
from typing import Union

import envtoml
from dotenv import load_dotenv

from .attr_dict import AttrDict


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""


class ConfigLoader:
    """
    A simple configuration loader that reads a TOML file and provides dot-notation access.
    """

    def __init__(self, config_path: str = "/workspaces/GEST/config.toml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file {self.config_path} does not exist."
            )
        load_dotenv()
        config = envtoml.load(self.config_path)
        self._config = AttrDict(config)

    def get(self, key: str) -> Union[AttrDict, str, None]:
        """Dot-notation access for sections, returns AttrDicts recursively."""
        parts = key.split(".")
        value = self._config
        for part in parts:
            value = getattr(value, part)
        return value
