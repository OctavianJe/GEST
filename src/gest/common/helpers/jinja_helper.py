from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, Template

from .config_loader import ConfigLoader, ConfigurationError


class JinjaHelper:
    """Helper class to manage Jinja templates."""

    def __init__(self, dir: Optional[Path] = None):
        if dir is None:
            dir = self._get_default_dir()
        self.env = Environment(loader=FileSystemLoader(dir), autoescape=False)

    def get_template(self, template_name: str) -> Template:
        """Returns a Jinja template by name."""

        return self.env.get_template(template_name)

    @staticmethod
    def _get_default_dir() -> Path:
        """Get the default directory from the configuration."""

        dir = ConfigLoader().get("gest.templates.dir")

        if not isinstance(dir, str):
            raise ConfigurationError(
                "Invalid directory in configuration. Expected a string path."
            )

        return Path(dir)
