import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError


class TimeHelper:
    """Helper class for time operations."""

    def get_now_local(
        self,
        timezone: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        """Get the current local time in specifci timezone."""

        if timezone is None:
            timezone = self._get_default_timezone()

        if format is None:
            format = self._get_default_format()

        return datetime.datetime.now(ZoneInfo(timezone)).strftime(format)

    @staticmethod
    def _get_default_timezone() -> str:
        """Get the default timezone from the configuration."""

        timezone = ConfigLoader().get("datetime.timezone")

        if not isinstance(timezone, str):
            raise ConfigurationError(
                "Invalid timezone in configuration. Expected a string (e.g. 'Europe/Bucharest')."
            )

        return timezone

    @staticmethod
    def _get_default_format() -> str:
        """Get the default format from the configuration."""

        format = ConfigLoader().get("datetime.format")

        if not isinstance(format, str):
            raise ConfigurationError(
                "Invalid format in configuration. Expected a string (e.g. '%Y-%m-%d %H:%M:%S')."
            )

        return format
