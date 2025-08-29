import base64
from email.message import EmailMessage
from typing import Dict, List

from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from retry import retry

from gest.common.helpers.decorators.enable_guard_decorator import requires_enabled

from .config_loader import ConfigLoader, ConfigurationError


class GmailHelper:
    """Helper class to manage alert emails via Gmail (using OAuth2)."""

    _from: str
    _to: List[str]
    _enabled: bool

    def __init__(self) -> None:
        self._from = self._get_from_email()
        self._to = self._get_to_emails()
        self._enabled = self._get_enabled()

        creds, _ = default(scopes=["https://www.googleapis.com/auth/gmail.send"])
        self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    @requires_enabled(attr="_enabled", expected=True)
    @retry(HttpError, delay=2, backoff=2, tries=5)
    def send_email(self, subject: str, body: str) -> Dict:
        """Send the message and return the Gmail API response JSON."""

        msg = self._build_message(subject, body)
        encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        try:
            return (
                self._service.users()
                .messages()
                .send(userId="me", body={"raw": encoded})
                .execute()
            )
        except HttpError as err:
            raise RuntimeError(f"Gmail API error: {err}") from err

    def _build_message(self, subject: str, body: str) -> EmailMessage:
        """Build the email message to be sent."""

        msg = EmailMessage()

        msg["To"] = ", ".join(self._to)
        msg["From"] = self._from
        msg["Subject"] = subject
        msg.set_content(body)

        return msg

    @staticmethod
    def _get_enabled() -> bool:
        """Check if email alerts are enabled in the configuration."""

        enabled = ConfigLoader().get("gest.alerts.enabled")

        if not isinstance(enabled, bool):
            raise ConfigurationError(
                "Invalid 'enabled' property in configuration. Expected a bool."
            )

        return enabled

    @staticmethod
    def _get_from_email() -> str:
        """Get the 'From' email address from the configuration."""

        email = ConfigLoader().get("gest.alerts.from_email")

        if not isinstance(email, str):
            raise ConfigurationError(
                "Invalid email in configuration. Expected a string."
            )

        if email.strip() == "":
            raise ConfigurationError(
                "Empty email address found in configuration. All emails must be non-empty strings."
            )

        return email

    @staticmethod
    def _get_to_emails() -> List[str]:
        """Get the 'To' email addresses from the configuration."""

        emails = ConfigLoader().get("gest.alerts.to_emails")

        if not isinstance(emails, list):
            raise ConfigurationError(
                "Invalid email list in configuration. Expected a list of strings."
            )

        if any(not isinstance(email, str) for email in emails):
            raise ConfigurationError(
                "Invalid email in configuration. Expected a string."
            )

        if any(email.strip() == "" for email in emails):
            raise ConfigurationError(
                "Empty email address found in configuration. All emails must be non-empty strings."
            )

        return emails
