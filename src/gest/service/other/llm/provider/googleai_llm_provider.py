import json
import logging
import re
from http import HTTPStatus
from json import JSONDecodeError
from typing import ClassVar, List, Optional

from google.genai import Client
from google.genai.errors import APIError, ServerError
from google.genai.types import (
    ContentUnion,
    GenerateContentConfig,
    ModelContent,
    UserContent,
)
from pydantic import PrivateAttr

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigurationError
from gest.service.other.llm.provider.enums.googleai_api_error_status_enum import (
    GoogleAIAPIErrorStatusEnum,
)

from .base_llm_provider import (
    BaseLLMProvider,
    LLMExpectedResultType,
    LLMProviderEnum,
)
from .exception.provider_request_exceptions import (
    BaseLLMProviderRequestException,
    BaseLLMProviderResourceExhaustedRequestError,
    BaseRateLimitError,
)
from .exception.provider_response_exceptions import (
    ProviderJSONDecodeError,
    ProviderValueError,
)


class GoogleAILLMProvider(BaseLLMProvider):
    """GoogleAI LLM provider implementation."""

    _llm_provider: ClassVar[LLMProviderEnum] = LLMProviderEnum.GOOGLE_AI
    _conversation_history: List[ContentUnion] = []

    # Keep track of the current available API key used for the provider
    _current_key: ClassVar[Optional[str]] = None

    _model: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _backup_keys: List[str] = PrivateAttr()
    _attempted_backup_keys: List[str] = PrivateAttr(default_factory=list)
    _client: Client = PrivateAttr()

    def __init__(self):
        super().__init__()

        self._assign_configs(configs=self._get_configs(llm_provider=self._llm_provider))

        self._client = Client(api_key=self._api_key)

    def _chat_implementation(
        self,
        user_prompt: str,
        expected_result_type: LLMExpectedResultType,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Append the user prompt to the conversation history
        self._conversation_history.append(
            UserContent(
                parts=user_prompt,
            )
        )

        # Send the entire conversation history to the API
        try:
            completion = self._client.models.generate_content(
                model=self._model,
                contents=self._conversation_history,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
        except ServerError as exc:
            if exc.code == HTTPStatus.SERVICE_UNAVAILABLE:
                logging.error(
                    "The model is overloaded for GoogleAI LLM provider. Please try again later."
                )
                raise BaseRateLimitError(exc) from exc

            if exc.code == HTTPStatus.INTERNAL_SERVER_ERROR:
                logging.error(
                    "An internal server error occurred for GoogleAI LLM provider. Please try again later."
                )
                raise BaseRateLimitError(exc) from exc

            logging.error(
                f"An error occurred while generating content with GoogleAI LLM provider: {exc}"
            )
            raise
        except APIError as exc:
            if exc.code == HTTPStatus.TOO_MANY_REQUESTS:
                if exc.status == GoogleAIAPIErrorStatusEnum.RESOURCE_EXHAUSTED:
                    # Record this key as attempted
                    self._attempted_backup_keys.append(self._api_key)

                    # Get remaining available backup keys
                    available_backup_keys = [
                        k
                        for k in self._backup_keys
                        if k not in self._attempted_backup_keys
                    ]

                    # Rotate to the next backup key
                    if not available_backup_keys:
                        raise BaseLLMProviderRequestException(
                            "All GoogleAI API keys have been exhausted. "
                            "Please provide additional API keys or upgrade your plan."
                        )

                    # Pop the next key
                    next_key = available_backup_keys[0]
                    # Remove it from the list of backup keys
                    self._backup_keys.remove(next_key)

                    # Update the shared and instance key
                    GoogleAILLMProvider._current_key = next_key
                    self._api_key = next_key

                    # Reinit client with the new API key
                    self._client = Client(api_key=self._api_key)

                    logging.info(
                        f"Rotated GoogleAI API key due to {exc.code} {exc.status}. "
                        f"Using next ('{self._api_key}') backup key."
                    )

                    raise BaseLLMProviderResourceExhaustedRequestError(
                        "Resource exhausted on current API key. "
                        "Switched to a backup key."
                    )

                logging.error("Rate limit exceeded for GoogleAI LLM provider.")
                raise BaseRateLimitError(exc) from exc

            logging.error(
                f"An error occurred while generating content with GoogleAI LLM provider: {exc}"
            )
            raise

        # Extract and append the assistant's response to the conversation history
        response = completion.text

        if not isinstance(response, str):
            logging.error("The response from the LLM is not a string.")
            raise ProviderValueError(
                "Response was of type 'None'. Expected a response of type 'str'."
            )

        # Append the assistant's response to the conversation history
        self._conversation_history.append(
            ModelContent(
                parts=response,
            )
        )

        try:
            # GoogleAI LLM provider returns response in markdown format. We need to extract the JSON part.
            match = re.search(
                r"^\s*(?:```(?:json)?\s*([\s\S]*?)\s*```|([\s\S]+))\s*$", response, re.I
            )

            if match:
                content = (match.group(1) or match.group(2)).strip()

                if expected_result_type == LLMExpectedResultType.STRING:
                    return content

                if expected_result_type == LLMExpectedResultType.JSON:
                    return json.loads(content)

            # Raise exception
            logging.error("The response from the LLM is not a valid one.")
            raise ProviderValueError(
                f"The response is not valid. A '{expected_result_type}' was expected."
            )

        except JSONDecodeError as exc:
            logging.error(
                f"Failed to decode JSON from the response. Response: '{response}'"
            )
            raise ProviderJSONDecodeError(
                f"The response is not a valid JSON string. JSONDecodeError: '{exc}'"
            ) from exc

    def _assign_configs(self, configs: AttrDict) -> None:
        """Get the default GoogleAI LLM provider configs from the configuration."""

        self._model = self._assign_model_config(configs)

        # Keep track of the current API key used for the provider
        if GoogleAILLMProvider._current_key is None:
            self._api_key = self._assign_api_key_config(configs)
            GoogleAILLMProvider._current_key = self._api_key
        else:
            self._api_key = GoogleAILLMProvider._current_key

        all_backups = self._assign_backup_keys_config(configs)
        # Explicitly remove the current API key from the backup keys
        self._backup_keys = [k for k in all_backups if k != self._api_key]

    def _assign_model_config(self, configs: AttrDict) -> str:
        """Assign the model configuration for GoogleAI LLM provider."""

        model = configs.model

        if not isinstance(model, str):
            raise ConfigurationError("Provide a model name for GoogleAI LLM provider.")

        return model

    def _assign_api_key_config(self, configs: AttrDict) -> str:
        """Assign the API key configuration for GoogleAI LLM provider."""

        api_key = configs.api_key

        if not isinstance(api_key, str):
            raise ConfigurationError("Provide an API key for GoogleAI LLM provider.")

        return api_key

    def _assign_backup_keys_config(self, configs: AttrDict) -> List[str]:
        """Assign the backup API keys configuration for GoogleAI LLM provider."""

        backup_keys = configs.backup_keys

        if not isinstance(backup_keys, list):
            raise ConfigurationError(
                "Provide a list of backup API keys for GoogleAI LLM provider."
            )

        if not all(isinstance(key, str) for key in backup_keys):
            raise ConfigurationError(
                "Provide string elements in the list of backup API keys for GoogleAI LLM provider."
            )

        return backup_keys
