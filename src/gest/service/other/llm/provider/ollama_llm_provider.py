import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from json import JSONDecodeError
from typing import ClassVar, List, Optional, Set

import requests
from bs4 import BeautifulSoup, Tag
from ollama import Client, Message
from pydantic import PrivateAttr
from requests import HTTPError

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigurationError

from .base_llm_provider import (
    BaseLLMProvider,
    LLMExpectedResultType,
    LLMProviderEnum,
)
from .exception.provider_request_exceptions import (
    BaseRateLimitError,
)
from .exception.provider_response_exceptions import (
    ProviderJSONDecodeError,
    ProviderValueError,
)


@dataclass(frozen=True, slots=True)
class OllamaModel:
    """Immutable value object for a base model in the Ollama library."""

    name: str
    url: str


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    _llm_provider: ClassVar[LLMProviderEnum] = LLMProviderEnum.OLLAMA
    _conversation_history: List[Message] = []

    _model: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _think: bool = PrivateAttr()
    _client: Client = PrivateAttr()

    def __init__(self):
        super().__init__()

        self._assign_configs(configs=self._get_configs(llm_provider=self._llm_provider))
        self._client = Client(host=self._base_url)

    def _chat_implementation(
        self,
        user_prompt: str,
        expected_result_type: LLMExpectedResultType,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Use the conversation history if available, otherwise create a new one.
        # If a system prompt is provided and this is a new conversation, add it as the first message.
        if not self._conversation_history:
            if system_prompt:
                self._conversation_history.append(
                    Message(
                        role="system",
                        content=system_prompt,
                    )
                )

        # Append the user prompt to the conversation history
        self._conversation_history.append(
            Message(
                role="user",
                content=user_prompt,
            )
        )

        # Send the entire conversation history to the API
        try:
            chat_response = self._client.chat(
                model=self._model,
                messages=self._conversation_history,
                think=self._think,
            )
        except HTTPError as exc:
            if (
                exc.response is not None
                and exc.response.status_code == HTTPStatus.TOO_MANY_REQUESTS
            ):
                logging.error("Rate limit exceeded for Ollama LLM provider. ")
                raise BaseRateLimitError(exc) from exc

            logging.error(
                f"An error occurred while generating content with Ollama LLM provider: {exc}"
            )
            raise

        # Extract and append the assistant's response to the conversation history
        response = chat_response.message.content

        if not isinstance(response, str):
            logging.error("The response from the LLM is not a string.")
            raise ProviderValueError(
                "Response was of type 'None'. Expected a response of type 'str'."
            )

        # Append the assistant's response to the conversation history
        self._conversation_history.append(chat_response.message)

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
        """Get the default Ollama LLM provider configs from the configuration."""

        self._base_url = self._assign_base_url_config(configs)
        self._model = self._assign_model_config(configs)
        self._think = self._assign_think_config(configs)

    def _assign_model_config(self, configs: AttrDict) -> str:
        """Assign the model configuration for Ollama LLM provider."""

        model = configs.model

        if not isinstance(model, str):
            raise ConfigurationError("Provide a model name for Ollama LLM provider.")

        local_models = self._list_local_models()
        if model in local_models:
            return model

        if ":" not in model:
            raise ConfigurationError(
                "Provide the model in '<base>:<tag>' format (e.g. 'llama3:8b' or 'phi3:latest')"
            )

        base, _, tag = model.partition(":")

        available_model_names = {m.name: m for m in self._get_ollama_available_models()}
        if base not in available_model_names:
            raise ConfigurationError(
                f"Invalid model name '{base}' for Ollama LLM provider. "
                f"List of all available models: {', '.join(sorted(available_model_names.keys()))}"
            )

        available_model_tags = self._get_ollama_available_model_tags(
            model=available_model_names[base]
        )
        if tag not in available_model_tags:
            raise ConfigurationError(
                f"Invalid model tag '{tag}' for Ollama LLM provider. "
                f"List of all available tags for '{base}' model: {available_model_tags}"
            )

        return model

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_ollama_available_models(
        limit: int = 200,
        timeout: int = 30,
        url: str = "https://ollamadb.dev/api/v1/models",
    ) -> Set[OllamaModel]:
        """Retrieve a set of available models from the Ollama API."""

        try:
            resp = requests.get(url, params={"limit": limit}, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise ConfigurationError(
                f"Could not retrieve models from {url}. Exception: '{exc}'"
            ) from exc
        except Exception as exc:
            raise ConfigurationError(
                f"An unexpected error occurred. Exception: '{exc}'."
            ) from exc

        return {
            OllamaModel(name=m["model_identifier"], url=m["url"])
            for m in payload.get("models", [])
        }

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_ollama_available_model_tags(
        model: OllamaModel, timeout: int = 30
    ) -> Set[str]:
        """Retrieve a set of available tags for a model from the Ollama API. Also add 'latest' tag explicitly."""

        try:
            html = requests.get(model.url, timeout=timeout).text
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not retrieve model page {model.url}") from exc

        soup = BeautifulSoup(html, "html.parser")
        pattern: re.Pattern[str] = re.compile(
            rf"^/library/{re.escape(model.name)}:(?P<tag>[^\"/]+)$"
        )

        tags: Set[str] = set()
        tags.add("latest")

        for node in soup.find_all("a", href=True):
            if not isinstance(node, Tag):
                continue

            href = node.get("href")

            if isinstance(href, str) and (m := pattern.match(href)):
                tags.add(m.group("tag"))

        return tags

    def _list_local_models(self, timeout: int = 30) -> Set[str]:
        """
        Return locally available model names in the form 'base:tag' from the running Ollama.
        """
        try:
            base = (self._base_url or "").rstrip("/")
            if not base:
                return set()

            resp = requests.get(f"{base}/api/tags", timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            return {
                str(m.get("name", "")).partition(":")[0]
                for m in payload.get("models", [])
                if isinstance(m, dict) and m.get("name")
            }
        except Exception as exc:
            logging.warning(
                f"Could not list local Ollama models at {self._base_url}: {exc}"
            )
            return set()

    def _assign_base_url_config(self, configs: AttrDict) -> str:
        """Assign the base url configuration for Ollama LLM provider."""

        base_url = configs.base_url

        if not isinstance(base_url, str):
            raise ConfigurationError("Provide a base url for Ollama LLM provider.")

        return base_url

    def _assign_think_config(self, configs: AttrDict) -> bool:
        """Assign the think configuration for Ollama LLM provider."""

        think = configs.think

        if not isinstance(think, bool):
            raise ConfigurationError(
                "Provide a think property for Ollama LLM provider."
            )

        return think
