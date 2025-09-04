from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, List, Optional

from pydantic import BaseModel, PrivateAttr
from retry import retry

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError

from .exception.provider_request_exceptions import (
    BaseLLMProviderResourceExhaustedRequestError,
    BaseRateLimitError,
)


class LLMProviderEnum(str, Enum):
    """Enumeration for LLM providers."""

    OLLAMA = "ollama"
    OPEN_AI = "openai"
    GOOGLE_AI = "googleai"
    CUSTOM_INFERENCE = "custom_inference"


class LLMExpectedResultType(str, Enum):
    """Enumeration for LLM expected result."""

    JSON = "json"
    STRING = "string"


class BaseLLMProvider(BaseModel, ABC):
    """Base class for LLM providers."""

    _llm_provider: ClassVar[LLMProviderEnum]
    _conversation_history: List[Any] = PrivateAttr(default_factory=list)

    @retry(BaseRateLimitError, delay=1, backoff=2)
    @retry(BaseLLMProviderResourceExhaustedRequestError, delay=1)
    def chat(
        self,
        user_prompt: str,
        expected_result_type: LLMExpectedResultType,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Exponential-backoff wrapper that delegates to the provider-specific implementation."""

        return self._chat_implementation(
            user_prompt, expected_result_type, system_prompt
        )

    @abstractmethod
    def _chat_implementation(
        self,
        user_prompt: str,
        expected_result_type: LLMExpectedResultType,
        system_prompt: Optional[str] = None,
    ) -> str: ...

    @staticmethod
    def _get_configs(llm_provider: LLMProviderEnum) -> AttrDict:
        """Get the default text similarity embedding configs from the configuration."""

        llm_configs = ConfigLoader().get(f"llm.{llm_provider.value}")

        if isinstance(llm_configs, AttrDict):
            return llm_configs

        raise ConfigurationError("Invalid configuration format for LLM provider.")
