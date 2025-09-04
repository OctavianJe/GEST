import logging

from gest.service.other.llm.provider.custom_inference_llm_provider import (
    CustomInferenceLLMProvider,
)
from gest.service.other.llm.provider.googleai_llm_provider import GoogleAILLMProvider
from gest.service.other.llm.provider.ollama_llm_provider import OllamaLLMProvider
from gest.service.other.llm.provider.openai_llm_provider import OpenAILLMProvider

from .base_llm_provider import BaseLLMProvider, LLMProviderEnum


class LLMProviderFactory:
    """Factory class for creating LLM provider instances."""

    @staticmethod
    def create_provider(type: LLMProviderEnum) -> BaseLLMProvider:
        """Creates an instance of the specified LLM provider."""

        logging.info(f"Creating LLM provider of type: {type.value}")

        if type == LLMProviderEnum.OLLAMA:
            return OllamaLLMProvider()

        if type == LLMProviderEnum.OPEN_AI:
            return OpenAILLMProvider()

        if type == LLMProviderEnum.GOOGLE_AI:
            return GoogleAILLMProvider()

        if type == LLMProviderEnum.CUSTOM_INFERENCE:
            return CustomInferenceLLMProvider()

        raise NotImplementedError(
            f"Invalid LLM Provider '{type}' for provider factory."
            f"Available providers are {', '.join([provider.value for provider in LLMProviderEnum])}."
        )
