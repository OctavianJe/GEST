import json
import logging
import re
from json import JSONDecodeError
from typing import ClassVar, List, Optional, cast, get_args

from openai import OpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared import ChatModel
from pydantic import PrivateAttr

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


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    _llm_provider: ClassVar[LLMProviderEnum] = LLMProviderEnum.OPEN_AI
    _conversation_history: List[ChatCompletionMessageParam] = []

    _model: ChatModel = PrivateAttr()
    _api_key: str = PrivateAttr()
    _client: OpenAI = PrivateAttr()

    def __init__(self):
        super().__init__()

        self._assign_configs(configs=self._get_configs(llm_provider=self._llm_provider))
        self._client = OpenAI(api_key=self._api_key)

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
                    ChatCompletionSystemMessageParam(
                        content=system_prompt,
                        role="system",
                    )
                )

        # Append the user prompt to the conversation history
        self._conversation_history.append(
            ChatCompletionUserMessageParam(
                content=user_prompt,
                role="user",
            )
        )

        # Send the entire conversation history to the API
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=self._conversation_history,
            )
        except RateLimitError as exc:
            logging.error("Rate limit exceeded for OpenAI LLM provider.")
            raise BaseRateLimitError(exc) from exc

        # Extract and append the assistant's response to the conversation history
        response = completion.choices[0].message.content

        if not isinstance(response, str):
            logging.error("The response from the LLM is not a string.")
            raise ProviderValueError(
                "Response was of type 'None'. Expected a response of type 'str'."
            )

        # Append the assistant's response to the conversation history
        self._conversation_history.append(
            ChatCompletionAssistantMessageParam(
                content=response,
                role="assistant",
            )
        )

        try:
            # OpenAI LLM provider returns response in markdown format. We need to extract the JSON part.
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
        """Get the default OpenAI LLM provider configs from the configuration."""

        self._model = self._assign_model_config(configs)
        self._api_key = self._assign_api_key_config(configs)

    def _assign_model_config(self, configs: AttrDict) -> ChatModel:
        """Assign the model configuration for OpenAI LLM provider."""

        model = configs.model

        if not isinstance(model, str):
            raise ConfigurationError("Provide a model name for OpenAI LLM provider.")

        available_models = get_args(ChatModel)
        if model not in available_models:
            raise ConfigurationError(
                "Invalid model name for OpenAI LLM provider. "
                f"List of all available models: {available_models}"
            )

        return cast(ChatModel, model)

    def _assign_api_key_config(self, configs: AttrDict) -> str:
        """Assign the API key configuration for OpenAI LLM provider."""

        api_key = configs.api_key

        if not isinstance(api_key, str):
            raise ConfigurationError("Provide an API key for OpenAI LLM provider.")

        return api_key
