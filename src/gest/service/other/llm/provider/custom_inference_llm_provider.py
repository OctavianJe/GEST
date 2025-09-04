import logging
from typing import Any, ClassVar, List, Optional

from pydantic import PrivateAttr

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigurationError

from .base_llm_provider import (
    BaseLLMProvider,
    LLMExpectedResultType,
    LLMProviderEnum,
)


class CustomInferenceLLMProvider(BaseLLMProvider):
    """
    Custom Inference LLM provider implementation.
    Used to be overriden in Colab at inference.
    """

    _llm_provider: ClassVar[LLMProviderEnum] = LLMProviderEnum.CUSTOM_INFERENCE
    _conversation_history: List[Any] = []

    _model_name: str = PrivateAttr()

    def __init__(self):
        super().__init__()

        self._assign_configs(configs=self._get_configs(llm_provider=self._llm_provider))

    def _chat_implementation(
        self,
        user_prompt: str,
        expected_result_type: LLMExpectedResultType,
        system_prompt: Optional[str] = None,
    ) -> str:
        logging.error(
            "Method not overriden by inference runtime. Will raise NotImplementedError"
        )

        raise NotImplementedError(
            "Inference environment should implement '_chat_implementation()' method."
        )

    def _assign_configs(self, configs: AttrDict) -> None:
        """Get the default CustomInference LLM provider configs from the configuration."""

        self._model_name = self._assign_model_config(configs)

    def _assign_model_config(self, configs: AttrDict) -> str:
        """Assign the model configuration for CustomInference LLM provider."""

        model_name = configs.model

        if not isinstance(model_name, str):
            raise ConfigurationError(
                "Provide a model name for CustomInference LLM provider."
            )

        return model_name
