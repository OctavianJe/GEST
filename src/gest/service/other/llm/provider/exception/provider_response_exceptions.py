class BaseLLMProviderResponseError(Exception):
    """Base class for LLM provider errors. Any LLM provider-specific response error should inherit from this class."""


class ProviderValueError(BaseLLMProviderResponseError):
    """Raised when the LLM provider returns an invalid value."""


class ProviderJSONDecodeError(BaseLLMProviderResponseError):
    """Raised when the LLM provider returns an invalid JSON response."""
