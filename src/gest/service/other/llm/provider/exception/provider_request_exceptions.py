class BaseLLMProviderRequestError(Exception):
    """Base class for LLM provider errors. Any LLM provider-specific request error should inherit from this class."""


class BaseRateLimitError(BaseLLMProviderRequestError):
    """Raised when an LLM provider returns a 429 TOO_MANY_REQUESTS error."""


class BaseLLMProviderResourceExhaustedRequestError(Exception):
    """Raised when an LLM provider returns a resource exhausted error."""


class BaseLLMProviderRequestException(Exception):
    """Base class for LLM provider errors which should be raised and terminate the application main process."""
