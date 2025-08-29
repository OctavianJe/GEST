import logging
from functools import wraps
from typing import Callable, TypeVar, cast

T = TypeVar("T")


def requires_enabled(
    *,
    attr: str = "_enabled",
    expected: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory that turns a method into a no-op unless a flag matches.

    Parameters
    ----------
    attr     - attribute to inspect on ``self`` (default ``"_enabled"``)
    expected - value the attribute must equal for the call to proceed
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            value = getattr(self, attr, None)
            if value != expected:
                logging.info(
                    "%s skipped (%s=%r, expected %r)",
                    func.__qualname__,
                    attr,
                    value,
                    expected,
                )
                return cast(
                    T,
                    {
                        "skipped": True,
                        "reason": f"{attr}={value!r} (expected {expected!r})",
                    },
                )
            logging.info("%s executed (%s=%r)", func.__qualname__, attr, value)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
