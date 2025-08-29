from typing import Union


class AttrDict(dict):
    """A dictionary that allows attribute-style access to its keys."""

    def __getattr__(self, item) -> Union["AttrDict", str, None]:
        """Get an item using dot notation, returning an AttrDict if the item is a dict."""
        val = self.get(item)
        if isinstance(val, dict):
            return AttrDict(val)
        return val
