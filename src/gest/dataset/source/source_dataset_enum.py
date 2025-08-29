from enum import Enum


class SourceDatasetEnum(str, Enum):
    """Names of the *original* datasets used as sources for GEST."""

    ACTIVITY_NET_CAPTIONS = "ActivityNet Captions"
