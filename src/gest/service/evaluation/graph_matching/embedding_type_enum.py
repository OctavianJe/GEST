from enum import Enum

from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError


class EmbeddingType(Enum):
    GLOVE50 = "glove50"
    GLOVE300 = "glove300"
    W2V_GOOGLE = "w2v_google"

    def __init__(self, key: str):
        embeddings = ConfigLoader().get("graph_matching.embeddings")

        if not isinstance(embeddings, dict):
            raise ConfigurationError("Embeddings configuration must be a dictionary.")

        conf = embeddings.get(key)
        if not conf:
            raise ConfigurationError(f"Config missing for embedding type: {key}")

        try:
            self.model_name = conf["model_name"]
            self.file_name = conf["file_name"]
            self.url = conf["url"]
            self.dim = int(conf["dim"])
        except Exception as e:
            raise ConfigurationError(
                f"Error processing config for embedding type {key}: {conf}. Details: {e}"
            )

        if not all([self.model_name, self.file_name, self.url, self.dim]):
            raise ConfigurationError(
                f"Incomplete config for embedding type {key}: {conf}"
            )
