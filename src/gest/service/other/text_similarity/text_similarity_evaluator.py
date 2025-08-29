import threading
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError


class TextSimilarityEvaluator:
    """Singleton evaluator for text similarity."""

    _instance: "Optional[TextSimilarityEvaluator]" = None
    _lock = threading.Lock()

    def __new__(cls) -> "TextSimilarityEvaluator":
        """Singleton pattern to ensure only one instance of TextSimilarityEvaluator exists."""

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        """Initialize the text similarity evaluator configuration if not already initialized."""

        if getattr(self, "_initialized", False):
            return

        # Load configs
        self._configs = self._get_configs()

        # Prepare cache directory
        cache_dir = self._configs.cache_dir
        if not isinstance(cache_dir, str):
            raise ConfigurationError(
                "Invalid cache directory for text similarity embedding."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model_name = self._configs.model_name
        if not isinstance(model_name, str):
            raise ConfigurationError(
                "Invalid model name for text similarity embedding."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))

        self._initialized = True

    def compute_text_similarity(self, original: str, recomputed: str) -> float:
        """Compute the cosine similarity between two texts using a pre-trained SentenceTransformer model."""

        embeddings = self.model.encode([original, recomputed])

        return cosine_similarity(
            X=np.array([embeddings[0]]), Y=np.array([embeddings[1]])
        )[0][0]

    def _get_configs(self) -> AttrDict:
        """Get the default text similarity model configs from the configuration."""

        model_configs = ConfigLoader().get("text_similarity.models")

        if isinstance(model_configs, AttrDict):
            return model_configs

        raise ConfigurationError(
            "Invalid configuration format for text similarity model."
        )
