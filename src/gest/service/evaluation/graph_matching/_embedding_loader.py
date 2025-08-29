import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError

from ._simple_vectors import SimpleVectors
from .embedding_type_enum import EmbeddingType


class EmbeddingLoader:
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)

    def load_vectors(self, model_name: EmbeddingType) -> SimpleVectors:
        """Load and return a read-only *word-embedding* model."""

        raw = self._download(model_name)
        txt = self._extract(raw, model_name)

        kv: dict[str, np.ndarray] = {}
        is_binary: bool = txt.suffix == ".bin"

        if is_binary:
            with open(txt, "rb") as f:
                header = f.readline()
                vocab_size, dim = map(int, header.split())
                for _ in range(vocab_size):
                    word_bytes = bytearray()
                    while True:
                        ch = f.read(1)
                        if ch == b" ":
                            break
                        word_bytes.extend(ch)
                    word = word_bytes.decode("latin-1")
                    vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
                    kv[word] = vec
                    f.read(1)
        else:
            with open(txt, "r", encoding="utf8") as f:
                for line in f:
                    parts = line.rstrip().split(" ")
                    kv[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

        return SimpleVectors(kv)

    def _download(self, model_type: EmbeddingType) -> Path:
        """Download the embedding file if not already cached."""

        url = model_type.url
        file_name = model_type.file_name
        dest = self._cache / file_name

        if not dest.exists():
            print(f"Downloading {model_type.model_name} …")
            with urllib.request.urlopen(url) as r, open(dest, "wb") as w:
                shutil.copyfileobj(r, w)

        return dest

    def _extract(self, file_path: Path, model_type: EmbeddingType) -> Path:
        """Extract the downloaded file to a usable format."""

        if file_path.suffix == ".zip":
            with zipfile.ZipFile(file_path) as zf:
                wanted = f"glove.6B.{model_type.dim}d.txt"
                zf.extract(wanted, self._cache)
                return self._cache / wanted

        if file_path.suffix == ".gz":
            out = self._cache / file_path.stem
            if not out.exists():
                with gzip.open(file_path, "rb") as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return out

        if file_path.suffix in {".bin", ".txt"}:
            return file_path

        raise RuntimeError(f"Don't know how to extract {file_path}")

    def _get_default_cache_dir(self) -> Path:
        """Get the default cache directory from the configuration."""

        cache_dir = ConfigLoader().get("graph_matching.embeddings.cache_dir")

        if not isinstance(cache_dir, str):
            raise ConfigurationError(
                "Invalid cache directory in configuration. Expected a string path."
            )

        return Path(cache_dir)
