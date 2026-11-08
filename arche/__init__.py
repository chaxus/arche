"""arche — modern dense decoder primitives (Llama 3.2 lineage)."""

from arche.config import Llama32Config, llama32_1b, llama32_3b
from arche.embedding import TokenEmbedding

__all__ = [
    "Llama32Config",
    "TokenEmbedding",
    "llama32_1b",
    "llama32_3b",
]

__version__ = "0.1.0"
