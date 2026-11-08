"""arche — modern dense decoder primitives (Llama 3.2 lineage)."""

from arche.attention import CausalSelfAttention
from arche.block import TransformerBlock
from arche.config import Llama32Config, llama32_1b, llama32_3b
from arche.embedding import TokenEmbedding
from arche.mlp import SwiGLU
from arche.model import Llama
from arche.norms import RMSNorm

__all__ = [
    "CausalSelfAttention",
    "Llama",
    "Llama32Config",
    "RMSNorm",
    "SwiGLU",
    "TokenEmbedding",
    "TransformerBlock",
    "llama32_1b",
    "llama32_3b",
]

__version__ = "0.1.0"
