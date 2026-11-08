"""arche — modern dense decoder primitives (Llama 3.2 lineage)."""

from arche.attention import CausalSelfAttention
from arche.block import TransformerBlock
from arche.cache import KVCache
from arche.config import Llama32Config, llama32_1b, llama32_3b
from arche.embedding import TokenEmbedding
from arche.generate import generate
from arche.load_weights import DEFAULT_MODEL_ID, from_pretrained, try_from_pretrained
from arche.mlp import SwiGLU
from arche.mla import MLACache, MultiHeadLatentAttention
from arche.moe import MoESwiGLU
from arche.model import Llama
from arche.norms import RMSNorm

__all__ = [
    "CausalSelfAttention",
    "KVCache",
    "DEFAULT_MODEL_ID",
    "Llama",
    "MLACache",
    "MultiHeadLatentAttention",
    "Llama32Config",
    "MoESwiGLU",
    "RMSNorm",
    "SwiGLU",
    "TokenEmbedding",
    "TransformerBlock",
    "from_pretrained",
    "generate",
    "llama32_1b",
    "llama32_3b",
    "try_from_pretrained",
]

__version__ = "0.1.0"
