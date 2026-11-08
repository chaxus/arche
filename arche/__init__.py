"""arche — modern dense decoder primitives (Llama 3.2 lineage)."""

from arche.align_hf import AlignReport, compare_logits_to_hf, hf_weights_reachable
from arche.attention import CausalSelfAttention
from arche.batching import DecodeRequest, generate_continuous
from arche.block import TransformerBlock
from arche.cache import KVCache
from arche.chat import apply_chat_template, encode_chat
from arche.checkpoint import load_checkpoint, save_checkpoint
from arche.config import Llama32Config, llama32_1b, llama32_3b
from arche.cuda_graphs import capture_module_step, cuda_available, generate_with_optional_graph
from arche.device import describe_device, pick_device
from arche.embedding import TokenEmbedding
from arche.generate import generate
from arche.load_deepseek_mla import load_deepseek_mla_shards, load_deepseek_shaped_state
from arche.load_mixtral import load_mixtral_shards_into_moe, load_mixtral_shaped_state, mixtral_expert_to_moe_key
from arche.load_weights import DEFAULT_MODEL_ID, from_pretrained, try_from_pretrained
from arche.mlp import SwiGLU
from arche.mla import MLACache, MultiHeadLatentAttention
from arche.moe import MoESwiGLU
from arche.model import Llama
from arche.norms import RMSNorm
from arche.paged_kv import BlockTable, PagedKVPool, append_kv, gather_kv
from arche.quantize import Int8Linear, quantize_llama_int8

__all__ = [
    "AlignReport",
    "BlockTable",
    "CausalSelfAttention",
    "DecodeRequest",
    "KVCache",
    "DEFAULT_MODEL_ID",
    "Int8Linear",
    "Llama",
    "MLACache",
    "MultiHeadLatentAttention",
    "Llama32Config",
    "MoESwiGLU",
    "PagedKVPool",
    "RMSNorm",
    "SwiGLU",
    "TokenEmbedding",
    "TransformerBlock",
    "append_kv",
    "apply_chat_template",
    "capture_module_step",
    "compare_logits_to_hf",
    "cuda_available",
    "describe_device",
    "encode_chat",
    "from_pretrained",
    "gather_kv",
    "generate",
    "generate_continuous",
    "generate_with_optional_graph",
    "hf_weights_reachable",
    "llama32_1b",
    "llama32_3b",
    "load_checkpoint",
    "load_deepseek_mla_shards",
    "load_deepseek_shaped_state",
    "load_mixtral_shards_into_moe",
    "load_mixtral_shaped_state",
    "mixtral_expert_to_moe_key",
    "pick_device",
    "quantize_llama_int8",
    "save_checkpoint",
    "try_from_pretrained",
]

__version__ = "0.1.0"
