"""
Transformer Architecture & Scalable Attention Library
Equipped with FlashAttention / SDPA, Grouped-Query Attention (GQA),
Rotary Position Embeddings (RoPE), Dynamic KV-Cache, and Causal Transformer LM.
"""

from transformer.config import TransformerConfig
from transformer.attention.rope import RotaryEmbedding, apply_rotary_pos_emb
from transformer.attention.kv_cache import KVCache
from transformer.attention.causal_attention import CausalSelfAttention, repeat_kv
from transformer.modules.norm import RMSNorm
from transformer.modules.ffn import SwiGLU
from transformer.modules.block import TransformerBlock
from transformer.models.causal_lm import CausalTransformerLM

__version__ = "1.0.0"

__all__ = [
    "TransformerConfig",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "KVCache",
    "CausalSelfAttention",
    "repeat_kv",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
    "CausalTransformerLM",
]
