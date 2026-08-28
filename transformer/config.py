"""
Transformer Library - Configuration
Modern, scalable Transformer configuration supporting FlashAttention, GQA/MQA, RoPE, and SwiGLU.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class TransformerConfig:
    """
    Configuration for modern scalable Transformer models.
    
    Attributes:
        d_model (int): Model hidden dimension. Default: 512.
        n_heads (int): Number of query attention heads. Default: 8.
        n_kv_heads (Optional[int]): Number of key/value heads for GQA/MQA.
                                   If None or equal to n_heads -> Standard Multi-Head Attention (MHA).
                                   If 1 -> Multi-Query Attention (MQA).
                                   If 1 < n_kv_heads < n_heads -> Grouped-Query Attention (GQA).
        head_dim (Optional[int]): Dimension per head. If None, computed as d_model // n_heads.
        n_layers (int): Number of transformer decoder blocks. Default: 6.
        vocab_size (int): Vocabulary size. Default: 32000.
        max_seq_len (int): Maximum sequence context length. Default: 4096.
        mlp_ratio (float): Hidden expansion ratio for FFN/SwiGLU. Default: 4.0.
        norm_eps (float): Epsilon for RMSNorm. Default: 1e-6.
        dropout (float): Dropout probability. Default: 0.0.
        rope_theta (float): Base frequency parameter for Rotary Embeddings (RoPE). Default: 10000.0.
        use_rope (bool): Whether to apply Rotary Positional Embeddings. Default: True.
        use_flash_attn (bool): Whether to use FlashAttention / SDPA kernel acceleration. Default: True.
        bias (bool): Whether linear projections include bias (False is standard in modern LLMs). Default: False.
        tie_word_embeddings (bool): Whether to tie lm_head weights with token embeddings. Default: True.
    """
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    n_layers: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 4096
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_flash_attn: bool = True
    bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (for GQA)"

    @property
    def num_queries_per_kv(self) -> int:
        """Number of query heads sharing a single KV head group (GQA group size)."""
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TransformerConfig":
        return cls.from_dict(json.loads(json_str))
