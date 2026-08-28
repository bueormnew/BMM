"""
BUEORM Delta Attention (BDA) - Configuration
Spec v2: Architectural and Hyperparameter Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BDAConfig:
    """
    Configuration for BUEORM Delta Attention (BDA) layers, blocks, and models.
    
    Attributes:
        d_model (int): Model embedding dimension (hidden dimension). Default: 256.
        n_heads (int): Number of attention heads H. Default: 4.
        d_k (int): Dimension of keys/queries per head. Default: 32.
        d_v (int): Dimension of values per head. Default: 32.
        rank_r (int): Bottleneck rank r for LRFG and DEM (r << d_k). Default: 8.
        ema_lambda (float): EMA decay factor lambda for ASN. Default: 0.01.
        eps (float): Small constant epsilon for numerical stability. Default: 1e-6.
        stability_margin (float): Target spectral norm margin for Stability Projection. Default: 1.0.
        chunk_size (int): Block / chunk size C for Chunk-Recurrent Parallel Training. Default: 16.
        n_layers (int): Total number of transformer blocks in the hybrid model. Default: 6.
        hybrid_interval (int): Frequency of Full Attention layers (e.g. every N-th layer is Full Attention). Default: 4.
        vocab_size (int): Vocabulary size for the language model. Default: 1000.
        max_seq_len (int): Maximum sequence length supported. Default: 2048.
        mlp_ratio (float): Expansion ratio for the feed-forward / SwiGLU network. Default: 4.0.
        dropout (float): Dropout probability. Default: 0.0.
        bias (bool): Whether linear layers include bias where applicable. Default: True.
        device: Optional device.
        dtype (str): Default precision ('float32', 'bfloat16', 'float16'). Default: 'float32'.
    """
    d_model: int = 256
    n_heads: int = 4
    d_k: int = 32
    d_v: int = 32
    rank_r: int = 8
    ema_lambda: float = 0.01
    eps: float = 1e-6
    stability_margin: float = 1.0
    chunk_size: int = 16
    n_layers: int = 6
    hybrid_interval: int = 4
    vocab_size: int = 1000
    max_seq_len: int = 2048
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = True
    device: Optional[str] = None
    dtype: str = "float32"

    def __post_init__(self):
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_k <= 0 or self.d_v <= 0:
            raise ValueError("Dimensions must be positive integers.")
