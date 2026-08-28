"""
Transformer Attention & Causal LM - Comprehensive Unit Tests
"""

import pytest
import torch
import torch.nn as nn

from transformer.config import TransformerConfig
from transformer.attention.rope import RotaryEmbedding, apply_rotary_pos_emb
from transformer.attention.kv_cache import KVCache
from transformer.attention.causal_attention import CausalSelfAttention
from transformer.modules.norm import RMSNorm
from transformer.modules.ffn import SwiGLU
from transformer.modules.block import TransformerBlock
from transformer.models.causal_lm import CausalTransformerLM


def test_rope_embedding():
    dim = 32
    seq_len = 16
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    
    q = torch.randn(2, 4, seq_len, dim)
    k = torch.randn(2, 4, seq_len, dim)
    
    cos, sin = rope(q, seq_len=seq_len)
    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q, q_rot)


def test_kv_cache():
    cache = KVCache(max_seq_len=32)
    
    k1 = torch.randn(2, 2, 8, 16)
    v1 = torch.randn(2, 2, 8, 16)
    full_k, full_v = cache.update(k1, v1)
    assert full_k.shape == (2, 2, 8, 16)
    assert cache.seq_len == 8
    
    k2 = torch.randn(2, 2, 1, 16)
    v2 = torch.randn(2, 2, 1, 16)
    full_k, full_v = cache.update(k2, v2)
    assert full_k.shape == (2, 2, 9, 16)
    assert cache.seq_len == 9


def test_mha_and_gqa_attention():
    # 1. Standard MHA (n_heads == n_kv_heads == 4)
    cfg_mha = TransformerConfig(d_model=64, n_heads=4, n_kv_heads=4, use_flash_attn=True)
    attn_mha = CausalSelfAttention(cfg_mha)
    x = torch.randn(2, 12, 64)
    out_mha, _ = attn_mha(x)
    assert out_mha.shape == (2, 12, 64)
    
    # 2. Grouped-Query Attention (n_heads=8, n_kv_heads=2)
    cfg_gqa = TransformerConfig(d_model=64, n_heads=8, n_kv_heads=2, use_flash_attn=True)
    attn_gqa = CausalSelfAttention(cfg_gqa)
    out_gqa, _ = attn_gqa(x)
    assert out_gqa.shape == (2, 12, 64)


def test_causal_attention_kv_cache_equivalence():
    config = TransformerConfig(d_model=64, n_heads=4, n_kv_heads=2, use_rope=True)
    attn = CausalSelfAttention(config)
    attn.eval()
    
    x = torch.randn(2, 8, 64)
    
    # Full sequence forward
    with torch.no_grad():
        out_full, _ = attn(x)
        
    # Step-by-step with KV cache
    with torch.no_grad():
        cache = KVCache()
        step_outs = []
        for t in range(8):
            x_t = x[:, t:t+1, :]
            out_t, cache = attn(x_t, kv_cache=cache, start_pos=t)
            step_outs.append(out_t)
        out_step = torch.cat(step_outs, dim=1)
        
    assert torch.allclose(out_full, out_step, atol=1e-5)


def test_transformer_block():
    config = TransformerConfig(d_model=64, n_heads=4, n_kv_heads=2)
    block = TransformerBlock(config)
    
    x = torch.randn(2, 10, 64)
    out, cache = block(x)
    assert out.shape == (2, 10, 64)


def test_causal_lm_forward_and_loss():
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=3,
        vocab_size=100,
        max_seq_len=64
    )
    model = CausalTransformerLM(config)
    
    input_ids = torch.randint(0, 100, (2, 12))
    targets = torch.randint(0, 100, (2, 12))
    
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 12, 100)
    assert loss is not None
    assert loss.item() > 0.0
    
    # Backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_causal_lm_generate():
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        vocab_size=50,
        max_seq_len=64
    )
    model = CausalTransformerLM(config)
    model.eval()
    
    prompt = torch.randint(0, 50, (2, 6))
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert generated.shape == (2, 14)
    assert (generated[:, :6] == prompt).all()
