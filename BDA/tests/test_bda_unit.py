"""
BUEORM Delta Attention (BDA) - Comprehensive PyTest Unit Suite
Covers all modules, primitives, layers, blocks, hybrid model, and streaming generation.
"""

import pytest
import torch
import torch.nn as nn

from BDA.config import BDAConfig
from BDA.ops.state import MemoryState
from BDA.ops.gates import LRFG, DEM
from BDA.ops.normalization import ASN
from BDA.ops.stability import StabilityProjection
from BDA.layers.bda_step import bda_head_step
from BDA.layers.crpt import CRPTFunction, crpt_forward_native
from BDA.layers.bda_layer import BDALayer
from BDA.layers.attention import CausalFullAttention
from BDA.layers.block import RMSNorm, SwiGLU, BDABlock, HybridBlock
from BDA.models.hybrid_model import BDAHybridModel


# ---------------------------------------------------------------------------
# 1. OPS Tests: MemoryState, LRFG, DEM, ASN, StabilityProjection
# ---------------------------------------------------------------------------

def test_memory_state_primitives():
    B, H, d_v, d_k = 2, 4, 32, 16
    S = MemoryState.init_state(B, H, d_v, d_k)
    assert S.shape == (B, H, d_v, d_k)
    assert (S == 0.0).all()
    
    alpha = torch.rand(B, H, d_k)
    S_decayed = MemoryState.apply_decay(S, alpha)
    assert S_decayed.shape == (B, H, d_v, d_k)
    
    k_tilde = torch.randn(B, H, d_k)
    v = torch.randn(B, H, d_v)
    error = MemoryState.compute_error(S_decayed, k_tilde, v)
    assert error.shape == (B, H, d_v)
    # Since S is zero, error should be -v
    assert torch.allclose(error, -v, atol=1e-6)
    
    k_hat = torch.randn(B, H, d_k)
    beta = torch.rand(B, H)
    S_new = MemoryState.update_state(S_decayed, error, k_hat, beta)
    assert S_new.shape == (B, H, d_v, d_k)
    
    q = torch.randn(B, H, d_k)
    o = MemoryState.read_state(S_new, q)
    assert o.shape == (B, H, d_v)
    
    snorm = MemoryState.spectral_norm(S_new)
    assert snorm.shape == (B, H)
    assert (snorm >= 0.0).all()


def test_lrfg_and_dem_gates():
    B, T, d_model, H, d_k, rank_r = 2, 8, 64, 4, 16, 4
    x = torch.randn(B, T, d_model)
    
    lrfg = LRFG(d_model=d_model, n_heads=H, d_k=d_k, rank_r=rank_r)
    alpha, z = lrfg(x)
    assert alpha.shape == (B, T, H, d_k)
    assert z.shape == (B, T, rank_r)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()
    
    k_hat = torch.randn(B, T, H, d_k)
    dem = DEM(n_heads=H, d_k=d_k, rank_r=rank_r)
    k_tilde, e_mask = dem(k_hat, z)
    assert k_tilde.shape == (B, T, H, d_k)
    assert e_mask.shape == (B, T, H, d_k)
    assert (e_mask >= 0.0).all() and (e_mask <= 1.0).all()
    assert torch.allclose(k_tilde, k_hat * e_mask, atol=1e-6)


def test_asn_normalization():
    B, T, d_model, H, d_k = 2, 10, 64, 4, 16
    asn = ASN(d_model=d_model, n_heads=H, ema_lambda=0.01, eps=1e-6)
    
    x = torch.randn(B, T, d_model)
    k_hat = torch.randn(B, T, H, d_k)
    
    beta_asn_seq, m_final, raw_beta = asn.forward_sequence(x, k_hat)
    assert beta_asn_seq.shape == (B, T, H)
    assert m_final.shape == (B, H)
    assert (raw_beta >= 0.0).all() and (raw_beta <= 1.0).all()
    assert (m_final > 0.0).all()


def test_stability_projection_clipping():
    sp = StabilityProjection(margin=1.0, eps=1e-6)
    d_k = 16
    alpha = torch.full((d_k,), 0.8)
    k_hat = torch.randn(d_k)
    k_tilde = torch.randn(d_k)
    beta_asn = torch.tensor(10.0)  # very large beta that should be projected down
    
    beta_sp, beta_max, alpha_max = sp(alpha, beta_asn, k_hat, k_tilde)
    assert beta_sp <= beta_max + 1e-7
    assert beta_sp < beta_asn
    assert torch.isclose(alpha_max, torch.tensor(0.8), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Layers Tests: BDALayer, CRPT, Attention, BDABlock
# ---------------------------------------------------------------------------

def test_bda_layer_forward_and_step_consistency():
    B, T, d_model = 2, 16, 64
    config = BDAConfig(
        d_model=d_model,
        n_heads=4,
        d_k=16,
        d_v=16,
        rank_r=4,
        chunk_size=8
    )
    layer = BDALayer(config)
    layer.eval()
    
    x = torch.randn(B, T, d_model)
    
    # 1. Full sequence forward
    with torch.no_grad():
        out_seq, (S_final_seq, m_final_seq) = layer(x, use_crpt_autograd=False)
        
    # 2. Step by step streaming
    with torch.no_grad():
        outs = []
        state = None
        for t in range(T):
            out_t, state = layer.step(x[:, t, :], state=state)
            outs.append(out_t)
        out_step = torch.stack(outs, dim=1)
        S_final_step, m_final_step = state
        
    assert torch.allclose(out_seq, out_step, atol=1e-5)
    assert torch.allclose(S_final_seq, S_final_step, atol=1e-5)
    assert torch.allclose(m_final_seq, m_final_step, atol=1e-5)


def test_crpt_autograd_gradient():
    B, T, d_model = 2, 16, 64
    config = BDAConfig(
        d_model=d_model,
        n_heads=2,
        d_k=16,
        d_v=16,
        rank_r=4,
        chunk_size=8
    )
    layer = BDALayer(config)
    layer.train()
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    out, (S_f, m_f) = layer(x, use_crpt_autograd=True)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert (x.grad.abs().sum() > 0.0)


def test_causal_full_attention_cache():
    B, T, d_model = 2, 8, 64
    attn = CausalFullAttention(d_model=d_model, n_heads=4)
    attn.eval()
    
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out_full, _ = attn(x)
        
        # Step with KV cache
        kv_cache = None
        step_outs = []
        for t in range(T):
            out_t, kv_cache = attn(x[:, t:t+1, :], kv_cache=kv_cache)
            step_outs.append(out_t)
        out_step = torch.cat(step_outs, dim=1)
        
    assert torch.allclose(out_full, out_step, atol=1e-5)


def test_bda_block():
    config = BDAConfig(d_model=64, n_heads=4, d_k=16, d_v=16, rank_r=4)
    block = BDABlock(config)
    x = torch.randn(2, 12, 64)
    out, state = block(x)
    assert out.shape == (2, 12, 64)
    assert state[0].shape == (2, 4, 16, 16)


# ---------------------------------------------------------------------------
# 3. Model Tests: BDAHybridModel Forward, Loss, and Autoregressive Generation
# ---------------------------------------------------------------------------

def test_hybrid_model_forward_and_loss():
    config = BDAConfig(
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        rank_r=4,
        n_layers=4,
        hybrid_interval=2,  # Layer 2 and Layer 4 are Full Attention
        vocab_size=100,
        max_seq_len=128
    )
    model = BDAHybridModel(config)
    
    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    
    logits, loss, caches = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.item() > 0.0
    assert len(caches) == 4
    
    # Backprop test on entire hybrid model
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_hybrid_model_generate():
    config = BDAConfig(
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        rank_r=4,
        n_layers=3,
        hybrid_interval=3,
        vocab_size=50,
        max_seq_len=64
    )
    model = BDAHybridModel(config)
    model.eval()
    
    prompt = torch.randint(0, 50, (2, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert generated.shape == (2, 15)
    # Check that the prompt is preserved as prefix
    assert (generated[:, :5] == prompt).all()
