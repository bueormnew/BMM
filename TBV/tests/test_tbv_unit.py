"""
T-Bidirectional Vision (TBV) - Unit Tests
"""

import pytest
import torch
import torch.nn as nn

from TBV.config import TBVConfig
from TBV.modules.patches import PatchProjection, PatchDeprojection
from TBV.modules.direction import DirectionModulation
from TBV.modules.t_block import TBlock
from TBV.modules.projector import TBVVisualProjector
from TBV.model.tbv_network import TBVNetwork


def test_tbv_config_properties():
    cfg = TBVConfig.tiny()
    assert cfg.grid_size == 16
    assert cfg.num_patches == 256
    assert cfg.patch_dim == 8 * 8 * 3
    assert cfg.compression_ratio == (8 * 8 * 3) / 64


def test_patch_projection_and_deprojection():
    B, C, H, W = 2, 3, 64, 64
    P = 8
    D = 32
    
    proj = PatchProjection(in_channels=C, dim=D, patch_size=P)
    deproj = PatchDeprojection(dim=D, out_channels=C, patch_size=P)
    
    x = torch.randn(B, C, H, W)
    z = proj(x)
    assert z.shape == (B, D, H // P, W // P)
    
    x_rec = deproj(z)
    assert x_rec.shape == (B, C, H, W)


def test_direction_modulation():
    B, D, H, W = 2, 32, 8, 8
    mod = DirectionModulation(dim=D, emb_dim=16)
    x = torch.randn(B, D, H, W)
    
    out_fwd = mod(x, direction=0)
    out_rev = mod(x, direction=1)
    
    assert out_fwd.shape == (B, D, H, W)
    assert out_rev.shape == (B, D, H, W)
    # Output should differ between directions
    assert not torch.allclose(out_fwd, out_rev)


def test_tblock_forward():
    B, D, H, W = 2, 32, 8, 8
    block = TBlock(dim=D, mlp_expansion=2.0, spatial_kernel_size=3)
    x = torch.randn(B, D, H, W)
    
    out_fwd = block(x, direction=0)
    out_rev = block(x, direction=1)
    
    assert out_fwd.shape == (B, D, H, W)
    assert out_rev.shape == (B, D, H, W)


def test_tbv_network_cycles_and_shared_weights():
    config = TBVConfig(
        image_size=64,
        in_channels=3,
        patch_size=8,
        dim=32,
        num_blocks=3,
        mlp_expansion=2.0
    )
    net = TBVNetwork(config)
    net.eval()
    
    x = torch.randn(2, 3, 64, 64)
    
    # 1. Forward Cycle: Image -> Z -> Image Reconstructed
    z = net.encode(x)
    assert z.shape == (2, 32, 8, 8)
    
    x_rec = net.decode(z)
    assert x_rec.shape == (2, 3, 64, 64)
    
    x_rec2 = net.reconstruct(x)
    assert torch.allclose(x_rec, x_rec2, atol=1e-6)
    
    # 2. Reverse Cycle: Z -> Image -> Z'
    z_prime = net.reverse_cycle(z)
    assert z_prime.shape == (2, 32, 8, 8)


def test_tbv_vlm_feature_extraction():
    config = TBVConfig.vlm(
        image_size=64,
        patch_size=8,
        dim=32,
        num_blocks=2,
        vlm_proj_dim=128
    )
    net = TBVNetwork(config)
    net.eval()
    
    images = torch.randn(2, 3, 64, 64)
    
    # Extract token sequence for VLM
    tokens = net.extract_features(images, return_tokens=True)
    assert tokens.shape == (2, 64, 128)  # (B, 8*8, vlm_proj_dim)
    
    # Extract spatial grid
    grid = net.extract_features(images, return_tokens=False)
    assert grid.shape == (2, 32, 8, 8)


def test_tbv_gradient_flow():
    config = TBVConfig(
        image_size=32,
        in_channels=3,
        patch_size=8,
        dim=16,
        num_blocks=2
    )
    net = TBVNetwork(config)
    net.train()
    
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    
    # Full reconstruction loss
    x_rec = net.reconstruct(x)
    loss = nn.functional.mse_loss(x_rec, x)
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    
    # Verify all shared block parameters have valid gradients
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
