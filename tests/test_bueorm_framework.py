"""
Bueorm Framework - Master Test Suite
Exhaustive verification of BDA, TBV, Transformer, Hybrid, MoE, VLM,
Multi-Format Serialization (.bueorm, .safetensors, .gguf, .pt), Quantization, Trainer, and Inference Pipeline.
"""

import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import bueorm
from bueorm.config import BueormConfig, MoEConfig, VLMConfig
from bueorm.models import (
    BDALanguageModel,
    TBVVisionModel,
    TransformerLM,
    HybridLanguageModel,
    BueormVLM,
    BueormModel,
    create_model,
)
from bueorm.moe import SparseMoELayer, TopKRouter
from bueorm.utils import ModelBuilder, calculate_active_vs_total_params
from bueorm.core import (
    quantize_model,
    get_model_memory_footprint,
    save_model,
    load_model,
    pipeline,
    register_model,
    MODEL_REGISTRY,
)
from bueorm.trainer import Trainer, TrainingArguments


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Config, Builder & Registry Tests
# ---------------------------------------------------------------------------

def test_bueorm_config_presets():
    cfg_bda = BueormConfig.bda_small()
    assert cfg_bda.model_type == "bda"
    
    cfg_hybrid = BueormConfig.hybrid_4_to_1()
    assert cfg_hybrid.hybrid_pattern == "4bda:1attn"
    
    cfg_vlm = BueormConfig.vlm_bda()
    assert cfg_vlm.is_multimodal is True
    
    cfg_moe = BueormConfig.moe_hybrid(num_experts=8, top_k=2)
    assert cfg_moe.use_moe is True
    assert cfg_moe.moe_config.num_experts == 8

    # Serialization roundtrip
    json_str = cfg_moe.to_json()
    cfg_loaded = BueormConfig.from_json(json_str)
    assert cfg_loaded.moe_config.num_experts == 8


def test_model_builder_fluent_lego():
    # Construcción fluida como bloques de Lego: TBV + Hybrid (BDA + ATTN) + MoE con 16k contexto
    model = (
        ModelBuilder("lego-vlm-moe")
        .with_vision(image_size=64, patch_size=8, tbv_dim=32, num_blocks=2)
        .with_language_hybrid(pattern="3bda:1attn", d_model=64, n_heads=4, n_layers=4, max_seq_len=16384, vocab_size=100)
        .with_moe(num_experts=4, top_k=2)
        .build()
    )
    assert isinstance(model, BueormVLM)
    
    # Cálculo de parámetros activos vs totales
    calc = calculate_active_vs_total_params(model.config)
    assert calc["total_parameters"] > 0
    assert calc["active_parameters"] <= calc["total_parameters"]


def test_custom_registry_extension():
    @register_model("custom_dummy")
    class DummyCustomModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x)

    assert "custom_dummy" in MODEL_REGISTRY
    inst = MODEL_REGISTRY.get("custom_dummy")()
    assert isinstance(inst, DummyCustomModel)


# ---------------------------------------------------------------------------
# 2. Individual Models: BDA, TBV, Transformer
# ---------------------------------------------------------------------------

def test_bda_language_model():
    cfg = BueormConfig.bda_small(d_model=64, n_heads=4, d_k=16, d_v=16, n_layers=2, vocab_size=100)
    model = BDALanguageModel(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 8, 100)
    assert loss is not None

    generated = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    assert generated.shape == (2, 13)


def test_tbv_vision_model():
    cfg = BueormConfig.tbv_small(image_size=64, patch_size=8, tbv_dim=32, tbv_num_blocks=2)
    model = TBVVisionModel(cfg)
    model.eval()

    images = torch.randn(2, 3, 64, 64)
    # Forward cycle
    z = model.encode(images)
    assert z.shape == (2, 32, 8, 8)
    rec = model.decode(z)
    assert rec.shape == (2, 3, 64, 64)
    
    # Direct feature extraction
    feats = model.extract_features(images, return_tokens=True)
    assert feats.shape == (2, 64, 32)


def test_transformer_lm_flash_attn():
    cfg = BueormConfig.transformer_small(d_model=64, n_heads=4, n_layers=2, vocab_size=100, use_flash_attn=True)
    model = TransformerLM(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 8))
    logits, loss, _ = model(input_ids, targets=input_ids)
    assert logits.shape == (2, 8, 100)
    assert loss is not None

    generated = model.generate(input_ids, max_new_tokens=4, temperature=0.0)
    assert generated.shape == (2, 12)


# ---------------------------------------------------------------------------
# 3. Hybrid Architecture (e.g. 4 BDA : 1 FlashAttention)
# ---------------------------------------------------------------------------

def test_hybrid_language_model():
    cfg = BueormConfig(
        model_type="hybrid",
        hybrid_pattern="2bda:1attn",
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        n_layers=6,  # [bda, bda, attn, bda, bda, attn]
        vocab_size=100,
        max_seq_len=64
    )
    model = HybridLanguageModel(cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 10))
    logits, loss, _ = model(input_ids, targets=input_ids)
    assert logits.shape == (2, 10, 100)
    assert loss is not None

    # Hybrid step generation (mixing recurrent memory & KV cache)
    generated = model.generate(input_ids, max_new_tokens=6, temperature=0.0)
    assert generated.shape == (2, 16)


# ---------------------------------------------------------------------------
# 4. Mixture of Experts (MoE)
# ---------------------------------------------------------------------------

def test_moe_layer_and_router():
    dim = 64
    moe = SparseMoELayer(dim=dim, num_experts=4, top_k=2)
    moe.train()

    x = torch.randn(2, 8, dim)
    out, aux_loss = moe(x)
    assert out.shape == (2, 8, dim)
    assert aux_loss is not None
    assert aux_loss.item() >= 0.0


def test_hybrid_moe_model():
    cfg = BueormConfig.moe_hybrid(
        num_experts=4,
        top_k=2,
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        n_layers=3,
        vocab_size=100
    )
    model = HybridLanguageModel(cfg)
    model.train()

    input_ids = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    logits, loss, _ = model(input_ids, targets=targets)
    assert logits.shape == (2, 8, 100)
    assert loss is not None
    
    # Backprop including router and experts
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None


# ---------------------------------------------------------------------------
# 5. Multimodal Vision-Language Model (BueormVLM)
# ---------------------------------------------------------------------------

def test_multimodal_vlm():
    cfg = BueormConfig.vlm_bda(
        image_size=64,
        patch_size=8,
        tbv_dim=32,
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        n_layers=3,
        vocab_size=100
    )
    vlm = BueormVLM(cfg)
    vlm.eval()

    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 6))
    targets = torch.randint(0, 100, (2, 6))

    # Multimodal forward
    logits, loss, _ = vlm(input_ids=input_ids, images=images, targets=targets)
    assert logits.shape == (2, 70, 100)
    assert loss is not None

    # Conditioned generation
    gen = vlm.generate(prompt_ids=input_ids, images=images, max_new_tokens=4, temperature=0.0)
    assert gen.shape == (2, 10)


# ---------------------------------------------------------------------------
# 6. Multi-Format Serialization (.bueorm, .safetensors, .gguf, .pt)
# ---------------------------------------------------------------------------

def test_serialization_all_formats(temp_dir):
    cfg = BueormConfig.hybrid_4_to_1(d_model=64, n_heads=4, d_k=16, d_v=16, n_layers=5, vocab_size=100)
    model = create_model("hybrid", config=cfg)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        orig_logits, _, _ = model(input_ids)

    # 1. Test .bueorm format (Proprietary Optimized Container)
    bueorm_path = os.path.join(temp_dir, "model.bueorm")
    save_model(model, bueorm_path)
    loaded_bueorm = load_model(bueorm_path)
    with torch.no_grad():
        bueorm_logits, _, _ = loaded_bueorm(input_ids)
    assert torch.allclose(orig_logits, bueorm_logits, atol=1e-5)

    # 2. Test .safetensors format
    safe_path = os.path.join(temp_dir, "model.safetensors")
    save_model(model, safe_path)
    loaded_safe = load_model(safe_path)
    with torch.no_grad():
        safe_logits, _, _ = loaded_safe(input_ids)
    assert torch.allclose(orig_logits, safe_logits, atol=1e-5)

    # 3. Test .gguf format
    gguf_path = os.path.join(temp_dir, "model.gguf")
    save_model(model, gguf_path)
    loaded_gguf = load_model(gguf_path)
    with torch.no_grad():
        gguf_logits, _, _ = loaded_gguf(input_ids)
    assert torch.allclose(orig_logits, gguf_logits, atol=1e-5)

    # 4. Test .pt format
    pt_path = os.path.join(temp_dir, "model.pt")
    save_model(model, pt_path)
    loaded_pt = load_model(pt_path)
    with torch.no_grad():
        pt_logits, _, _ = loaded_pt(input_ids)
    assert torch.allclose(orig_logits, pt_logits, atol=1e-5)

    # 5. Export all simultaneously
    exports = bueorm.export_model(model, output_dir=os.path.join(temp_dir, "exports"))
    assert "bueorm" in exports
    assert "safetensors" in exports
    assert "gguf" in exports
    assert "pt" in exports


# ---------------------------------------------------------------------------
# 7. Quantization Engine
# ---------------------------------------------------------------------------

def test_quantization():
    cfg = BueormConfig.hybrid_4_to_1(d_model=64, n_heads=4, d_k=16, d_v=16, n_layers=3, vocab_size=100)
    model = create_model("hybrid", config=cfg)
    
    footprint_fp32 = get_model_memory_footprint(model)
    assert footprint_fp32["total_memory_mb"] > 0
    
    # Weight int8 quantization
    quant_model = quantize_model(model, mode="int8_weight")
    
    input_ids = torch.randint(0, 100, (1, 6))
    out = quant_model.generate(input_ids, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 9)


# ---------------------------------------------------------------------------
# 8. Inference Pipeline
# ---------------------------------------------------------------------------

def test_inference_pipeline():
    cfg = BueormConfig.hybrid_4_to_1(d_model=64, n_heads=4, d_k=16, d_v=16, n_layers=2, vocab_size=100)
    model = create_model("hybrid", config=cfg)
    
    pipe = pipeline(model)
    out = pipe(inputs=[10, 20, 30], max_new_tokens=4, temperature=0.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 7)


# ---------------------------------------------------------------------------
# 9. Trainer Framework
# ---------------------------------------------------------------------------

def test_trainer(temp_dir):
    cfg = BueormConfig.bda_small(d_model=32, n_heads=2, d_k=16, d_v=16, n_layers=2, vocab_size=50)
    model = create_model("bda", config=cfg)

    # Synthetic dataset
    x_data = torch.randint(0, 50, (16, 8))
    dataset = TensorDataset(x_data)

    args = TrainingArguments(
        output_dir=temp_dir,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=2,
        save_format="bueorm"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    res = trainer.train()
    assert res["global_step"] > 0
    assert os.path.exists(res["checkpoint"])
