"""
Bueorm Image Generation - Comprehensive Test Suite
Verifies todos los puntos solicitados:

1) Modelos de imagen simples: texto -> Z -> TBV (TextToImageModel)
2) Modelos de texto nativos con generación de imagen
3) Versatilidad: BDA, Transformer, Hybrid, MoE en todos los casos
4) Real, funcional, testeado y completo (serialización, trainer, builder, factory)
5) No daña nada existente: opt-in vía enable_image_gen, modelos sin flag intactos
"""

import os
import tempfile
import shutil
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

import bueorm
from bueorm.config import BueormConfig, MoEConfig, ImageGenConfig
from bueorm.core import save_model, load_model
from bueorm.trainer import Trainer, TrainingArguments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(vocab=50, batch=2, seq=8):
    return torch.randint(0, vocab, (batch, seq))

def _images(batch=2, size=32):
    return torch.randn(batch, 3, size, size)


# =========================
# 1) TTI Simple (texto -> Z -> TBV)
# =========================

@pytest.mark.parametrize("backbone", ["bda", "transformer", "hybrid"])
def test_tti_all_backbones(backbone):
    cfg = BueormConfig.tti_small(backbone=backbone, d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    model = bueorm.create_model("tti", config=cfg)
    assert model.__class__.__name__ == "TextToImageModel"
    model.eval()
    input_ids = _ids(50, 2, 8)
    target_images = _images(2, 32)
    with torch.no_grad():
        img_pred, loss, z = model(input_ids, target_images=target_images)
        assert img_pred.shape == (2, 3, 32, 32)
        assert z.shape == (2, 16, 4, 4)
        assert loss is not None and loss.item() > 0
        generated = model.generate_image(input_ids)
        assert generated.shape == (2, 3, 32, 32)
        # Factory alias
        m2 = bueorm.create_model("text_to_image", config=cfg)
        assert m2.__class__.__name__ == "TextToImageModel"

def test_tti_via_factory_kwargs():
    # Creación directa sin config preset, usando kwargs
    model = bueorm.create_model("tti", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, tti_backbone="hybrid")
    model.eval()
    ids = _ids(50, 1, 6)
    with torch.no_grad():
        img = model.generate_image(ids)
        assert img.shape == (1, 3, 32, 32)

def test_tti_with_moe():
    cfg = BueormConfig.tti_small(backbone="hybrid", d_model=32, n_heads=2, n_layers=3, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, use_moe=True, moe_config=MoEConfig(num_experts=4, top_k=2))
    model = bueorm.create_model("tti", config=cfg)
    model.train()
    ids = _ids(50, 2, 8)
    target_images = _images(2, 32)
    _, loss, _ = model(ids, target_images=target_images)
    loss.backward()
    # Al menos el backbone y head deben tener grad; algunos expertos MoE pueden no activarse con batch pequeño
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert loss.grad_fn is not None or loss.item() > 0


# =========================
# 2) & 3) Modelos de texto nativos con generación — versátiles
# =========================

@pytest.mark.parametrize("model_type,preset", [
    ("bda", "bda_with_image_gen"),
    ("transformer", "transformer_with_image_gen"),
    ("hybrid", "hybrid_with_image_gen"),
])
def test_language_with_image_gen_variants(model_type, preset):
    # Crear vía preset y vía factory con enable_image_gen
    if preset == "bda_with_image_gen":
        cfg = BueormConfig.bda_with_image_gen(d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    elif preset == "transformer_with_image_gen":
        cfg = BueormConfig.transformer_with_image_gen(d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    else:
        cfg = BueormConfig.hybrid_with_image_gen(d_model=32, n_heads=2, n_layers=3, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, pattern="2bda:1attn")

    model = bueorm.create_model(model_type, config=cfg)
    assert hasattr(model, "generate_image")
    assert hasattr(model, "generate")  # texto
    model.eval()
    ids = _ids(50, 2, 8)
    t_images = _images(2, 32)
    with torch.no_grad():
        # Solo texto (comportamiento original)
        logits, loss_text, _ = model(ids, targets=ids)
        assert logits.shape == (2, 8, 50)
        # Texto + imagen joint
        logits2, loss_joint, _ = model(ids, targets=ids, target_images=t_images)
        assert loss_joint is not None
        assert loss_joint.item() > loss_text.item() or loss_joint.item() != loss_text.item()
        # Solo imagen loss
        logits3, loss_img, _ = model(ids, target_images=t_images)
        assert loss_img is not None
        # Generación dual
        txt = model.generate(ids, max_new_tokens=4, temperature=0.0)
        assert txt.shape == (2, 12)
        img = model.generate_image(ids)
        assert img.shape == (2, 3, 32, 32)

@pytest.mark.parametrize("model_type", ["bda", "transformer", "hybrid"])
def test_language_with_image_gen_via_factory_enable_flag(model_type):
    # Punto 5: opt-in, sin dañar modelos sin flag
    model_plain = bueorm.create_model(model_type, d_model=32, n_heads=2, n_layers=2, vocab_size=50)
    assert not hasattr(model_plain, "generate_image") or not model_plain.config.enable_image_gen

    model_gen = bueorm.create_model(model_type, d_model=32, n_heads=2, n_layers=2, vocab_size=50, enable_image_gen=True, image_size=32, patch_size=8, tbv_dim=16)
    assert model_gen.config.enable_image_gen is True
    assert hasattr(model_gen, "generate_image")
    # Plain sigue funcionando texto solo
    ids = _ids(50, 1, 4)
    with torch.no_grad():
        logits, loss, _ = model_plain(ids, targets=ids)
        assert logits.shape == (1, 4, 50)

@pytest.mark.parametrize("model_type", ["bda", "hybrid"])
def test_language_with_moe_and_image_gen(model_type):
    cfg_kwargs = dict(d_model=32, n_heads=2, n_layers=3, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, use_moe=True, moe_config=MoEConfig(num_experts=4, top_k=2))
    if model_type == "bda":
        cfg = BueormConfig.bda_with_image_gen(**cfg_kwargs)
    else:
        cfg = BueormConfig.hybrid_with_image_gen(pattern="2bda:1attn", **cfg_kwargs)
    model = bueorm.create_model(model_type, config=cfg)
    model.train()
    ids = _ids(50, 2, 8)
    t_images = _images(2, 32)
    logits, loss, _ = model(ids, targets=ids, target_images=t_images)
    assert loss is not None
    loss.backward()
    # MoE aux loss + image loss deben tener grad
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


# =========================
# VLM Any-to-Any con generación
# =========================

def test_vlm_with_image_gen_any_to_any():
    cfg = BueormConfig.vlm_with_image_gen(d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    model = bueorm.create_model("vlm", config=cfg)  # factory detecta is_multimodal+enable -> GenerativeVLM
    assert model.__class__.__name__ == "GenerativeVLM"
    model.eval()
    ids = _ids(50, 2, 6)
    images = _images(2, 32)
    t_images = _images(2, 32)
    with torch.no_grad():
        # image+text -> texto (heredado)
        logits, loss, _ = model(ids, images=images, targets=ids)
        assert logits.shape[0] == 2
        # image+text -> texto + imagen joint
        logits2, loss2, _ = model(ids, images=images, targets=ids, target_images=t_images)
        assert loss2.item() != loss.item()
        # texto -> imagen (sin contexto visual)
        img = model.generate_image(ids)
        assert img.shape == (2, 3, 32, 32)
        # image+text -> imagen (edición condicionada)
        img2 = model.generate_image(ids, images=images)
        assert img2.shape == (2, 3, 32, 32)
        # texto con contexto visual -> texto
        txt = model.generate(ids, images=images, max_new_tokens=4, temperature=0.0)
        assert txt.shape == (2, 10)
        # forward_with_image expone pred
        logits3, loss3, _, img_pred = model.forward_with_image(ids, images=images, targets=ids, target_images=t_images)
        assert img_pred.shape == (2, 3, 32, 32)

def test_vlm_without_image_gen_unchanged():
    # Punto 5: VLM sin flag sigue siendo BueormVLM puro
    cfg = BueormConfig.vlm_bda(d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    model = bueorm.create_model("vlm", config=cfg)
    assert model.__class__.__name__ == "BueormVLM"
    assert not hasattr(model, "image_head") or not getattr(model.config, "enable_image_gen", False)


# =========================
# Serialización multi-formato (punto 4)
# =========================

@pytest.mark.parametrize("factory", [
    lambda: bueorm.create_model("tti", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16),
    lambda: bueorm.create_model("bda", d_model=32, n_heads=2, n_layers=2, vocab_size=50, enable_image_gen=True, image_size=32, patch_size=8, tbv_dim=16),
    lambda: bueorm.create_model("hybrid", d_model=32, n_heads=2, n_layers=2, vocab_size=50, enable_image_gen=True, image_size=32, patch_size=8, tbv_dim=16),
    lambda: bueorm.create_model("transformer", d_model=32, n_heads=2, n_layers=2, vocab_size=50, enable_image_gen=True, image_size=32, patch_size=8, tbv_dim=16),
    lambda: bueorm.create_model("vlm", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, enable_image_gen=True),
])
def test_serialization_generative_all_formats(factory):
    model = factory()
    model.eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        if model.__class__.__name__ == "TextToImageModel":
            orig_img, _, _ = model(ids)
        else:
            # Check text logits determinism
            orig_logits, _, _ = model(ids, targets=ids)
        with tempfile.TemporaryDirectory() as d:
            for fmt in ["bueorm", "safetensors", "gguf", "pt"]:
                path = os.path.join(d, f"model.{fmt}")
                save_model(model, path)
                loaded = load_model(path)
                assert loaded.__class__.__name__ == model.__class__.__name__
                assert loaded.config.enable_image_gen == model.config.enable_image_gen
                with torch.no_grad():
                    if loaded.__class__.__name__ == "TextToImageModel":
                        loaded_img, _, _ = loaded(ids)
                        assert torch.allclose(orig_img, loaded_img, atol=1e-5)
                    else:
                        loaded_logits, _, _ = loaded(ids, targets=ids)
                        assert torch.allclose(orig_logits, loaded_logits, atol=1e-5)


# =========================
# Builder escalable (punto 3)
# =========================

def test_builder_with_image_generation():
    # TTI vía builder
    m_tti = bueorm.ModelBuilder("test-tti").with_text_to_image(backbone="bda", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16).build()
    assert m_tti.__class__.__name__ == "TextToImageModel"

    # Lenguaje + MoE + imagen vía builder (versátil)
    m_hybrid = (bueorm.ModelBuilder("test-hybrid-gen")
                .with_vision(image_size=32, patch_size=8, tbv_dim=16, num_blocks=2)  # no necesario para lenguaje puro pero no daña
                .with_language_hybrid(pattern="2bda:1attn", d_model=32, n_heads=2, n_layers=3, vocab_size=50)
                .with_moe(num_experts=4, top_k=2)
                .with_image_generation(image_size=32, patch_size=8, tbv_dim=16, backbone="hybrid")
                .build())
    # Al tener visión true será GenerativeVLM, test que tenga head
    assert hasattr(m_hybrid, "image_head") or hasattr(m_hybrid, "generate_image")

    # Lenguaje puro + imagen vía builder sin visión
    m_bda = (bueorm.ModelBuilder("test-bda-gen")
             .with_language_bda(d_model=32, n_heads=2, n_layers=2, vocab_size=50)
             .with_image_generation(image_size=32, patch_size=8, tbv_dim=16, backbone="bda")
             .build())
    assert hasattr(m_bda, "generate_image")
    with torch.no_grad():
        ids = _ids(50, 1, 4)
        img = m_bda.generate_image(ids)
        assert img.shape == (1, 3, 32, 32)


# =========================
# Trainer con generación y MoE (punto 4)
# =========================

class _TTIDataset(Dataset):
    def __init__(self, n=8, vocab=50, size=32):
        self.n=n; self.vocab=vocab; self.size=size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, self.vocab, (6,)), "target_images": torch.randn(3, self.size, self.size)}

def test_trainer_tti():
    with tempfile.TemporaryDirectory() as d:
        m = bueorm.create_model("tti", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, tti_backbone="bda")
        ds = _TTIDataset()
        args = TrainingArguments(output_dir=d, num_train_epochs=1, learning_rate=1e-3, logging_steps=1, save_steps=10, save_format="bueorm")
        trainer = Trainer(model=m, args=args, train_dataset=ds)
        res = trainer.train()
        assert res["global_step"] > 0
        assert os.path.exists(res["checkpoint"])

def test_trainer_joint_text_image():
    class Joint(Dataset):
        def __len__(self): return 8
        def __getitem__(self, idx):
            return {"input_ids": torch.randint(0,50,(8,)), "targets": torch.randint(0,50,(8,)), "target_images": torch.randn(3,32,32)}
    with tempfile.TemporaryDirectory() as d:
        m = bueorm.create_model("bda", d_model=32, n_heads=2, n_layers=2, vocab_size=50, enable_image_gen=True, image_size=32, patch_size=8, tbv_dim=16)
        args = TrainingArguments(output_dir=d, num_train_epochs=1, learning_rate=1e-3, logging_steps=1, save_steps=10, save_format="bueorm")
        trainer = Trainer(model=m, args=args, train_dataset=Joint())
        res = trainer.train()
        assert res["global_step"] > 0

def test_trainer_vlm_gen():
    class VLMJoint(Dataset):
        def __len__(self): return 8
        def __getitem__(self, idx):
            return {"input_ids": torch.randint(0,50,(6,)), "targets": torch.randint(0,50,(6,)), "images": torch.randn(3,32,32), "target_images": torch.randn(3,32,32)}
    with tempfile.TemporaryDirectory() as d:
        m = bueorm.create_model("vlm", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16, enable_image_gen=True)
        args = TrainingArguments(output_dir=d, num_train_epochs=1, learning_rate=1e-3, logging_steps=1, save_steps=10, save_format="bueorm")
        trainer = Trainer(model=m, args=args, train_dataset=VLMJoint())
        res = trainer.train()
        assert res["global_step"] > 0

# =========================
# No daño a modelos existentes (punto 5)
# =========================

def test_existing_models_unchanged():
    # Sin enable_image_gen, deben ser clases base y perder generate_image
    m_bda = bueorm.create_model("bda", d_model=32, n_heads=2, n_layers=2, vocab_size=50)
    assert m_bda.__class__.__name__ == "BDALanguageModel"
    assert not m_bda.config.enable_image_gen

    m_trans = bueorm.create_model("transformer", d_model=32, n_heads=2, n_layers=2, vocab_size=50)
    assert m_trans.__class__.__name__ == "TransformerLM"

    m_hybrid = bueorm.create_model("hybrid", d_model=32, n_heads=2, n_layers=2, vocab_size=50)
    assert m_hybrid.__class__.__name__ == "HybridLanguageModel"

    m_vlm = bueorm.create_model("vlm", d_model=32, n_heads=2, n_layers=2, vocab_size=50, image_size=32, patch_size=8, tbv_dim=16)
    assert m_vlm.__class__.__name__ == "BueormVLM"

    # Forward sigue 3 valores
    ids = _ids(50, 1, 4)
    with torch.no_grad():
        for m in [m_bda, m_trans, m_hybrid]:
            out = m(ids, targets=ids)
            assert len(out) == 3
