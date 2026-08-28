# Bueorm Model Maker

**Bueorm** es una biblioteca modular de arquitecturas neuronales avanzadas y framework de modelos de lenguaje, visión y multimodalidad desarrollado por **Gerson Fabián Buenahora Ormaza (BUEORM)**.

---

## 🌟 Arquitecturas Integradas

1. **BDA (BUEORM Delta Attention):** Memoria asociativa recurrente de estado fijo ($S \in \mathbb{R}^{d_v \times d_k}$) con complejidad de inferencia $O(1)$, olvido por canal (**LRFG**), máscara de borrado desacoplada (**DEM**), normalización adaptativa (**ASN**), cota de estabilidad analítica (**SP**) y entrenamiento por bloques (**CRPT**).
2. **TBV (T-Bidirectional Vision):** Red neuronal de transformación visual bidireccional única $T(x, d)$ con pesos estrictamente compartidos entre $T(\text{Image}, +1) \to Z$ y $T(Z, -1) \to \text{Image}$, equipada con proyector de características espaciales 2D para VLM.
3. **Transformer:** Atención causal escalable moderna con **FlashAttention / SDPA**, **Grouped-Query Attention (GQA)**, **Rotary Position Embeddings (RoPE)** y **KV-Cache dinámico**.
4. **Modelos Híbridos Universales:** Combinación de capas BDA y FlashAttention en proporciones configurables (ej. 4 BDA : 1 FlashAttention).
5. **Modelos Multimodales (VLM):** Fusión de visión (TBV) + proyección 2D + razonamiento autorregresivo de lenguaje (BDA / Transformer / Híbrido).
6. **Mixture of Experts (MoE) Nativo:** Enrutador Top-$k$ disperso con pérdida auxiliar de balanceo de carga (`aux_loss`), compatible con todas las arquitecturas.
7. **Motor Multi-Formato:** Guardado y carga en `.bueorm` (contenedor autocontenido optimizado), `.safetensors`, `.gguf` (v3 binario puro) y `.pt`.

---

## 🚀 Guía Rápida de Uso

### 1. Crear y Ejecutar un Modelo BDA Puro
```python
import torch
import bueorm

# Crear modelo BDA puro
model = bueorm.create_model("bda", d_model=256, n_heads=4, n_layers=6)

# Inferencia rápida
prompt = torch.tensor([[10, 25, 33, 42]])
output_tokens = model.generate(prompt, max_new_tokens=20)
print("Generado:", output_tokens)
```

### 2. Modelo Híbrido BDA + FlashAttention (ej. 4:1)
```python
# Crea un modelo con 4 capas BDA por cada 1 capa de FlashAttention
model = bueorm.create_model(
    "hybrid",
    hybrid_pattern="4bda:1attn",
    d_model=512,
    n_heads=8,
    n_layers=10
)

# Forward pass con pérdida
input_ids = torch.randint(0, 1000, (2, 32))
targets = torch.randint(0, 1000, (2, 32))
logits, loss, _ = model(input_ids, targets=targets)
```

### 3. Modelo Multimodal VLM (TBV + BDA + FlashAttention)
```python
# Crear VLM completo
vlm = bueorm.create_model(
    "vlm",
    image_size=128,
    patch_size=8,
    d_model=256,
    n_layers=6,
    hybrid_pattern="3bda:1attn"
)

# Inferencia multimodal condicionada por imagen
image = torch.randn(1, 3, 128, 128)
text_prompt = torch.tensor([[1, 45, 88]])
generated = vlm.generate(prompt_ids=text_prompt, images=image, max_new_tokens=30)
```

### 4. Mixture of Experts (MoE)
```python
# Crear modelo Híbrido con 8 expertos (Top-2 activos)
moe_model = bueorm.create_model(
    "hybrid",
    use_moe=True,
    moe_config=bueorm.MoEConfig(num_experts=8, top_k=2),
    d_model=256,
    n_layers=6
)
```

### 5. Guardar y Cargar en Múltiples Formatos
```python
# 1. Formato propio autocontenido .bueorm
bueorm.save_model(model, "mi_modelo.bueorm")
loaded_bueorm = bueorm.load_model("mi_modelo.bueorm")

# 2. Formato .safetensors
bueorm.save_model(model, "mi_modelo.safetensors")

# 3. Formato .gguf (exportación nativa GGUF v3)
bueorm.save_model(model, "mi_modelo.gguf")

# 4. Formato estándar .pt
bueorm.save_model(model, "mi_modelo.pt")

# Exportar a todos los formatos simultáneamente
bueorm.export_model(model, output_dir="./exports", base_name="bueorm_v1")
```

### 6. Cuantización a `int8`
```python
# Cuantización de pesos int8 (reduce la VRAM hasta 4x)
quantized_model = bueorm.quantize_model(model, mode="int8_weight")
memory_info = bueorm.get_model_memory_footprint(quantized_model)
print("Memoria Total:", memory_info["total_memory_mb"], "MB")
```

### 7. Entrenamiento con el Trainer de Bueorm
```python
from torch.utils.data import TensorDataset

dataset = TensorDataset(torch.randint(0, 1000, (100, 32)))

training_args = bueorm.TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    learning_rate=3e-4,
    save_format="bueorm"
)

trainer = bueorm.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

---

## 🧪 Tests Unitarios

Para ejecutar toda la suite de pruebas del framework:
```bash
python -m pytest tests/ -v
```
