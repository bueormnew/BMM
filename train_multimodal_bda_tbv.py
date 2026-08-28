"""
BUEORM Multimodal VLM (TBV Vision + BDA Language)
Synthetic Multimodal Learning Experiment - Causal Setup

Objective:
Verify that TBV (Bidirectional Vision Backbone) + 2D Projector + BDA (Delta Attention Language Backbone)
jointly learn a visual question-answering task with 100% causal correctness.
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import bueorm
from bueorm.config import BueormConfig, VLMConfig

# ─── Configuración del Experimento ──────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 4
IMAGE_SIZE = 32
PATCH_SIZE = 8
VOCAB_SIZE = 256
D_MODEL = 128
TBV_DIM = 64
N_LAYERS = 4
N_HEADS = 4
NUM_STEPS = 120
LR = 1e-3

# ─── Generación de Dataset Sintético Multimodal ─────────────────────────────
def generate_synthetic_multimodal_data(num_samples: int = 1920):
    """
    Generates synthetic images with distinct spatial patterns and corresponding Q&A token sequences.
    
    Classes:
      0: Top-Left Quadrant activated (Red pattern)
      1: Bottom-Right Quadrant activated (Green pattern)
      2: Center Box activated (Blue pattern)
      3: Diagonal Line activated (White pattern)
      
    Causal Language Setup:
      Input tokens:  [1, 10, 20]             (Prompt: "What is this image?")
      Target tokens: [10, 20, 100 + class_id] (Causally shifted target: answer is 100 + class_id)
    """
    torch.manual_seed(42)
    images = torch.zeros(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    
    for i in range(num_samples):
        cls = labels[i].item()
        noise = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.05
        if cls == 0:
            images[i, 0, :16, :16] = 1.0 # Top-Left Red
        elif cls == 1:
            images[i, 1, 16:, 16:] = 1.0 # Bottom-Right Green
        elif cls == 2:
            images[i, 2, 10:22, 10:22] = 1.0 # Center Blue Box
        elif cls == 3:
            for d in range(IMAGE_SIZE):
                images[i, :, d, d] = 1.0 # Diagonal
        images[i] = torch.clamp(images[i] + noise, 0.0, 1.0)
        
    input_ids = torch.tensor([[1, 10, 20]] * num_samples, dtype=torch.long)
    target_ids = torch.zeros(num_samples, 3, dtype=torch.long)
    target_ids[:, 0] = 10
    target_ids[:, 1] = 20
    for i in range(num_samples):
        target_ids[i, 2] = 100 + labels[i].item()
        
    return images, input_ids, target_ids, labels

# ─── Construcción del Modelo Multimodal BDA + TBV ───────────────────────────
def build_vlm_model():
    cfg = BueormConfig(
        model_type="bda", # Pure BDA language backbone
        is_multimodal=True,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_k=32,
        d_v=32,
        n_layers=N_LAYERS,
        vocab_size=VOCAB_SIZE,
        max_seq_len=64,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        tbv_dim=TBV_DIM,
        tbv_num_blocks=2,
        vlm_config=VLMConfig(freeze_vision=False) # Joint end-to-end training
    )
    model = bueorm.BueormVLM(cfg)
    return model, cfg

def main():
    print("\n" + "═"*70)
    print("  🚀 EXPERIMENTO MULTIMODAL BUEORM: TBV (VISIÓN) + BDA (LENGUAJE)")
    print("═"*70)
    
    # 1. Crear dataset
    images, inputs, targets, labels = generate_synthetic_multimodal_data(num_samples=1920)
    dataset = TensorDataset(images, inputs, targets, labels)
    train_size = 1600
    test_size = 320
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Instanciar Modelo
    model, cfg = build_vlm_model()
    model = model.to(DEVICE)
    
    n_params_tbv = sum(p.numel() for p in model.vision_backbone.parameters()) / 1e6
    n_params_proj = sum(p.numel() for p in model.projector.parameters()) / 1e6
    n_params_bda = sum(p.numel() for p in model.language_model.parameters()) / 1e6
    n_params_total = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\n  Arquitectura Multimodal:")
    print(f"    • Vision Backbone (TBV Bidireccional): {n_params_tbv:.3f} M params")
    print(f"    • Visual Projector 2D:                 {n_params_proj:.3f} M params")
    print(f"    • Language Backbone (BDA Recurrente):  {n_params_bda:.3f} M params")
    print(f"    • Total VLM:                           {n_params_total:.3f} M params")
    print(f"    • Dispositivo:                         {DEVICE.upper()}")
    print("─"*70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)
    
    # 3. Bucle de Entrenamiento Multimodal
    model.train()
    step = 0
    t0 = time.perf_counter()
    initial_loss = None
    
    print(f"\n  Entrenando modelo multimodal BDA + TBV end-to-end...")
    
    for epoch in range(10):
        for img_b, in_b, tgt_b, lbl_b in train_loader:
            if step >= NUM_STEPS:
                break
            step += 1
            
            img_b = img_b.to(DEVICE)
            in_b = in_b.to(DEVICE)
            tgt_b = tgt_b.to(DEVICE)
            
            optimizer.zero_grad()
            logits, loss, _ = model(input_ids=in_b, images=img_b, targets=tgt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if step == 1:
                initial_loss = loss.item()
                
            if step % 20 == 0 or step == 1:
                N_patches = (cfg.image_size // cfg.patch_size) ** 2
                text_logits = logits[:, N_patches:, :]
                pred_tokens = torch.argmax(text_logits[:, -1, :], dim=-1)
                expected_tokens = 100 + lbl_b.to(DEVICE)
                acc = (pred_tokens == expected_tokens).float().mean().item() * 100.0
                
                print(f"  Paso {step:3d}/{NUM_STEPS} | Pérdida: {loss.item():.4f} | PPL: {math.exp(min(loss.item(), 20)):.2f} | Acc Batch: {acc:5.1f}%")
                
        if step >= NUM_STEPS:
            break
            
    train_time = time.perf_counter() - t0
    final_train_loss = loss.item()
    
    # 4. Evaluación en Test Set
    print("\n" + "─"*70)
    print("  🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA (DATASET NUNCA VISTO)")
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_total = 0.0
    
    with torch.no_grad():
        for img_b, in_b, tgt_b, lbl_b in test_loader:
            img_b = img_b.to(DEVICE)
            in_b = in_b.to(DEVICE)
            tgt_b = tgt_b.to(DEVICE)
            lbl_b = lbl_b.to(DEVICE)
            
            logits, loss, _ = model(input_ids=in_b, images=img_b, targets=tgt_b)
            test_loss_total += loss.item() * len(lbl_b)
            
            N_patches = (cfg.image_size // cfg.patch_size) ** 2
            text_logits = logits[:, N_patches:, :]
            pred_tokens = torch.argmax(text_logits[:, -1, :], dim=-1)
            expected_tokens = 100 + lbl_b
            
            test_correct += (pred_tokens == expected_tokens).sum().item()
            test_total += len(lbl_b)
            
    test_acc = (test_correct / test_total) * 100.0
    avg_test_loss = test_loss_total / test_total
    
    # 5. Prueba de Generación Autoregresiva Multimodal (Inferencia Visual)
    print("\n  🔍 PRUEBA DE GENERACIÓN AUTOREGRESIVA CON IMAGEN:")
    sample_images, sample_inputs, _, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:4].to(DEVICE)
    prompt_ids = sample_inputs[:4].to(DEVICE)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_ids=prompt_ids,
            images=sample_images,
            max_new_tokens=1,
            temperature=0.0
        )
        
    class_names = ["Top-Left Red", "Bottom-Right Green", "Center Blue Box", "Diagonal Line"]
    for i in range(4):
        actual_cls = sample_labels[i].item()
        pred_token = generated_tokens[i, -1].item() # The newly generated token
        pred_cls = pred_token - 100
        pred_name = class_names[pred_cls] if 0 <= pred_cls < 4 else f"Token desconocido ({pred_token})"
        is_ok = "✅ CORRECTO" if pred_cls == actual_cls else "❌ INCORRECTO"
        print(f"    Muestra {i+1}: Imagen={class_names[actual_cls]:<20} | Predicción={pred_name:<20} | {is_ok}")
        
    # 6. Reporte Final
    print("\n" + "═"*70)
    print("  📊 RESULTADOS FINALES MULTIMODAL BDA + TBV:")
    print(f"    • Pérdida Inicial:       {initial_loss:.4f} (PPL: {math.exp(min(initial_loss, 20)):.2f})")
    print(f"    • Pérdida Final:         {final_train_loss:.4f} (PPL: {math.exp(min(final_train_loss, 20)):.2f})")
    print(f"    • Pérdida en Test:       {avg_test_loss:.4f}")
    print(f"    • Precisión en Test:     {test_acc:.2f}% ({test_correct}/{test_total} aciertos)")
    print(f"    • Tiempo de entreno:     {train_time:.2f} s")
    print("═"*70)

if __name__ == "__main__":
    main()
