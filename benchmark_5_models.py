"""
Bueorm Model Benchmark - Comparación de 5 Arquitecturas
Mide: velocidad de entrenamiento, pérdida, perplejidad, velocidad de generación, memoria.
"""

import time
import math
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import bueorm
from bueorm.config import BueormConfig, MoEConfig

# ─── Configuración Común ──────────────────────────────────────────────────────
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 8
VOCAB_SIZE  = 1024
MAX_SEQ_LEN = 128
BATCH_SIZE  = 16
SEQ_LEN     = 64
NUM_STEPS   = 80        # Pasos de entrenamiento por modelo
GEN_TOKENS  = 32       # Tokens generados por muestra
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Dataset Sintético ───────────────────────────────────────────────────────
torch.manual_seed(42)
x_data = torch.randint(0, VOCAB_SIZE, (NUM_STEPS * BATCH_SIZE, SEQ_LEN))
y_data = torch.randint(0, VOCAB_SIZE, (NUM_STEPS * BATCH_SIZE, SEQ_LEN))
dataset = TensorDataset(x_data, y_data)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ─── Utilidades ──────────────────────────────────────────────────────────────
def get_mem_mb():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def get_peak_mem_mb():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def reset_peak():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

def benchmark_model(name: str, model: nn.Module):
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if DEVICE == "cuda":
        reset_peak()

    mem_before_mb = get_mem_mb()
    n_params      = count_params_m(model)
    losses        = []
    step_times    = []

    print(f"\n{'═'*60}")
    print(f"  Modelo: {name}  |  Parámetros: {n_params:.2f}M")
    print(f"{'═'*60}")

    total_t0 = time.perf_counter()

    for step, (xb, yb) in enumerate(loader):
        if step >= NUM_STEPS:
            break
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits, loss, _ = model(xb, targets=yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            avg_l = sum(losses[-20:]) / 20
            ppl   = math.exp(min(avg_l, 20))
            speed = BATCH_SIZE * SEQ_LEN / (sum(step_times[-20:]))
            print(f"  Paso {step+1:4d}/{NUM_STEPS}  |  Pérdida={avg_l:.4f}  |  PPL={ppl:.1f}  |  {speed:,.0f} tok/s")

    total_time = time.perf_counter() - total_t0
    mem_after_mb  = get_mem_mb()
    peak_mem_mb   = get_peak_mem_mb()

    final_loss = sum(losses[-10:]) / 10
    final_ppl  = math.exp(min(final_loss, 20))
    avg_step_s = sum(step_times) / len(step_times)
    throughput = BATCH_SIZE * SEQ_LEN / avg_step_s

    # ─── Velocidad de Generación ─────────────────────────────────────────
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (4, 8)).to(DEVICE)
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(prompt, max_new_tokens=GEN_TOKENS, temperature=0.0)
    gen_time_s = time.perf_counter() - t_gen_start
    gen_tokens_per_s = 4 * GEN_TOKENS / gen_time_s

    print(f"\n  {'─'*56}")
    print(f"  RESULTADOS FINALES:")
    print(f"    Parámetros:            {n_params:.3f} M")
    print(f"    Tiempo total entreno:  {total_time:.2f} s")
    print(f"    Tiempo por paso:       {avg_step_s*1000:.2f} ms")
    print(f"    Throughput entreno:    {throughput:,.0f} tok/s")
    print(f"    Pérdida final:         {final_loss:.4f}")
    print(f"    Perplejidad final:     {final_ppl:.2f}")
    print(f"    Velocidad generación:  {gen_tokens_per_s:.1f} tok/s")
    print(f"    Memoria Antes:         {mem_before_mb:.1f} MB")
    print(f"    Memoria Después:       {mem_after_mb:.1f} MB")
    print(f"    Memoria Pico:          {peak_mem_mb:.1f} MB")
    print(f"    Crecimiento Memoria:   {mem_after_mb - mem_before_mb:.1f} MB")
    print(f"  {'─'*56}")

    result = {
        "name":            name,
        "params_m":        n_params,
        "total_train_s":   total_time,
        "avg_step_ms":     avg_step_s * 1000,
        "throughput":      throughput,
        "final_loss":      final_loss,
        "final_ppl":       final_ppl,
        "gen_tok_per_s":   gen_tokens_per_s,
        "mem_before_mb":   mem_before_mb,
        "mem_after_mb":    mem_after_mb,
        "peak_mem_mb":     peak_mem_mb,
        "mem_growth_mb":   mem_after_mb - mem_before_mb,
    }
    return result


# ─── Definición de los 5 Modelos ─────────────────────────────────────────────
def build_model_1():
    cfg = BueormConfig(
        model_type="transformer",
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        n_layers=N_LAYERS, vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN, dropout=0.0
    )
    return bueorm.TransformerLM(cfg)

def build_model_2():
    cfg = BueormConfig(
        model_type="bda",
        d_model=D_MODEL, n_heads=N_HEADS, d_k=32, d_v=32,
        n_layers=N_LAYERS, vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN, dropout=0.0
    )
    return bueorm.BDALanguageModel(cfg)

def build_model_3():
    cfg = BueormConfig(
        model_type="hybrid",
        hybrid_pattern="5bda:1attn",
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        d_k=32, d_v=32,
        n_layers=6, vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN, dropout=0.0
    )
    return bueorm.HybridLanguageModel(cfg)

def build_model_4():
    cfg = BueormConfig(
        model_type="hybrid",
        hybrid_pattern="3bda:1attn",
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        d_k=32, d_v=32,
        n_layers=8, vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN, dropout=0.0
    )
    return bueorm.HybridLanguageModel(cfg)

def build_model_5():
    cfg = BueormConfig(
        model_type="hybrid",
        hybrid_pattern="3bda:1attn",
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_HEADS,
        d_k=32, d_v=32,
        n_layers=8, vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN, dropout=0.0,
        use_moe=True,
        moe_config=MoEConfig(num_experts=3, top_k=1, aux_loss_coef=0.01)
    )
    return bueorm.HybridLanguageModel(cfg)


# ─── Ejecución del Benchmark ─────────────────────────────────────────────────
MODELS = [
    ("Modelo 1 — Transformer Puro",         build_model_1),
    ("Modelo 2 — BDA Puro",                 build_model_2),
    ("Modelo 3 — BDA+ATT  (5:1)",           build_model_3),
    ("Modelo 4 — BDA+ATT  (3:1)",           build_model_4),
    ("Modelo 5 — BDA+ATT+MoE (3:1, 3E-1A)", build_model_5),
]

print("\n" + "═"*60)
print("   BUEORM BENCHMARK — 5 ARQUITECTURAS")
print(f"   Device: {DEVICE.upper()}  |  Pasos: {NUM_STEPS}  |  Batch: {BATCH_SIZE}")
print("═"*60)

results = []
for name, builder_fn in MODELS:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = builder_fn()
    r = benchmark_model(name, model)
    results.append(r)
    del model
    gc.collect()

# ─── Tabla Resumen ───────────────────────────────────────────────────────────
print("\n\n" + "═"*108)
print(f"  {'TABLA COMPARATIVA FINAL':^104}")
print("═"*108)
header = (
    f"  {'Modelo':<30} {'Params':>7} {'Loss':>7} {'PPL':>7} "
    f"{'Train tok/s':>12} {'Gen tok/s':>10} {'Tiempo(s)':>9} {'Pico Mem(MB)':>12}"
)
print(header)
print("─"*108)
for r in results:
    print(
        f"  {r['name']:<30} {r['params_m']:>6.2f}M {r['final_loss']:>7.4f} {r['final_ppl']:>7.2f} "
        f"{r['throughput']:>12,.0f} {r['gen_tok_per_s']:>10.1f} {r['total_train_s']:>9.2f} {r['peak_mem_mb']:>12.1f}"
    )
print("═"*108)
