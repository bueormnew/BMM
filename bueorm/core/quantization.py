"""
Bueorm Core - Model Quantization Engine
Supports dynamic int8 quantization, weight-only int8 quantization,
precision casting (fp16, bf16, fp32), and memory footprint profiling.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Any, Optional


class Int8Linear(nn.Module):
    """
    Symmetric 8-bit weight-quantized linear layer:
        y = x @ (W_int8 * scale)^T + bias
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> "Int8Linear":
        out_f, in_f = float_linear.weight.shape
        has_bias = float_linear.bias is not None
        layer = cls(in_f, out_f, bias=has_bias)
        
        # Symmetric per-channel weight quantization
        w = float_linear.weight.data.float()
        max_val = torch.amax(torch.abs(w), dim=-1, keepdim=True)
        scale = torch.clamp(max_val / 127.0, min=1e-8)
        w_int8 = torch.clamp(torch.round(w / scale), min=-128, max=127).to(torch.int8)
        
        layer.weight_int8.copy_(w_int8)
        layer.scale.copy_(scale)
        if has_bias:
            layer.bias.data.copy_(float_linear.bias.data)
            
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on the fly during matrix multiplication
        w_dequant = (self.weight_int8.to(x.dtype) * self.scale.to(x.dtype))
        out = torch.matmul(x, w_dequant.t())
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


def quantize_model(
    model: nn.Module,
    mode: str = "int8_weight",
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Quantizes a model for efficient deployment and reduced VRAM footprint.
    
    Args:
        model: nn.Module instance
        mode: Quantization mode:
              - 'int8_weight': Replaces Linear layers with Int8Linear.
              - 'int8_dynamic': Uses torch.quantization.quantize_dynamic (CPU int8).
              - 'fp16' / 'float16': Casts all parameters to float16.
              - 'bf16' / 'bfloat16': Casts all parameters to bfloat16.
        target_modules: Optional list of module class names to quantize.
        
    Returns:
        Quantized model.
    """
    if mode in ("fp16", "float16"):
        return model.half()
    elif mode in ("bf16", "bfloat16"):
        return model.to(dtype=torch.bfloat16)
    elif mode == "int8_dynamic":
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    elif mode == "int8_weight":
        def _replace_linear(module: nn.Module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Skip 1-dim or very small projection layers if requested
                    if child.in_features >= 16 and child.out_features >= 16:
                        setattr(module, name, Int8Linear.from_float(child))
                else:
                    _replace_linear(child)
        _replace_linear(model)
        return model
    else:
        raise ValueError(f"Unknown quantization mode '{mode}'. Choose from 'int8_weight', 'int8_dynamic', 'fp16', 'bf16'.")


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """Calculates total parameter and buffer memory size in Megabytes (MB)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_mb = (param_size + buffer_size) / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())
    return {
        "num_parameters": num_params,
        "param_memory_mb": param_size / (1024 ** 2),
        "buffer_memory_mb": buffer_size / (1024 ** 2),
        "total_memory_mb": total_mb
    }
