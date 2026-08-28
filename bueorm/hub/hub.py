"""
Bueorm Hub - High-Level Model Management and Multi-Format Export
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, List

from bueorm.core.serialization import save_model, load_model
from bueorm.config import BueormConfig


def export_model(
    model: nn.Module,
    output_dir: str,
    base_name: str = "model",
    formats: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Exports a trained model into all supported formats: .bueorm, .safetensors, .gguf, .pt.
    
    Args:
        model: Trained model instance
        output_dir: Directory to save exports
        base_name: Base filename prefix
        formats: List of formats (default: ['bueorm', 'safetensors', 'gguf', 'pt'])
        
    Returns:
        Dict mapping format name to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    if formats is None:
        formats = ["bueorm", "safetensors", "gguf", "pt"]
        
    saved_paths = {}
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{base_name}.{fmt}")
        saved_path = save_model(model, filepath, format=fmt)
        saved_paths[fmt] = saved_path
        
    return saved_paths
