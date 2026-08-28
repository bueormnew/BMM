"""
Bueorm Core - Multi-Format Model Serializer and Loader
Supports:
    1. .bueorm      - Proprietary ultra-optimized, self-contained single-file container.
    2. .safetensors - Safe tensor format for secure cross-platform weight storage.
    3. .gguf        - GGUF v3 binary format for universal cross-platform deployment.
    4. .pt          - PyTorch standard state dictionary.
"""

import os
import zlib
import json
import struct
import io
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

import safetensors
import safetensors.torch
from bueorm.config import BueormConfig
from bueorm.core.registry import MODEL_REGISTRY
from bueorm.core.gguf_io import GGUFWriter, GGUFReader, GGML_TYPE_F32, GGML_TYPE_F16


BUEORM_MAGIC = b"BUEORM\x01"


def save_model(
    model: nn.Module,
    filepath: str,
    format: Optional[str] = None,
    config: Optional[BueormConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves a Bueorm model in the specified format (.bueorm, .safetensors, .gguf, .pt).
    
    Args:
        model: Model instance (with .config attribute or config argument)
        filepath: Destination file path
        format: 'bueorm', 'safetensors', 'gguf', 'pt' (auto-inferred from extension if None)
        config: Optional BueormConfig instance
        extra_metadata: Optional custom dictionary of metadata
    """
    if config is None:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("A valid BueormConfig must be provided or attached as model.config.")

    if format is None:
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        format = ext if ext in ("bueorm", "safetensors", "gguf", "pt") else "bueorm"

    metadata = {
        "model_class": model.__class__.__name__,
        "model_type": getattr(config, "model_type", "unknown"),
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
        "version": getattr(config, "version", "1.0.0"),
        "extra": extra_metadata or {}
    }

    state_dict = model.state_dict()

    if format == "bueorm":
        # .bueorm: High performance self-contained binary bundle
        meta_bytes = json.dumps(metadata).encode("utf-8")
        compressed_meta = zlib.compress(meta_bytes, level=6)
        
        # Serialize state_dict to memory buffer
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        tensor_bytes = buf.getvalue()
        
        with open(filepath, "wb") as f:
            f.write(BUEORM_MAGIC)
            f.write(struct.pack("<Q", len(compressed_meta)))
            f.write(compressed_meta)
            f.write(struct.pack("<Q", len(tensor_bytes)))
            f.write(tensor_bytes)

    elif format == "safetensors":
        # .safetensors: Safe tensors format + metadata string
        # Clone tensors to avoid shared memory pointer error when weights are tied (e.g. lm_head & tok_embeddings)
        cpu_tensors = {k: v.clone().contiguous().cpu() for k, v in state_dict.items()}
        safe_meta = {
            "bueorm_meta": json.dumps(metadata)
        }
        safetensors.torch.save_file(cpu_tensors, filepath, metadata=safe_meta)

    elif format == "gguf":
        # .gguf: Binary GGUF v3 export
        writer = GGUFWriter(filepath)
        writer.add_metadata_string("bueorm.model_class", metadata["model_class"])
        writer.add_metadata_string("bueorm.config", json.dumps(metadata["config"]))
        writer.add_metadata_string("general.architecture", metadata["model_type"])
        
        for name, tensor in state_dict.items():
            writer.add_tensor(name, tensor.clone(), ggml_type=GGML_TYPE_F32)
        writer.write()

    elif format == "pt":
        # .pt: Standard PyTorch dictionary
        payload = {
            "state_dict": state_dict,
            "metadata": metadata,
            "config": config.to_dict() if hasattr(config, "to_dict") else {}
        }
        torch.save(payload, filepath)
    else:
        raise ValueError(f"Unsupported format '{format}'. Choose from 'bueorm', 'safetensors', 'gguf', 'pt'.")

    return filepath


def load_model(
    filepath: str,
    device: Optional[Union[str, torch.device]] = "cpu",
    model_class: Optional[Any] = None,
    config: Optional[BueormConfig] = None
) -> nn.Module:
    """
    Loads a model from a .bueorm, .safetensors, .gguf, or .pt file.
    Automatically reconstructs the model architecture, hyperparameters, and weights.
    """
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")

    if ext == "bueorm":
        with open(filepath, "rb") as f:
            magic = f.read(len(BUEORM_MAGIC))
            if magic != BUEORM_MAGIC:
                raise ValueError(f"Invalid .bueorm file: Header magic mismatch.")
                
            meta_len = struct.unpack("<Q", f.read(8))[0]
            compressed_meta = f.read(meta_len)
            metadata = json.loads(zlib.decompress(compressed_meta).decode("utf-8"))
            
            tensor_len = struct.unpack("<Q", f.read(8))[0]
            tensor_bytes = f.read(tensor_len)
            buf = io.BytesIO(tensor_bytes)
            state_dict = torch.load(buf, map_location=device, weights_only=True)
            
        if config is None:
            config = BueormConfig.from_dict(metadata["config"])
        cls_name = metadata.get("model_class")

    elif ext == "safetensors":
        state_dict = safetensors.torch.load_file(filepath, device=str(device))
        with safetensors.safe_open(filepath, framework="pt", device=str(device)) as f:
            raw_meta = f.metadata() or {}
        if "bueorm_meta" in raw_meta:
            metadata = json.loads(raw_meta["bueorm_meta"])
            if config is None:
                config = BueormConfig.from_dict(metadata["config"])
            cls_name = metadata.get("model_class")
        else:
            cls_name = None
            metadata = {}

    elif ext == "gguf":
        reader = GGUFReader(filepath)
        state_dict = {k: v.to(device) for k, v in reader.tensors.items()}
        meta_cfg = reader.metadata.get("bueorm.config")
        cls_name = reader.metadata.get("bueorm.model_class")
        if config is None and meta_cfg:
            config = BueormConfig.from_dict(json.loads(meta_cfg))
        metadata = reader.metadata

    elif ext == "pt":
        payload = torch.load(filepath, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            metadata = payload.get("metadata", {})
            if config is None:
                cfg_data = payload.get("config", metadata.get("config", {}))
                config = BueormConfig.from_dict(cfg_data)
            cls_name = metadata.get("model_class")
        else:
            state_dict = payload
            cls_name = None
            metadata = {}
    else:
        raise ValueError(f"Unknown file extension '{ext}'. Expected .bueorm, .safetensors, .gguf, or .pt.")

    # Determine Model Class from Registry if not explicitly provided
    if model_class is None:
        if cls_name and cls_name in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY.get(cls_name)
        elif config is not None and config.model_type in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY.get(config.model_type)
        else:
            from bueorm.models.factory import BueormModel
            model = BueormModel.from_config(config)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            return model

    # Instantiate model with config
    model = model_class(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model
