"""
Bueorm Core - Pure Python GGUF v3 Binary Writer and Parser
Implements self-contained export and loading of models in standard GGUF binary format.
"""

import struct
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Union
from io import BytesIO


# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I8 = 8


class GGUFWriter:
    """
    Writes PyTorch state_dicts and configuration metadata into GGUF v3 binary format.
    """
    def __init__(self, filepath: str, alignment: int = 32):
        self.filepath = filepath
        self.alignment = alignment
        self.kv_metadata: Dict[str, Tuple[int, Any]] = {}
        self.tensors: List[Tuple[str, torch.Tensor, int]] = []

    def add_metadata_string(self, key: str, value: str):
        self.kv_metadata[key] = (GGUF_TYPE_STRING, value)

    def add_metadata_uint32(self, key: str, value: int):
        self.kv_metadata[key] = (GGUF_TYPE_UINT32, int(value))

    def add_metadata_float32(self, key: str, value: float):
        self.kv_metadata[key] = (GGUF_TYPE_FLOAT32, float(value))

    def add_metadata_bool(self, key: str, value: bool):
        self.kv_metadata[key] = (GGUF_TYPE_BOOL, bool(value))

    def add_tensor(self, name: str, tensor: torch.Tensor, ggml_type: int = GGML_TYPE_F32):
        self.tensors.append((name, tensor.detach().cpu().contiguous(), ggml_type))

    def _write_string(self, f, s: str):
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    def _write_value(self, f, vtype: int, val: Any):
        f.write(struct.pack("<I", vtype))
        if vtype == GGUF_TYPE_STRING:
            self._write_string(f, val)
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", val))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", val))
        elif vtype == GGUF_TYPE_BOOL:
            f.write(struct.pack("<?", val))
        elif vtype == GGUF_TYPE_INT32:
            f.write(struct.pack("<i", val))
        elif vtype == GGUF_TYPE_UINT64:
            f.write(struct.pack("<Q", val))
        else:
            self._write_string(f, str(val))

    def write(self):
        with open(self.filepath, "wb") as f:
            # 1. Magic 'GGUF' (0x46554747) and Version 3
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            
            # 2. Tensor count and Metadata KV count
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_metadata)))
            
            # 3. Write Key-Value Metadata
            for key, (vtype, val) in self.kv_metadata.items():
                self._write_string(f, key)
                self._write_value(f, vtype, val)
                
            # 4. Prepare Tensor Info Header & Data Buffers
            tensor_headers = []
            tensor_data_buffers = []
            current_offset = 0
            
            for name, tensor, ggml_type in self.tensors:
                if ggml_type == GGML_TYPE_F32:
                    raw_bytes = tensor.float().numpy().tobytes()
                elif ggml_type == GGML_TYPE_F16:
                    raw_bytes = tensor.half().numpy().tobytes()
                elif ggml_type == GGML_TYPE_I8:
                    raw_bytes = tensor.to(torch.int8).numpy().tobytes()
                else:
                    raw_bytes = tensor.float().numpy().tobytes()
                    ggml_type = GGML_TYPE_F32

                # Alignment padding
                pad = (self.alignment - (current_offset % self.alignment)) % self.alignment
                current_offset += pad
                offset = current_offset
                current_offset += len(raw_bytes)
                
                dims = list(tensor.shape)
                tensor_headers.append((name, dims, ggml_type, offset))
                tensor_data_buffers.append((pad, raw_bytes))
                
            # 5. Write Tensor Info Headers
            for name, dims, ggml_type, offset in tensor_headers:
                self._write_string(f, name)
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", offset))
                
            # 6. Pad to global tensor data alignment
            pos = f.tell()
            pad_header = (self.alignment - (pos % self.alignment)) % self.alignment
            if pad_header > 0:
                f.write(b"\x00" * pad_header)
                
            # 7. Write Tensor Binary Payloads
            for pad, raw_bytes in tensor_data_buffers:
                if pad > 0:
                    f.write(b"\x00" * pad)
                f.write(raw_bytes)


class GGUFReader:
    """
    Parses GGUF binary files and extracts metadata and tensors into PyTorch format.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, torch.Tensor] = {}
        self.read()

    def _read_string(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, f, vtype: int) -> Any:
        if vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif vtype == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUF_TYPE_BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif vtype == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        else:
            return f.read(4)

    def read(self):
        with open(self.filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Invalid GGUF file: expected magic 'GGUF', got {magic}")
                
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            
            # Read KV metadata
            for _ in range(n_kv):
                key = self._read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                val = self._read_value(f, vtype)
                self.metadata[key] = val
                
            # Read Tensor Info
            tensor_infos = []
            for _ in range(n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensor_infos.append((name, dims, ggml_type, offset))
                
            # Data block starts at aligned position
            pos = f.tell()
            alignment = 32
            pad = (alignment - (pos % alignment)) % alignment
            data_start = pos + pad
            
            # Read tensor payloads
            for name, dims, ggml_type, offset in tensor_infos:
                f.seek(data_start + offset)
                num_elements = int(np.prod(dims)) if len(dims) > 0 else 1
                
                if ggml_type == GGML_TYPE_F32:
                    raw = f.read(num_elements * 4)
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(dims)
                    self.tensors[name] = torch.from_numpy(arr.copy())
                elif ggml_type == GGML_TYPE_F16:
                    raw = f.read(num_elements * 2)
                    arr = np.frombuffer(raw, dtype=np.float16).reshape(dims)
                    self.tensors[name] = torch.from_numpy(arr.copy()).float()
                elif ggml_type == GGML_TYPE_I8:
                    raw = f.read(num_elements * 1)
                    arr = np.frombuffer(raw, dtype=np.int8).reshape(dims)
                    self.tensors[name] = torch.from_numpy(arr.copy())
                else:
                    raw = f.read(num_elements * 4)
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(dims)
                    self.tensors[name] = torch.from_numpy(arr.copy())
