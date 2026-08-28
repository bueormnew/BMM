"""
Bueorm Core - Universal Inference Engine & Pipeline
Provides high-level, unified inference across Language Models (BDA/Transformer/Hybrid/MoE),
Vision Models (TBV), and Multimodal VLMs.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, List, Dict, Any


class InferencePipeline:
    """
    Unified inference pipeline for all Bueorm models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        inputs: Optional[Union[str, torch.Tensor, List[int]]] = None,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Universal inference call.
        
        - If language model: generates text/token sequence.
        - If vision model (TBV): encodes, decodes or reconstructs images.
        - If VLM: conditioned generation from image + text prompt.
        """
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        
        # 1. Vision Model (TBV)
        if model_type == "tbv" or hasattr(self.model, "reconstruct"):
            if images is None and isinstance(inputs, torch.Tensor):
                images = inputs
            if images is None:
                raise ValueError("Images tensor must be provided for TBV vision model.")
            images = images.to(self.device)
            mode = kwargs.get("mode", "reconstruct")
            if mode == "encode":
                return self.model.encode(images)
            elif mode == "decode":
                return self.model.decode(images)
            elif mode == "extract_features":
                return self.model.extract_features(images, return_tokens=kwargs.get("return_tokens", True))
            else:
                return self.model.reconstruct(images)

        # 2. Multimodal VLM
        elif getattr(getattr(self.model, "config", None), "is_multimodal", False) or hasattr(self.model, "generate_vlm"):
            if isinstance(inputs, str):
                if self.tokenizer is not None:
                    input_ids = torch.tensor([self.tokenizer.encode(inputs)], device=self.device)
                else:
                    # Synthetic ASCII fallback tokens
                    input_ids = torch.tensor([[ord(c) % 1000 for c in inputs]], device=self.device)
            elif isinstance(inputs, torch.Tensor):
                input_ids = inputs.to(self.device)
            elif isinstance(inputs, list):
                input_ids = torch.tensor([inputs], device=self.device)
            else:
                input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)

            if images is not None:
                images = images.to(self.device)
                
            return self.model.generate(
                prompt_ids=input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # 3. Language Model (BDA, Transformer, Hybrid, MoE)
        else:
            if isinstance(inputs, str):
                if self.tokenizer is not None:
                    input_ids = torch.tensor([self.tokenizer.encode(inputs)], device=self.device)
                else:
                    input_ids = torch.tensor([[ord(c) % 1000 for c in inputs]], device=self.device)
            elif isinstance(inputs, torch.Tensor):
                input_ids = inputs.to(self.device)
            elif isinstance(inputs, list):
                input_ids = torch.tensor([inputs], device=self.device)
            else:
                raise ValueError("Valid text prompt, token list, or tensor must be provided.")

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            return self.model.generate(
                prompt_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )


def pipeline(
    model_or_path: Union[str, nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    tokenizer: Optional[Any] = None
) -> InferencePipeline:
    """Convenience function to instantiate an InferencePipeline from a model or checkpoint path."""
    if isinstance(model_or_path, str):
        from bueorm.core.serialization import load_model
        model = load_model(model_or_path, device=device)
    else:
        model = model_or_path
    return InferencePipeline(model=model, device=device, tokenizer=tokenizer)
