"""
Bueorm Trainer - Production Training Framework
Supports language models, vision reconstruction (TBV), multimodal VLMs,
Mixture of Experts (MoE) load balancing loss, gradient accumulation, and multi-format checkpointing.
"""

import os
import math
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from torch.utils.data import DataLoader, Dataset

from bueorm.core.serialization import save_model


@dataclass
class TrainingArguments:
    """
    Hyperparameters and runtime settings for training Bueorm models.
    """
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    save_format: str = "bueorm"  # "bueorm", "safetensors", "gguf", "pt"
    device: Optional[str] = None
    seed: int = 42


class Trainer:
    """
    Universal Trainer for Bueorm models.
    """
    def __init__(
        self,
        model: nn.Module,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.args = args or TrainingArguments()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self._default_collator

        # Device setup
        if self.args.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)
            
        self.model.to(self.device)

        # Optimizer setup
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        os.makedirs(self.args.output_dir, exist_ok=True)

    def _default_collator(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        if isinstance(batch[0], dict):
            collated = {}
            for k in batch[0].keys():
                collated[k] = torch.stack([item[k] for item in batch])
            return collated
        elif isinstance(batch[0], (tuple, list)):
            return {f"input_{i}": torch.stack([item[i] for item in batch]) for i in range(len(batch[0]))}
        elif isinstance(batch[0], torch.Tensor):
            return {"input_ids": torch.stack(batch)}
        else:
            return {"input_ids": torch.tensor(batch)}

    def train(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Executes complete training loop.
        """
        if dataloader is None:
            if self.train_dataset is None:
                raise ValueError("Either train_dataset or dataloader must be provided to Trainer.")
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=getattr(self.args, "batch_size", 4),
                shuffle=True,
                collate_fn=self.data_collator
            )

        self.model.train()
        global_step = 0
        total_loss = 0.0
        start_time = time.time()

        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass — supports legacy + generative (text->image, unified)
                if "target_images" in batch and "input_ids" in batch and hasattr(self.model, "generate_image"):
                    # Generative training: text -> image (TTI, language_with_image, generative_vlm)
                    input_ids = batch["input_ids"]
                    target_images = batch["target_images"]
                    targets = batch.get("targets", None)
                    images = batch.get("images", None)  # optional context image for VLM
                    # Dispatch per model signature
                    if images is not None:
                        # Generative VLM: can have both images context + target_images
                        result = self.model(input_ids=input_ids, images=images, targets=targets, target_images=target_images)
                    elif targets is not None:
                        # Language with image gen: joint text+image loss
                        result = self.model(input_ids=input_ids, targets=targets, target_images=target_images)
                    else:
                        # Pure TTI: text->image only
                        # TextToImageModel returns (image_pred, loss, z)
                        result = self.model(input_ids=input_ids, target_images=target_images)
                    # result is (logits/image_pred, loss, ...), loss at index 1
                    loss = result[1]
                    if loss is None:
                        raise ValueError("Generative model returned None loss — check batch contains valid target_images")
                elif hasattr(self.model, "reconstruct") and "images" in batch and "input_ids" not in batch:
                    # Vision TBV Training (pure)
                    images = batch["images"]
                    rec = self.model.reconstruct(images)
                    loss = nn.functional.mse_loss(rec, images)
                elif "images" in batch and "input_ids" in batch:
                    # Multimodal VLM Training (legacy, also supports generative VLM without target_images)
                    images = batch["images"]
                    input_ids = batch["input_ids"]
                    targets = batch.get("targets", input_ids)
                    target_images = batch.get("target_images", None)
                    if target_images is not None and hasattr(self.model, "generate_image"):
                        _, loss, _ = self.model(input_ids=input_ids, images=images, targets=targets, target_images=target_images)
                    else:
                        _, loss, _ = self.model(input_ids=input_ids, images=images, targets=targets)
                else:
                    # Language Model Training (including generative language without target_images)
                    input_ids = batch.get("input_ids", next(iter(batch.values())))
                    targets = batch.get("targets", input_ids)
                    target_images = batch.get("target_images", None)
                    if target_images is not None and hasattr(self.model, "generate_image"):
                        _, loss, _ = self.model(input_ids=input_ids, targets=targets, target_images=target_images)
                    else:
                        _, loss, _ = self.model(input_ids=input_ids, targets=targets)

                # Gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.args.gradient_accumulation_steps

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.args.logging_steps == 0:
                        avg_loss = total_loss / self.args.logging_steps
                        total_loss = 0.0

                    # Save checkpoint
                    if global_step % self.args.save_steps == 0:
                        self.save_checkpoint(step=global_step)

        # Final save
        final_path = self.save_checkpoint(step=global_step, is_final=True)
        elapsed = time.time() - start_time
        return {"global_step": global_step, "elapsed_time": elapsed, "checkpoint": final_path}

    def save_checkpoint(self, step: int, is_final: bool = False) -> str:
        """Saves checkpoint in configured format (.bueorm, .safetensors, .gguf, .pt)."""
        filename = f"model_final.{self.args.save_format}" if is_final else f"checkpoint_{step}.{self.args.save_format}"
        save_path = os.path.join(self.args.output_dir, filename)
        save_model(self.model, save_path, format=self.args.save_format)
        return save_path
