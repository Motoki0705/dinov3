# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        mask_k_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.lora_rank = rank
        self.lora_alpha = alpha
        self.lora_scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        if mask_k_bias:
            if out_features % 3 != 0:
                raise ValueError("Masked K-bias requires output features divisible by three")
            self.register_buffer("bias_mask", torch.empty(out_features, device=device, dtype=dtype))
        self.reset_lora_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        has_bias_mask = hasattr(linear, "bias_mask")
        if type(linear) is not nn.Linear and not has_bias_mask:
            raise TypeError(f"LoRA only supports torch.nn.Linear targets, got {type(linear).__name__}")

        module = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            mask_k_bias=has_bias_mask,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.weight = linear.weight
        module.bias = linear.bias
        if has_bias_mask:
            module.bias_mask = linear.bias_mask
        return module

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: Tensor) -> Tensor:
        bias = self.bias
        if bias is not None and hasattr(self, "bias_mask"):
            bias = bias * self.bias_mask.to(bias.dtype)
        base = F.linear(input, self.weight, bias)
        lora_input = self.lora_dropout(input)
        update = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        return base + update * self.lora_scaling


@dataclass(frozen=True)
class LoRAStats:
    replaced_modules: tuple[str, ...]
    trainable_parameters: int


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    path = module_name.split(".")
    for name in path[:-1]:
        parent = parent[int(name)] if isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, name)
    return parent, path[-1]


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str],
    freeze_backbone: bool = True,
) -> LoRAStats:
    if not target_modules:
        raise ValueError("LoRA target_modules must not be empty")

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name and any(name.endswith(target) for target in target_modules)
    ]
    if not matches:
        raise ValueError(f"No Linear modules matched LoRA targets: {list(target_modules)}")

    replaced = []
    for name, module in matches:
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target '{name}' is not a Linear module: {type(module).__name__}")
        parent, child_name = _resolve_parent(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear.from_linear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(name)

    if freeze_backbone:
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return LoRAStats(tuple(replaced), trainable_parameters)


def reset_lora_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()
