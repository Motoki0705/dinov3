#!/usr/bin/env python3
"""Export a LoRA-SSL teacher checkpoint as a plain backbone state-dict.

The SSL pipeline (``dinov3/train/train.py``) saves an EMA *teacher* checkpoint of
shape ``{"teacher": state_dict}`` where ``state_dict`` nests the backbone under a
``backbone.`` prefix alongside the DINO/iBOT heads, and the attention
projections carry LoRA adapters (``lora_A`` / ``lora_B``) on top of frozen base
weights.

Downstream code that consumes a backbone in isolation typically expects a *plain*
backbone state-dict (no ``backbone.`` prefix, no LoRA adapters, no heads) that can
be loaded into a freshly built ``dinov3_vit*`` model with ``strict=True``. This
tool bridges the two by:

1. Unwrapping the ``teacher`` payload and keeping only ``backbone.*`` tensors.
2. Folding each LoRA update back into its base weight
   (``W <- W + (alpha / rank) * (B @ A)``) and dropping the adapter tensors.
3. Materialising a reference backbone via :mod:`dinov3.hub.backbones` and
   re-exporting *its* ``state_dict`` so the result matches the loader exactly
   (extra buffers/heads are discarded, key order is canonical).

Usage::

    PYTHONPATH=${PWD} python tools/export_lora_backbone.py \
        --teacher <OUTPUT_DIR>/eval/training_24999/teacher_checkpoint.pth \
        --output  <OUTPUT_DIR>/backbone_vitb16.pth \
        --arch    dinov3_vitb16

The ``--rank`` and ``--alpha`` values must match the ``lora`` block of the
training config that produced the checkpoint (defaults follow the supplied
``dinov3_vitb16_lora.yaml`` / ``dinov3_vitl16_lora.yaml`` configs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch

import dinov3.hub.backbones as backbones

_BACKBONE_PREFIX = "backbone."


def _load_teacher_state(path: Path) -> dict[str, torch.Tensor]:
    """Load the teacher checkpoint and return its flat state-dict."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "teacher" in payload:
        payload = payload["teacher"]
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in {path}, got {type(payload).__name__}."
        )
    return dict(payload)


def _strip_backbone(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only ``backbone.*`` entries, dropping the prefix and the heads."""
    backbone = {
        key[len(_BACKBONE_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(_BACKBONE_PREFIX)
    }
    if not backbone:
        raise RuntimeError(
            "No 'backbone.*' tensors found in the teacher checkpoint; "
            "is this a DINOv3 SSL teacher_checkpoint.pth?"
        )
    return backbone


def _merge_lora(
    state: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> tuple[dict[str, torch.Tensor], int]:
    """Fold LoRA adapters into their base weights and drop the adapter tensors."""
    lora_modules = sorted(
        {key[: -len(".lora_A")] for key in state if key.endswith(".lora_A")}
    )
    merged: dict[str, torch.Tensor] = {}
    for prefix in lora_modules:
        weight_key = f"{prefix}.weight"
        if weight_key not in state:
            raise RuntimeError(f"LoRA adapter '{prefix}' has no base weight to merge.")
        lora_a = state[f"{prefix}.lora_A"].float()
        lora_b = state[f"{prefix}.lora_B"].float()
        base = state[weight_key]
        update = (lora_b @ lora_a) * scaling
        merged[weight_key] = (base.float() + update).to(base.dtype)

    skip: set[str] = set()
    for prefix in lora_modules:
        skip.update({f"{prefix}.lora_A", f"{prefix}.lora_B", f"{prefix}.weight"})

    result = {key: value for key, value in state.items() if key not in skip}
    result.update(merged)
    return result, len(lora_modules)


def _export_via_reference(
    merged: Mapping[str, torch.Tensor],
    *,
    arch: str,
) -> dict[str, torch.Tensor]:
    """Load ``merged`` into a fresh reference backbone and re-export its state."""
    try:
        builder = getattr(backbones, arch)
    except AttributeError as error:
        raise ValueError(
            f"Unknown backbone arch '{arch}'. Expected a dinov3.hub.backbones entry "
            "such as 'dinov3_vitb16' or 'dinov3_vitl16'."
        ) from error

    backbone = builder(pretrained=False)
    result = backbone.load_state_dict(merged, strict=False)
    real_missing = [k for k in result.missing_keys if not k.endswith("bias_mask")]
    if real_missing:
        raise RuntimeError(
            "Reference backbone is missing tensors after merge "
            f"(first few): {real_missing[:8]}"
        )
    if result.unexpected_keys:
        print(
            f"[export] note: {len(result.unexpected_keys)} unexpected keys ignored "
            f"(e.g. {result.unexpected_keys[:4]})"
        )
    return backbone.state_dict()


def export_backbone(
    teacher_path: Path,
    output_path: Path,
    *,
    arch: str,
    rank: int,
    alpha: float,
) -> None:
    scaling = alpha / rank
    state = _load_teacher_state(teacher_path)
    backbone_state = _strip_backbone(state)
    merged, num_lora = _merge_lora(backbone_state, scaling=scaling)
    exported = _export_via_reference(merged, arch=arch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)
    print(
        f"[export] {arch}: merged {num_lora} LoRA module(s) at scaling={scaling:g} "
        f"-> {output_path} ({len(exported)} tensors)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        required=True,
        help="Path to the SSL teacher_checkpoint.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the plain backbone state-dict.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="dinov3_vitb16",
        help="dinov3.hub.backbones builder name (default: dinov3_vitb16).",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8).")
    parser.add_argument(
        "--alpha", type=float, default=16.0, help="LoRA alpha (default: 16)."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    export_backbone(
        args.teacher,
        args.output,
        arch=args.arch,
        rank=args.rank,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
