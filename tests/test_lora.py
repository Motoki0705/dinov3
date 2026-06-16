import pytest
import torch
from torch import nn

from dinov3.layers.attention import LinearKMaskedBias
from dinov3.models.lora import LoRALinear, apply_lora, reset_lora_parameters


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "qkv": nn.Linear(4, 12),
                                "proj": nn.Linear(4, 4),
                            }
                        ),
                        "mlp": nn.ModuleDict({"fc1": nn.Linear(4, 8)}),
                    }
                )
            ]
        )


def test_apply_lora_replaces_targets_and_freezes_base_weights():
    model = TinyBackbone()
    stats = apply_lora(
        model,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["attn.qkv", "attn.proj"],
    )

    assert stats.replaced_modules == ("blocks.0.attn.qkv", "blocks.0.attn.proj")
    assert isinstance(model.blocks[0]["attn"]["qkv"], LoRALinear)
    assert model.blocks[0]["attn"]["qkv"].weight.requires_grad is False
    assert model.blocks[0]["attn"]["qkv"].lora_A.requires_grad is True
    assert model.blocks[0]["mlp"]["fc1"].weight.requires_grad is False


def test_lora_starts_as_identity_update_and_can_be_reset():
    linear = nn.Linear(4, 4)
    model = nn.Sequential(linear)
    original = linear(torch.ones(2, 4))
    apply_lora(model, rank=2, alpha=2, dropout=0.0, target_modules=["0"])

    assert torch.equal(model(torch.ones(2, 4)), original)
    nn.init.ones_(model[0].lora_B)
    reset_lora_parameters(model)
    assert torch.count_nonzero(model[0].lora_B) == 0


def test_apply_lora_rejects_missing_targets():
    with pytest.raises(ValueError, match="No Linear modules matched"):
        apply_lora(TinyBackbone(), rank=2, alpha=2, dropout=0.0, target_modules=["missing"])


def test_lora_preserves_masked_k_bias():
    model = nn.Sequential(LinearKMaskedBias(4, 12, bias=True))
    model[0].bias_mask.fill_(1)
    model[0].bias_mask[4:8].fill_(0)
    apply_lora(model, rank=2, alpha=2, dropout=0.0, target_modules=["0"])

    assert isinstance(model[0], LoRALinear)
    assert torch.equal(model[0].bias_mask[4:8], torch.zeros(4))
