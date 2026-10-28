"""Composable neural-network components for CAPRICORN."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class _GatedFeedForward(nn.Module):
    """Qwen-style gated feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class FusionLayer(nn.Module):
    """Apply self-attention to one concatenated context/question sequence."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        n_attention_layers: int = 2,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_heads < 1 or hidden_size % num_heads:
            raise ValueError("num_heads must be positive and divide hidden_size")
        if n_attention_layers < 1:
            raise ValueError("at least one self-attention layer is required")
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        if intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")

        self.attention = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=False)
            for _ in range(n_attention_layers)
        )
        self.attention_norm = nn.ModuleList(
            nn.RMSNorm(hidden_size, eps=1e-6) for _ in range(n_attention_layers)
        )
        self.feed_forward = nn.ModuleList(
            _GatedFeedForward(hidden_size, intermediate_size)
            for _ in range(n_attention_layers)
        )
        self.feed_forward_norm = nn.ModuleList(
            nn.RMSNorm(hidden_size, eps=1e-6) for _ in range(n_attention_layers)
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return one raw logit for each sequence in ``hidden``."""
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, length, hidden]")
        if hidden.shape[-1] != self.output.in_features:
            raise ValueError("hidden size does not match the fusion layer")
        if attention_mask is not None:
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape != hidden.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, sequence_length]")
            attention_mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        padding_mask = None if attention_mask is None else ~attention_mask

        for attention, norm, feed_forward, feed_forward_norm in zip(
            self.attention, self.attention_norm, self.feed_forward, self.feed_forward_norm
        ):
            normed = norm(hidden)
            update, _ = attention(
                normed, normed, normed,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            hidden = hidden + update
            hidden = hidden + feed_forward(feed_forward_norm(hidden))

        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output(pooled)


class _CapricornModelWithLogits(nn.Module):
    """Encode concatenated context/question candidates and return raw logits."""

    def __init__(
        self,
        backbone: nn.Module,
        fusion_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fusion_layer = fusion_layer
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "_CapricornModelWithLogits":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Return logits shaped ``[batch, candidates]``."""
        if "input_ids" not in inputs:
            raise ValueError("inputs must contain input_ids")
        input_ids = inputs["input_ids"]
        if input_ids.ndim not in (2, 3):
            raise ValueError("input_ids must have shape [batch, tokens] or [batch, candidates, tokens]")
        candidate_shape = input_ids.shape[:2] if input_ids.ndim == 3 else (input_ids.shape[0], 1)
        if input_ids.ndim == 3:
            flat_inputs = {
                name: value.reshape(-1, value.shape[-1])
                if isinstance(value, Tensor) and value.ndim == 3 else value
                for name, value in inputs.items()
            }
        else:
            flat_inputs = dict(inputs)

        backbone_device = next(self.backbone.parameters()).device
        flat_inputs = {
            name: value.to(backbone_device) if isinstance(value, Tensor) else value
            for name, value in flat_inputs.items()
        }
        attention_mask = flat_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(flat_inputs["input_ids"], dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)
        with torch.no_grad():
            result = self.backbone(**flat_inputs, return_dict=True)
        hidden = result.last_hidden_state
        fusion_parameter = next(self.fusion_layer.parameters(), None)
        if fusion_parameter is not None:
            hidden = hidden.to(device=fusion_parameter.device, dtype=fusion_parameter.dtype)
            attention_mask = attention_mask.to(fusion_parameter.device)
        logits = self.fusion_layer(hidden, attention_mask).squeeze(-1)
        return logits.reshape(candidate_shape)


class CapricornModel(nn.Module):
    """Inference wrapper that maps raw candidate logits to schema outputs."""

    def __init__(
        self,
        backbone: nn.Module,
        fusion_layer: nn.Module,
        output_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.logits_model = _CapricornModelWithLogits(backbone, fusion_layer)
        self.output_layer = output_layer

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Return schema-specific probabilities for a concatenated batch."""
        return self.output_layer(self.logits_model(inputs))
