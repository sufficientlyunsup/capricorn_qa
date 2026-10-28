"""Factories for constructing complete CAPRICORN models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from capricorn.bin.common import DEFAULT_QWEN3_LAYERS, MODEL_DIR

from .components import CapricornModel, FusionLayer
from .construct import make_output_layer, make_qwen3_base
from .schema import OutputSchema


class CapricornFactory:
    """Build CAPRICORN from a schema and a concatenated-sequence backbone."""

    def __init__(
        self,
        schema: OutputSchema,
        *,
        model_path: str | Path = MODEL_DIR,
        nlayers: int = DEFAULT_QWEN3_LAYERS,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        device_map: str | Mapping[str, Any] | None = "cuda",
        n_attention_layers: int = 2,
        intermediate_size: int | None = None,
    ) -> None:
        self.schema = schema
        self.model_path = model_path
        self.nlayers = nlayers
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None
        self.device_map = device_map
        self.n_attention_layers = n_attention_layers
        self.intermediate_size = intermediate_size
        self.model: CapricornModel | None = None

    def _build(self) -> CapricornModel:
        backbone = make_qwen3_base(
            nlayers=self.nlayers,
            model_path=self.model_path,
            dtype=self.dtype,
            device_map=str(self.device) if self.device is not None else self.device_map,
        )
        config = backbone.config
        fusion = FusionLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            n_attention_layers=self.n_attention_layers,
            intermediate_size=(
                config.intermediate_size
                if self.intermediate_size is None
                else self.intermediate_size
            ),
        )
        fusion_device = self.device or next(backbone.parameters()).device
        # Keep the added fusion stack in float32. Training explicitly promotes
        # this layer to float32 because its attention/FFN activations are more
        # numerically sensitive than the frozen half-precision backbone.
        fusion = fusion.to(device=fusion_device, dtype=torch.float32)
        self.model = CapricornModel(
            backbone=backbone,
            fusion_layer=fusion,
            output_layer=make_output_layer(self.schema),
        )
        return self.model

    def build_trainable_capricorn(self) -> CapricornModel:
        """Build a fresh model; inputs are assembled by the data collator."""
        return self._build()

    def build_from_saved_file(self, path: str | Path) -> CapricornModel:
        """Build a model and load a plain or wrapped state dictionary."""
        model = self._build()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, Mapping):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, Mapping):
            raise TypeError("saved file must contain a model state-dict mapping")
        model.load_state_dict(checkpoint)
        return model
