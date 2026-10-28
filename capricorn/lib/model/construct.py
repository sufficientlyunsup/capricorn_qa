"""Factories for constructing configurable CAPRICORN model layers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from capricorn.bin.common import DEFAULT_QWEN3_LAYERS, MODEL_DIR
from .schema import OutputSchema


def make_qwen3_base(
    nlayers: int = DEFAULT_QWEN3_LAYERS,
    *,
    model_path: str | Path = MODEL_DIR,
    dtype: torch.dtype = torch.float16,
    device_map: str | Mapping[str, Any] | None = "cuda",
) -> nn.Module:
    """Load the Qwen3 backbone, retaining only its first ``nlayers`` blocks.

    The checkpoint is loaded from ``model_path`` and its causal-language-model
    wrapper is removed, leaving the transformer backbone suitable for use as a
    CAPRICORN input layer.  The final normalization layer is retained.  This
    function loads the checkpoint before truncating the block list, so it does
    not reduce peak checkpoint-loading memory.

    Args:
        nlayers: Number of initial transformer blocks to retain.
        model_path: Local Qwen3 checkpoint directory.
        dtype: Parameter dtype used while loading the checkpoint.
        device_map: Hugging Face device map, or ``None`` for the default.
    """
    if not isinstance(nlayers, int) or isinstance(nlayers, bool) or nlayers < 1:
        raise ValueError("nlayers must be a positive integer")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
    )
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise AttributeError("Loaded Qwen3 checkpoint has no transformer layers")
    if nlayers > len(layers):
        raise ValueError(
            f"nlayers={nlayers} exceeds the checkpoint's {len(layers)} layers"
        )

    backbone.layers = nn.ModuleList(list(layers)[:nlayers])
    if hasattr(backbone, "config"):
        backbone.config.num_hidden_layers = nlayers
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


def make_qwen3_base_from_config(
    nlayers: int = DEFAULT_QWEN3_LAYERS,
    *,
    model_path: str | Path = MODEL_DIR,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Construct the retained Qwen backbone from config without base weights.

    The caller is expected to load the backbone weights from a CAPRICORN state
    dictionary afterward. This supports inference from a self-contained
    CAPRICORN checkpoint while keeping the Qwen base weights out of the
    default download path.
    """
    if not isinstance(nlayers, int) or isinstance(nlayers, bool) or nlayers < 1:
        raise ValueError("nlayers must be a positive integer")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if nlayers > config.num_hidden_layers:
        raise ValueError(
            f"nlayers={nlayers} exceeds the configured {config.num_hidden_layers} layers"
        )
    config.num_hidden_layers = nlayers
    backbone = AutoModel.from_config(config)
    backbone = backbone.to(dtype=dtype)
    if device is not None:
        backbone = backbone.to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


class _QuestionOutput(nn.Module):
    """Combine one scalar tensor per schema question."""

    def __init__(self, question_count: int) -> None:
        super().__init__()
        self.question_count = question_count

    def _stack(self, outputs: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        if isinstance(outputs, Tensor):
            if outputs.ndim < 1 or outputs.shape[-1] != self.question_count:
                raise ValueError(
                    f"Expected final dimension {self.question_count}, got {outputs.shape}"
                )
            return outputs
        if len(outputs) != self.question_count:
            raise ValueError(
                f"Expected {self.question_count} question outputs, got {len(outputs)}"
            )
        scalars = [
            output.squeeze(-1)
            if output.ndim > 1 and output.shape[-1] == 1
            else output
            for output in outputs
        ]
        return torch.stack(scalars, dim=-1)


class _ChoiceOutput(_QuestionOutput):
    def forward(self, outputs: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        """Return a simplex probability vector across the questions."""
        return torch.softmax(self._stack(outputs), dim=-1)


class _BernoulliOutput(_QuestionOutput):
    def forward(self, outputs: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        """Return independent Bernoulli probabilities for the questions."""
        return torch.sigmoid(self._stack(outputs))


def make_output_layer(schema: OutputSchema) -> nn.Module:
    """Construct the output combiner described by ``schema``."""
    if schema.type == "choice":
        return _ChoiceOutput(len(schema.questions))
    if schema.type == "bernoulli":
        return _BernoulliOutput(len(schema.questions))
    raise ValueError(f"Unsupported output schema type: {schema.type!r}")
