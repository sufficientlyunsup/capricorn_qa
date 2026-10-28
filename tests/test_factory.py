from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from capricorn.lib.model import CapricornFactory, OutputSchema


def test_factory_device_and_restores_concatenated_model(monkeypatch, tmp_path):
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5, 8)
            self.config = SimpleNamespace(
                hidden_size=8, num_attention_heads=2, intermediate_size=16
            )

        def forward(self, input_ids, attention_mask, return_dict):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    loader = Mock(side_effect=lambda **kwargs: Backbone().to(kwargs["device_map"]))
    monkeypatch.setattr("capricorn.lib.model.capricorn.make_qwen3_base", loader)
    factory = CapricornFactory(
        OutputSchema(name="test", type="bernoulli", questions=("question",)),
        device="cpu", dtype=torch.float32,
    )
    model = factory.build_trainable_capricorn()
    assert loader.call_args.kwargs["device_map"] == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = factory.build_from_saved_file(path)
    inputs = {
        "input_ids": torch.tensor([[[1, 2]], [[2, 3]]]),
        "attention_mask": torch.ones(2, 1, 2),
    }
    probabilities = restored(inputs)
    assert probabilities.shape == (2, 1)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    restored.train()
    assert not restored.logits_model.backbone.training
    assert all(not p.requires_grad for p in restored.logits_model.backbone.parameters())
    restored.logits_model(inputs).sum().backward()
    assert all(p.grad is None for p in restored.logits_model.backbone.parameters())
    assert restored.logits_model.fusion_layer.output.weight.grad is not None
