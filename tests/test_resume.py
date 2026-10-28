import argparse
import random
from types import SimpleNamespace

import torch
from torch import nn

from capricorn.bin import train_stage1 as training
from capricorn.bin.evaluate_stage1 import latest_checkpoint


def test_epoch_resume_matches_uninterrupted_training(monkeypatch, tmp_path):
    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_layer = nn.Linear(2, 3)

        def forward(self, inputs):
            return self.fusion_layer(inputs["features"])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits_model = Logits()

    def collator(tokenizer, max_tokens, *, template_seed=None):
        rng = random.Random(template_seed) if template_seed is not None else None

        def collate(rows):
            x = torch.tensor([[float(row), 1.] for row in rows])
            noise = torch.tensor([[rng.random() if rng else 0., 0.] for _ in rows])
            y = torch.nn.functional.one_hot(torch.tensor(rows) % 3, 3).float()
            return {"features": x + noise}, y

        collate.rng = rng
        return collate

    monkeypatch.setattr(training, "BlendedDataset", lambda *a, **kw: list(range(7)))
    monkeypatch.setattr(training, "make_collate_entailment", collator)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(training, "CapricornFactory",
                        lambda *a, **kw: SimpleNamespace(build_trainable_capricorn=Model))

    def args(directory, epochs, resume=None):
        return argparse.Namespace(
            resume=resume, seed=42, head=None, validation_head=None,
            model_path="unused", batch_size=3, max_context_tokens=32,
            nlayers=20, sa_layers=2, device="cpu", learning_rate=0.01, patience=10,
            min_delta=0., output=directory / "model.pt", epochs=epochs, log_every=100,
        )

    training.train(args(tmp_path / "full", 2))
    training.train(args(tmp_path / "split", 1))
    restart = tmp_path / "split/model-epoch-001.resume.pt"
    training.train(args(tmp_path / "split", 2, restart))
    full = torch.load(tmp_path / "full/model-epoch-002.pt", weights_only=True)
    split = torch.load(tmp_path / "split/model-epoch-002.pt", weights_only=True)
    for key in full:
        torch.testing.assert_close(full[key], split[key], rtol=0, atol=0)
    state = torch.load(tmp_path / "split/model-epoch-002.resume.pt", weights_only=True)
    assert state["epoch"] == 2
    assert state["optimizer"]["state"]


def test_evaluator_ignores_resume_bundles(tmp_path):
    weights = tmp_path / "model-epoch-001.pt"
    weights.touch()
    (tmp_path / "model-epoch-001.resume.pt").touch()
    assert latest_checkpoint(tmp_path / "model.pt") == weights
