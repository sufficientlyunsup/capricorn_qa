import argparse
import random
from types import SimpleNamespace

import torch
from torch import nn

from capricorn.bin import train_stage2 as training
from capricorn.bin.evaluate_stage1 import latest_checkpoint


def test_epoch_resume_matches_uninterrupted_training(monkeypatch, tmp_path):
    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_layer = nn.Linear(2, 1)

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
            y = (torch.tensor(rows) % 2).float().unsqueeze(-1)
            return {"features": x + noise}, y

        collate.rng = rng
        return collate

    monkeypatch.setattr(training, "blended_split", lambda *a, **kw: list(range(7)))
    monkeypatch.setattr(training, "make_collate_qa", collator)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(training, "CapricornFactory",
                        lambda *a, **kw: SimpleNamespace(build_trainable_capricorn=Model))

    def args(directory, epochs, resume=None):
        return argparse.Namespace(
            resume=resume, seed=42, max_keep_rows=3, data_dir=tmp_path, validation_head=None,
            checkpoint=initial,
            model_path="unused", batch_size=3, max_context_tokens=32,
            nlayers=20, sa_layers=2, device="cpu", learning_rate=0.01, patience=10,
            min_delta=0., output=directory / "model.pt", epochs=epochs, log_every=100,
        )

    initial = tmp_path / "stage1.pt"
    torch.save(Model().state_dict(), initial)
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



def test_arc_numeric_labels_and_source_row_cap(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from capricorn.train import QARemappingDataset

    source = tmp_path / "arc.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"question": "Which?", "choices": {"label": ["2", "1"], "text": ["yes", "no"]}, "answerKey": "2"},
        {"question": "Ignored", "choices": {"label": ["A"], "text": ["x"]}, "answerKey": "missing"},
    ]), source)
    dataset = QARemappingDataset(source, "arc", max_keep_rows=1)
    assert len(dataset) == 2
    assert [row.label for row in dataset] == [True, False]


def test_binary_evaluation_metrics(monkeypatch):
    from capricorn.bin import evaluate_stage2 as evaluation
    from capricorn.train import BernoulliTrial

    monkeypatch.setattr(evaluation, "make_collate_qa", lambda *args: lambda rows: (
        {}, torch.tensor([[float(row.label)] for row in rows])))

    class Model(nn.Module):
        def logits_model(self, inputs):
            return torch.tensor([[2.], [-2.], [-2.]])

    rows = [BernoulliTrial("", "", label) for label in [True, False, True]]
    accuracy, loss, count, majority = evaluation.evaluate_split(Model(), rows, None, 3, 32)
    assert accuracy == 2 / 3
    assert count == 3
    assert majority == 2 / 3
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2., -2., -2.]), torch.tensor([1., 0., 1.])).item()
    assert abs(loss - expected) < 1e-6
