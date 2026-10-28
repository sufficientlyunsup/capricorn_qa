from capricorn.lib.model import FusionLayer, OutputSchema
from capricorn.lib.model.construct import make_output_layer
import torch


def test_fusion_layer_produces_one_logit() -> None:
    layer = FusionLayer(
        hidden_size=8,
        num_heads=2,
        n_attention_layers=2,
    )
    output = layer(torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool))
    assert output.shape == (3, 1)


def test_choice_schema() -> None:
    schema = OutputSchema.model_validate_json(
        '{"name": "nli_relation", "type": "choice", "questions": ["entailment", "neutral", "contradiction"]}'
    )
    assert schema.name == "nli_relation"
    assert schema.type == "choice"
    assert len(schema.questions) == 3


def test_bernoulli_schema() -> None:
    schema = OutputSchema.model_validate_json(
        '{"name": "truth", "type": "bernoulli", "questions": ["The statement is true."]}'
    )
    assert schema.name == "truth"
    assert schema.type == "bernoulli"
    assert schema.questions == ("The statement is true.",)


def test_choice_output_normalizes_question_logits() -> None:
    schema = OutputSchema.model_validate_json(
        '{"name": "nli", "type": "choice", "questions": ["a", "b"]}'
    )
    output = make_output_layer(schema)([torch.tensor([1.0]), torch.tensor([2.0])])
    assert output.shape == (1, 2)
    assert output.shape == (1, 2)
    assert torch.allclose(output.sum(dim=-1), torch.ones(1))


def test_bernoulli_output_concatenates_question_scalars() -> None:
    schema = OutputSchema.model_validate_json(
        '{"name": "truth", "type": "bernoulli", "questions": ["a", "b"]}'
    )
    output = make_output_layer(schema)([torch.tensor([0.2]), torch.tensor([0.8])])
    assert torch.allclose(output, torch.sigmoid(torch.tensor([[0.2, 0.8]])))
