from unittest.mock import Mock

import torch

from capricorn.train import make_collate_entailment


def test_collate_concatenates_wrapped_context_and_question():
    tokenizer = Mock(return_value={"input_ids": torch.ones(6, 3, dtype=torch.long)})
    examples = [
        {"premise": "passage one", "hypothesis": "claim one", "label": 0},
        {"premise": "passage two", "hypothesis": "claim two", "label": 1},
    ]
    inputs, response = make_collate_entailment(tokenizer, 512)(examples)
    sequences = tokenizer.call_args.args[0]
    assert len(sequences) == 6
    assert sequences[0].startswith("<context>passage one</context><question>")
    assert "claim one" in sequences[0]
    assert inputs["input_ids"].shape == (2, 3, 3)
    torch.testing.assert_close(response, torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
