from unittest.mock import Mock

import torch

from capricorn.train import make_collate_race


def test_race_collate_encodes_four_question_options():
    tokenizer = Mock(return_value={"input_ids": torch.ones(8, 4, dtype=torch.long)})
    examples = [
        {
            "article": "A passage.",
            "question": "What follows?",
            "options": ["the first answer", "the second answer", "third", "fourth"],
            "answer": "B",
        },
        {
            "article": "Another passage.",
            "question": "Which is correct?",
            "options": ["one", "two", "three", "four"],
            "answer": 3,
        },
    ]

    inputs, response = make_collate_race(tokenizer, 512)(examples)

    sequences = tokenizer.call_args.args[0]
    assert len(sequences) == 8
    assert sequences[0].startswith("<context>A passage.</context><question>")
    assert 'Does the passage entail that the answer to "What follows?" is "the first answer"?' in sequences[0]
    assert 'Does the passage entail that the answer to "Which is correct?" is "four"?' in sequences[7]
    assert inputs["input_ids"].shape == (2, 4, 4)
    torch.testing.assert_close(
        response,
        torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]]),
    )
