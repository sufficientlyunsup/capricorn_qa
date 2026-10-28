import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from capricorn.train import RaceChoiceDataset


@pytest.mark.parametrize("question", [
    "Which statement is TRUE?",
    "What can we infer from the passage?",
    "It can be concluded from the text that _ .",
    "What do we know about John?",
    "We can learn from the passage that _.",
    "The passage implies that _.",
])
def test_accepts_statement_questions(question):
    assert RaceChoiceDataset.matches_question(question)


@pytest.mark.parametrize("question", [
    "Which of the following is NOT true?",
    "Which of the following isNOT True?",
    "Which facts might NOTbe true?",
    "What cannot be inferred?",
    "All statements are true except _.",
    "What is the best title?",
    "Why did John leave?",
    "We can infer that John was _.",
])
def test_excludes_negatives_and_unrelated_questions(question):
    assert not RaceChoiceDataset.matches_question(question)


def test_filters_before_head_and_preserves_rows(tmp_path):
    rows = [
        {"article": "text", "question": question,
         "options": ["a", "b", "c", "d"], "answer": "B"}
        for question in ["What is the title?", "Which statement is true?",
                         "What can we infer from the passage?"]
    ]
    path = tmp_path / "race.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = RaceChoiceDataset(path, head=1)
    assert len(dataset) == 1
    assert dataset[0] == rows[1]
