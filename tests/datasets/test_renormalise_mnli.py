from capricorn.train import renom_mnli
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_mnli_remapping():
    rows = parquet_rows("data/mnli/mnli/train-00000-of-00001.parquet", count=3)
    print_remapped(rows, renom_mnli)
    for row in rows:
        trials = renom_mnli(row)
        if row["label"] == -1:
            assert trials == []
        else:
            assert len(trials) == 3
            assert [trial.label for trial in trials] == [row["label"] == i for i in range(3)]
            assert all(trial.context == row["premise"] for trial in trials)


def test_mnli_all_labels():
    for label in (0, 1, 2):
        trials = renom_mnli({"premise": "A passage", "hypothesis": "A claim", "label": label})
        assert [t.label for t in trials] == [label == i for i in range(3)]
        assert all("A claim" in t.question for t in trials)
    assert renom_mnli({"label": -1}) == []
