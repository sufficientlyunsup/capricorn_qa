from capricorn.train import renom_snli
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_snli_remapping():
    rows = parquet_rows("data/snli/plain_text/train-00000-of-00001.parquet", count=3)
    print_remapped(rows, renom_snli)
    for row in rows:
        trials = renom_snli(row)
        if row["label"] == -1:
            assert trials == []
        else:
            assert len(trials) == 3
            assert [trial.label for trial in trials] == [row["label"] == i for i in range(3)]
            assert all(trial.context == row["premise"] for trial in trials)


def test_snli_all_labels():
    for label in (0, 1, 2):
        trials = renom_snli({"premise": "A passage", "hypothesis": "A claim", "label": label})
        assert [t.label for t in trials] == [label == i for i in range(3)]
        assert all("A claim" in t.question for t in trials)
    assert renom_snli({"label": -1}) == []
