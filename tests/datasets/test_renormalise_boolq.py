from capricorn.train import renom_boolq
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_boolq_remapping():
    print_remapped(
        parquet_rows("data/boolq/data/train-00000-of-00001.parquet"),
        renom_boolq,
    )
