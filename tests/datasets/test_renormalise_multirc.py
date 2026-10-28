from capricorn.train import renom_multirc
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_multirc_remapping():
    print_remapped(
        parquet_rows("data/multirc/multirc/train-00000-of-00001.parquet"),
        renom_multirc,
    )
