from capricorn.train import renom_arc
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_arc_remapping():
    print_remapped(
        parquet_rows("data/arc/ARC-Challenge/train-00000-of-00001.parquet"),
        renom_arc,
    )
