from capricorn.train import renom_commonsenseqa
from tests.datasets.renormalise_helpers import parquet_rows, print_remapped


def test_print_commonsenseqa_remapping():
    print_remapped(
        parquet_rows("data/commonsense_qa/data/train-00000-of-00001.parquet"),
        renom_commonsenseqa,
    )
