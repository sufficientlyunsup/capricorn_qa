from pathlib import Path
from pprint import pprint
from typing import Any

import pytest


def parquet_rows(path: str, count: int = 10) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        pytest.skip(f"local training file is not available: {path}")
    import pyarrow.parquet as parquet

    return parquet.read_table(path).slice(0, count).to_pylist()


def json_rows(path: str, count: int = 3) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        pytest.skip(f"local training file is not available: {path}")
    import json

    with path.open(encoding="utf-8") as stream:
        return json.load(stream)[:count]


def print_remapped(rows, remapper) -> None:
    for index, row in enumerate(rows):
        print(f"\nsource row {index}")
        pprint(row)
        print("Bernoulli trials")
        pprint(remapper(row))

