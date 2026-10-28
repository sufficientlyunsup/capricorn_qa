"""Display examples from the common MNLI, ANLI, and SNLI training schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from capricorn.train import BlendedDataset


TRAIN_FILES = (
    Path("data/mnli/mnli/train-00000-of-00001.parquet"),
    Path("data/anli/plain_text/train_r1-00000-of-00001.parquet"),
    Path("data/anli/plain_text/train_r2-00000-of-00001.parquet"),
    Path("data/anli/plain_text/train_r3-00000-of-00001.parquet"),
    Path("data/snli/plain_text/train-00000-of-00001.parquet"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate the common MNLI, ANLI, and SNLI training data "
            "and display examples."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of combined examples to display (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle the combined dataset (default: 42).",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")

    dataset = BlendedDataset(
        TRAIN_FILES,
        head=args.limit,
        shuffle_seed=args.seed,
    )

    for index in range(min(args.limit, len(dataset))):
        example = dataset[index]
        print(f"Example {index + 1}")
        print(f"Premise:    {example['premise']}")
        print(f"Hypothesis: {example['hypothesis']}")
        print(f"Label:      {example['label']}")
        print(f"Source:     {example['source']}")
        print()


if __name__ == "__main__":
    main()
