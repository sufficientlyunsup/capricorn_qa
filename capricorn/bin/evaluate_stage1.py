"""Evaluate a fitted CAPRICORN checkpoint on a selected benchmark family."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from capricorn.bin.common import MODEL_DIR
from capricorn.lib.model import CapricornFactory
from capricorn.train import (
    BlendedDataset,
    HellaSwagDataset,
    HELLASWAG_SCHEMA,
    RaceChoiceDataset,
    NLI_SCHEMA,
    RACE_SCHEMA,
    make_collate_entailment,
    make_collate_hellaswag,
    make_collate_race,
)

logger = logging.getLogger(__name__)

EVALUATION_FILES = {
    "entailment": {
        "MNLI matched": Path("data/mnli/mnli/validation_matched-00000-of-00001.parquet"),
        "MNLI mismatched": Path("data/mnli/mnli/validation_mismatched-00000-of-00001.parquet"),
        "ANLI R1": Path("data/anli/plain_text/test_r1-00000-of-00001.parquet"),
        "ANLI R2": Path("data/anli/plain_text/test_r2-00000-of-00001.parquet"),
        "ANLI R3": Path("data/anli/plain_text/test_r3-00000-of-00001.parquet"),
        "SNLI": Path("data/snli/plain_text/test-00000-of-00001.parquet"),
    },
    "multiple_choice": {
        "RACE statement subset": Path("data/race/all/test-00000-of-00001.parquet"),
    },
    "hellaswag": {
        "HellaSwag validation": Path("data/hellaswag/data/validation-00000-of-00001.parquet"),
    },
}

COLLATE_FUNCTIONS = {
    "entailment": make_collate_entailment,
    "multiple_choice": make_collate_race,
    "hellaswag": make_collate_hellaswag,
}

EVALUATION_SCHEMAS = {
    "entailment": NLI_SCHEMA,
    "multiple_choice": RACE_SCHEMA,
    "hellaswag": HELLASWAG_SCHEMA,
}


def latest_checkpoint(final_path: Path) -> Path:
    """Use the final checkpoint, otherwise the highest numbered epoch file."""
    if final_path.is_file():
        return final_path
    candidates = sorted(
        path for path in final_path.parent.glob(f"{final_path.stem}-epoch-*{final_path.suffix}")
        if not path.name.endswith(".resume.pt")
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found near {final_path}")
    return candidates[-1]


def evaluate_split(
    model,
    dataset,
    tokenizer,
    batch_size: int,
    max_context_tokens: int,
    collate_factory=make_collate_entailment,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_factory(tokenizer, max_context_tokens),
    )
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, response in loader:
            logits = model.logits_model(inputs)
            targets = response.argmax(dim=-1).to(device=logits.device)
            loss = loss_fn(logits.float(), targets)
            total_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.shape[0]
    return correct / total, total_loss / total, total


def load_evaluation_dataset(evaluation_type: str, path: Path, head: int | None):
    """Load one evaluation file using the schema-specific dataset shape."""
    if evaluation_type == "entailment":
        return BlendedDataset((path,), head=head, shuffle_seed=0)
    if evaluation_type == "multiple_choice":
        return RaceChoiceDataset(path, head=head)
    if evaluation_type == "hellaswag":
        return HellaSwagDataset(path, head=head)
    raise ValueError(f"Unknown evaluation type: {evaluation_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/capricorn-nli-dynamic.pt"))
    parser.add_argument(
        "--evaluation-type",
        choices=tuple(EVALUATION_FILES),
        default="entailment",
        help="Evaluation dataset and collator family to use.",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nlayers", type=int, default=20)
    parser.add_argument("--sa-layers", type=int, default=2, help="Added self-attention blocks used by the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head", type=int, default=None, help="Limit rows per split for a smoke test.")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.nlayers < 1:
        parser.error("nlayers must be positive")
    if args.sa_layers < 1:
        parser.error("sa-layers must be positive")
    if args.batch_size < 1 or (args.head is not None and args.head < 1):
        parser.error("batch size and head must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    checkpoint = latest_checkpoint(args.checkpoint)
    logger.info("loading checkpoint %s", checkpoint)
    evaluation_type = args.evaluation_type
    factory = CapricornFactory(
        EVALUATION_SCHEMAS[evaluation_type],
        nlayers=args.nlayers,
        n_attention_layers=args.sa_layers,
        model_path=args.model_path,
        device=args.device,
    )
    model = factory.build_from_saved_file(checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, path in EVALUATION_FILES[evaluation_type].items():
        dataset = load_evaluation_dataset(evaluation_type, path, args.head)
        accuracy, loss, total = evaluate_split(
            model,
            dataset,
            tokenizer,
            args.batch_size,
            args.max_context_tokens,
            COLLATE_FUNCTIONS[evaluation_type],
        )
        random_guess_pct = 100 / len(EVALUATION_SCHEMAS[evaluation_type].questions)
        print(
            f"{name}: accuracy={accuracy:.4f} random_guess={random_guess_pct:.2f}% "
            f"loss={loss:.4f} examples={total}"
        )


if __name__ == "__main__":
    main()
