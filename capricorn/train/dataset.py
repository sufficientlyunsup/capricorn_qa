"""Composable datasets used by CAPRICORN training."""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


PathInput = str | PathLike[str]
COMMON_COLUMNS = ("premise", "hypothesis", "label")
logger = logging.getLogger(__name__)


def _read_parquet_dataset(path: PathInput):
    """Read a local Parquet file without creating a Hub dataset cache lock."""
    from datasets import Dataset
    import pyarrow.parquet as parquet

    return Dataset.from_list(parquet.read_table(str(path)).to_pylist())

ENTAILMENT_QUESTION_TEMPLATES = (
    (
        'Does the passage entail this statement: "{hypothesis}"?',
        'Does this claim follow from the text: "{hypothesis}"?',
        'Does the passage establish that the following is true: "{hypothesis}"?',
        'Based on the passage, must this statement be true: "{hypothesis}"?',
        'Is the following conclusion justified by the text: "{hypothesis}"?',
        'Does the text provide sufficient evidence to conclude: "{hypothesis}"?',
        'Can this statement be inferred from the passage: "{hypothesis}"?',
        'Taking the passage as true, does it imply: "{hypothesis}"?',
        'Is this claim necessarily supported by the information given: "{hypothesis}"?',
        'Does the information in the passage confirm this statement: "{hypothesis}"?',
    ),
    (
        'Is this statement neither supported nor contradicted by the passage: "{hypothesis}"?',
        'Does the text leave the truth of this claim unresolved: "{hypothesis}"?',
        'Is there insufficient information in the passage to decide whether this is true or false: "{hypothesis}"?',
        'Could this statement be either true or false without conflicting with the passage: "{hypothesis}"?',
        'Does the passage establish neither this claim nor its opposite: "{hypothesis}"?',
        'Is the truth of this statement undetermined by the text: "{hypothesis}"?',
        'Does deciding whether this claim is true require information beyond the passage: "{hypothesis}"?',
        'Is the passage inconclusive about this statement: "{hypothesis}"?',
        'Does the information given neither confirm nor rule out this claim: "{hypothesis}"?',
        'Based only on the text, is it impossible to determine whether this statement holds: "{hypothesis}"?',
    ),
    (
        'Does the passage contradict this statement: "{hypothesis}"?',
        'Is this claim incompatible with the text: "{hypothesis}"?',
        'Does the passage establish that the following is false: "{hypothesis}"?',
        'Based on the passage, must this statement be false: "{hypothesis}"?',
        'Does this claim conflict with the information given: "{hypothesis}"?',
        'Does the text provide sufficient evidence to reject this statement: "{hypothesis}"?',
        'Can this claim be ruled out using the passage: "{hypothesis}"?',
        'Taking the passage as true, is this statement necessarily false: "{hypothesis}"?',
        'Does the passage imply the negation of this claim: "{hypothesis}"?',
        'Does the information in the text refute this statement: "{hypothesis}"?',
    ),
)


def _tokenize_sequences(tokenizer, sequences, max_context_tokens: int):
    """Log full context/question lengths before applying the token limit."""
    untruncated = tokenizer(sequences, padding=False, truncation=False)
    for index, token_ids in enumerate(untruncated["input_ids"]):
        length = len(token_ids)
        if length > max_context_tokens:
            logger.warning(
                "Truncating context/question sequence: batch_sequence=%d tokens=%d limit=%d",
                index + 1, length, max_context_tokens,
            )
    return tokenizer(
        sequences, padding=True, truncation=True, max_length=max_context_tokens,
        return_tensors="pt",
    )


def _reshape_candidates(encoded, batch_size: int, candidate_count: int):
    """Reshape flattened tokenizer output to ``[batch, candidates, tokens]``."""
    return {
        name: value.reshape(batch_size, candidate_count, -1)
        if isinstance(value, torch.Tensor) and value.ndim == 2 else value
        for name, value in encoded.items()
    }


def make_collate_entailment(tokenizer, max_context_tokens: int, *, template_seed: int | None = None):
    """Create a collator for concatenated context/question NLI sequences."""
    rng = random.Random(template_seed) if template_seed is not None else None

    def collate(examples):
        response = torch.tensor(
            [[int(example["label"] == index) for index in range(3)] for example in examples],
            dtype=torch.float32,
        )
        sequences = []
        for example in examples:
            questions = [
                (rng.choice(templates) if rng is not None else templates[0]).format(
                    hypothesis=example["hypothesis"]
                )
                for templates in ENTAILMENT_QUESTION_TEMPLATES
            ]
            sequences.extend(
                f"<context>{example['premise']}</context><question>{question}</question>"
                for question in questions
            )
        encoded = _tokenize_sequences(tokenizer, sequences, max_context_tokens)
        return _reshape_candidates(encoded, len(examples), 3), response

    collate.rng = rng
    return collate


HELLASWAG_QUESTION_TEMPLATE = 'Does the passage naturally continue with: "{hypothesis}"?'
RACE_QUESTION_TEMPLATE = (
    'Does the passage entail that the answer to "{question}" is "{option}"?'
)


def make_collate_race(tokenizer, max_context_tokens: int):
    """Create a collator for concatenated RACE context/question sequences."""

    def answer_index(answer):
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"RACE answer must be A, B, C, or D: {answer!r}")
            return ord(answer) - ord("A")
        index = int(answer)
        if index not in range(4):
            raise ValueError(f"RACE answer index must be in [0, 3]: {index}")
        return index

    def collate(examples):
        sequences = []
        for example in examples:
            sequences.extend(
                f"<context>{example['article']}</context>"
                f"<question>{RACE_QUESTION_TEMPLATE.format(question=example['question'], option=example['options'][option_index])}</question>"
                for option_index in range(4)
            )
        encoded = _tokenize_sequences(tokenizer, sequences, max_context_tokens)
        responses = torch.zeros((len(examples), 4), dtype=torch.float32)
        for row, example in enumerate(examples):
            responses[row, answer_index(example["answer"])] = 1.0
        return _reshape_candidates(encoded, len(examples), 4), responses

    return collate


def make_collate_hellaswag(tokenizer, max_context_tokens: int):
    """Create a collator for concatenated HellaSwag context/question sequences."""

    def collate(examples):
        sequences = []
        for example in examples:
            sequences.extend(
                f"<context>{example['ctx']}</context>"
                f"<question>{HELLASWAG_QUESTION_TEMPLATE.format(hypothesis=example['endings'][option_index])}</question>"
                for option_index in range(4)
            )
        encoded = _tokenize_sequences(tokenizer, sequences, max_context_tokens)
        responses = torch.zeros((len(examples), 4), dtype=torch.float32)
        for row, example in enumerate(examples):
            index = int(example["label"])
            if index not in range(4):
                raise ValueError(f"HellaSwag label must be in [0, 3]: {index}")
            responses[row, index] = 1.0
        return _reshape_candidates(encoded, len(examples), 4), responses

    return collate


class RaceChoiceDataset(TorchDataset):
    """RACE rows asking for a true, inferred, or supported statement.

    Selection uses question wording only, never answers or model predictions.
    This is a heuristic subset: options may still contain pronouns or depend
    on a speaker named in the question. ``head`` limits rows after filtering.
    Original columns and answer labels are preserved.
    """

    def __init__(self, path: PathInput, *, head: int | None = None) -> None:
        if head is not None and head < 0:
            raise ValueError("head must be non-negative or None")
        self.dataset = _read_parquet_dataset(path).filter(
            self.matches_question, input_columns=["question"]
        )
        if head is not None:
            self.dataset = self.dataset.select(range(min(head, len(self.dataset))))

    @staticmethod
    def matches_question(question: str) -> bool:
        """Select positive statement questions and generic inference stems."""
        question = re.sub(r"\s+", " ", question).strip()
        # Also catch malformed uppercase negatives such as isNOT and NOTbe.
        if "NOT" in question:
            return False
        question = question.lower()
        if re.search(r"\b(not|except|false|incorrect|untrue|cannot)\b|can.t", question):
            return False
        if re.search(r"\b(true|correct)\b", question) and re.search(
            r"\b(which|following|statement|fact)\b", question
        ):
            return True
        generic_blank = bool(re.search(r"\bthat\s*_+\s*[. ?]*$", question))
        if re.search(r"\b(infer|inferred|conclude|concluded)\b", question):
            if generic_blank or re.match(r"(what|which).*\b(infer|inferred|conclude|concluded)\b", question):
                return True
        if generic_blank and re.search(r"\b(learn|learned|learnt|know|known)\b", question):
            return True
        if re.match(r"what (?:can|do) (?:we|you) (?:learn|know)\b", question):
            return True
        if re.match(r"what can be (?:learned|learnt|known)\b", question):
            return True
        return generic_blank and bool(
            re.search(r"\b(imply|implies|implied|suggest|suggests)\b", question)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class HellaSwagDataset(TorchDataset):
    """Load HellaSwag rows with four candidate sentence endings."""

    def __init__(self, path: PathInput, *, head: int | None = None) -> None:
        if head is not None and head < 0:
            raise ValueError("head must be non-negative or None")
        self.dataset = _read_parquet_dataset(path)
        if head is not None:
            self.dataset = self.dataset.select(range(min(head, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class BlendedDataset(TorchDataset):
    """Load and concatenate compatible Parquet datasets.

    Parameters
    ----------
    paths:
        Parquet files containing the constituent datasets. They are loaded in
        the supplied order.
    head:
        Optional number of rows to retain from the start of *each* source
        file. This is useful for making a small, balanced development blend.
    The shared NLI ``label`` column is retained as an integer. Invalid labels
    such as SNLI's ``-1`` no-consensus examples are dropped. Response tensors
    are a batching concern and are constructed by the training collator.
    shuffle_seed:
        Optional seed for shuffling the concatenated dataset. If omitted, the
        source datasets remain in their supplied order.

    The resulting Hugging Face ``Dataset`` is available as ``.dataset``. The
    wrapper implements the PyTorch dataset interface, so it can be passed
    directly to ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        paths: Iterable[PathInput],
        *,
        head: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("At least one dataset path is required")
        if head is not None and head < 0:
            raise ValueError("head must be non-negative or None")

        self.head = head
        self.shuffle_seed = shuffle_seed

        self.dataset = self._load()

    def _load(self) -> Any:
        try:
            from datasets import Dataset, concatenate_datasets
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "BlendedDataset requires the Hugging Face datasets package; "
                "install it with `uv add datasets`."
            ) from exc

        source_datasets = []

        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file does not exist: {path}")

            dataset = Dataset.from_parquet(str(path))

            missing = set(COMMON_COLUMNS) - set(dataset.column_names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"{path} is missing required common columns: {missing_names}"
                )
            dataset = dataset.select_columns(COMMON_COLUMNS)

            if self.head is not None:
                dataset = dataset.select(range(min(self.head, len(dataset))))

            if "label" not in dataset.column_names:
                raise ValueError(f"{path} is missing label column")
            valid_labels = {0, 1, 2}
            dataset = dataset.filter(
                lambda example: example["label"] in valid_labels
            )
            dataset = dataset.add_column(
                "source",
                [self._source_name(path)] * len(dataset),
            )
            source_datasets.append(dataset)

        dataset = concatenate_datasets(source_datasets)
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        return dataset

    @staticmethod
    def _source_name(path: Path) -> str:
        """Derive the NLI source tag from a downloaded file path."""
        path_text = str(path).lower()
        if "mnli" in path_text:
            return "MNLI"
        if "anli" in path_text:
            for round_name in ("r1", "r2", "r3"):
                if round_name in path.stem.lower():
                    return f"ANLI-{round_name}"
            return "ANLI"
        if "snli" in path_text:
            return "SNLI"
        return path.parent.name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice | str) -> Any:
        return self.dataset[index]

    def __iter__(self):
        return iter(self.dataset)


def make_collate_qa(tokenizer, max_context_tokens, *, template_seed=None):
    def collate(rows):
        sequences = [
            f"<context>{row.context}</context><question>{row.question}</question>"
            for row in rows
        ]
        targets = torch.tensor([[float(row.label)] for row in rows])
        return _tokenize_sequences(tokenizer, sequences, max_context_tokens), targets

    # Same checkpoint RNG interface as the stage 1 collator; prompts are fixed.
    collate.rng = random.Random(template_seed)
    return collate
