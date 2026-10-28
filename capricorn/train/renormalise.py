"""Normalize heterogeneous QA rows into independent Bernoulli trials."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class BernoulliTrial:
    """One context/question pair and its independent Boolean target."""

    context: str
    question: str
    label: bool


Remapper = Callable[[Mapping[str, Any]], list[BernoulliTrial]]
PathInput = str | PathLike[str]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _candidate_question(question: Any, candidate: Any) -> str:
    return f'Is the answer to "{_text(question)}" "{_text(candidate)}"?'


def _choice_trials(
    context: Any,
    question: Any,
    candidates: Iterable[Any],
    answer_index: int,
) -> list[BernoulliTrial]:
    candidates = list(candidates)
    if answer_index < 0 or answer_index >= len(candidates):
        raise ValueError(
            f"answer index {answer_index} is outside candidate range [0, {len(candidates)})"
        )
    return [
        BernoulliTrial(
            context=_text(context),
            question=_candidate_question(question, candidate),
            label=index == answer_index,
        )
        for index, candidate in enumerate(candidates)
    ]


def _choice_mapping(example: Mapping[str, Any]) -> tuple[list[Any], int]:
    choices = example["choices"]
    if not isinstance(choices, Mapping):
        raise TypeError("choices must be a mapping containing text and label")
    texts = list(choices["text"])
    labels = [str(label) for label in choices["label"]]
    if len(labels) != len(texts) or len(set(labels)) != len(labels):
        raise ValueError("choice labels must be unique and match the number of texts")
    return texts, labels.index(str(example["answerKey"]))


def renom_boolq(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Map one BoolQ row to one passage-grounded Boolean question."""
    return [
        BernoulliTrial(
            context=_text(example["passage"]),
            question=_text(example["question"]),
            label=bool(example["answer"]),
        )
    ]


def renom_multirc(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Map one MultiRC question/answer pair to one independent trial."""
    return [
        BernoulliTrial(
            context=_text(example["paragraph"]),
            question=_candidate_question(example["question"], example["answer"]),
            label=bool(int(example["label"])),
        )
    ]


def renom_commonsenseqa(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Expand CommonsenseQA's five choices into Boolean trials."""
    choices, answer_index = _choice_mapping(example)
    return _choice_trials(
        context="",
        question=example["question"],
        candidates=choices,
        answer_index=answer_index,
    )


def renom_arc(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Expand an ARC-Easy or ARC-Challenge row into candidate trials."""
    choices, answer_index = _choice_mapping(example)
    return _choice_trials(
        context="",
        question=example["question"],
        candidates=choices,
        answer_index=answer_index,
    )


def _renom_nli(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    from capricorn.train.dataset import ENTAILMENT_QUESTION_TEMPLATES

    label = int(example["label"])
    if label == -1:
        return []  # No gold annotation, notably in SNLI.
    if label not in (0, 1, 2):
        raise ValueError(f"unexpected NLI label: {label}")
    return [
        BernoulliTrial(
            context=_text(example["premise"]),
            question=templates[0].format(hypothesis=_text(example["hypothesis"])),
            label=label == index,
        )
        for index, templates in enumerate(ENTAILMENT_QUESTION_TEMPLATES)
    ]


def renom_mnli(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Expand MNLI into entailment, neutral, and contradiction trials."""
    return _renom_nli(example)


def renom_anli(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Expand an ANLI row from any round into three Boolean trials."""
    return _renom_nli(example)


def renom_snli(example: Mapping[str, Any]) -> list[BernoulliTrial]:
    """Expand labelled SNLI rows into three trials; omit unlabelled rows."""
    return _renom_nli(example)


class QARemappingDataset(TorchDataset):
    """Load native QA data and expose its Bernoulli trials to PyTorch.

    ``max_keep_rows`` limits native source rows before remapping. Expansion
    happens before batching, so a DataLoader batch size refers to normalized
    trials rather than native source rows.
    """

    def __init__(
        self,
        source: PathInput,
        remapper: str,
        max_keep_rows: int | None = None,
    ):
        if max_keep_rows is not None and max_keep_rows < 0:
            raise ValueError("max_keep_rows must be non-negative or None")
        try:
            remap = REMAPPERS[remapper]
        except KeyError as error:
            available = ", ".join(sorted(REMAPPERS))
            raise ValueError(
                f"unknown remapper {remapper!r}; expected one of: {available}"
            ) from error

        examples = self._load_examples(Path(source))
        if max_keep_rows is not None:
            examples = examples[:max_keep_rows]
        self._trials = tuple(
            trial
            for example in examples
            for trial in remap(example)
        )

    @staticmethod
    def _load_examples(source: Path) -> list[Mapping[str, Any]]:
        if not source.is_file():
            raise FileNotFoundError(f"QA source file does not exist: {source}")
        if source.suffix == ".parquet":
            import pyarrow.parquet as parquet

            return parquet.read_table(source).to_pylist()
        if source.suffix == ".json":
            with source.open(encoding="utf-8") as stream:
                examples = json.load(stream)
            if not isinstance(examples, list):
                raise TypeError(f"expected a JSON list of examples in {source}")
            return examples
        raise ValueError(f"unsupported QA source format: {source.suffix}")

    def __len__(self) -> int:
        return len(self._trials)

    def __getitem__(self, index: int) -> BernoulliTrial:
        return self._trials[index]


REMAPPERS: dict[str, Remapper] = {
    "mnli": renom_mnli,
    "anli": renom_anli,
    "snli": renom_snli,
    "boolq": renom_boolq,
    "multirc": renom_multirc,
    "commonsenseqa": renom_commonsenseqa,
    "arc": renom_arc,
}


QA_DIRECTORIES = {
    "boolq": "boolq/data",
    "multirc": "multirc/multirc",
    "commonsenseqa": "commonsense_qa/data",
    "arc": "arc/ARC-Challenge",
}


def load_splits(data_dir, split, max_keep_rows=None):
    datasets = {
        name: QARemappingDataset(
            Path(data_dir) / directory / f"{split}-00000-of-00001.parquet",
            remapper=name, max_keep_rows=max_keep_rows,
        )
        for name, directory in QA_DIRECTORIES.items()
    }
    nli_files = {
        "mnli": (f"mnli/mnli/{'train' if split == 'train' else 'validation_matched'}-00000-of-00001.parquet",),
        "snli": (f"snli/plain_text/{split}-00000-of-00001.parquet",),
        "anli": tuple(f"anli/plain_text/{'train' if split == 'train' else 'dev'}_r{r}-00000-of-00001.parquet" for r in (1, 2, 3)),
    }
    if split not in ("train", "validation"):
        raise ValueError("Expected train or validation split")
    for name, paths in nli_files.items():
        # Share the source-row budget across ANLI rounds, rather than tripling it.
        caps = [None] * len(paths) if max_keep_rows is None else [
            max_keep_rows // len(paths) + (i < max_keep_rows % len(paths))
            for i in range(len(paths))
        ]
        datasets[name] = ConcatDataset([
            QARemappingDataset(Path(data_dir) / path, name, cap)
            for path, cap in zip(paths, caps)
        ])
    return datasets


def blended_split(data_dir, split, max_keep_rows=None):
    return ConcatDataset(list(load_splits(data_dir, split, max_keep_rows).values()))
