from __future__ import annotations

from itertools import islice
from typing import Iterable

from datasets import load_dataset


def iter_texts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_examples: int | None = None,
) -> Iterable[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    stream = (row[text_column] for row in dataset if row.get(text_column))
    if max_examples is not None:
        stream = islice(stream, max_examples)
    for text in stream:
        if isinstance(text, str) and text.strip():
            yield text


def batch_texts(texts: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
