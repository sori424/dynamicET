"""Local dataset helpers for dynamicET experiments.

This package intentionally provides a small ``Dataset`` class with the subset
of the Hugging Face Dataset API used by this repository.  Keeping it local
avoids ambiguity with the external ``datasets`` package when running scripts
from the project root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable


class Dataset:
    """Minimal map-style dataset used by the experiment DataLoaders."""

    def __init__(self, data: Mapping[str, Any]):
        if not data:
            raise ValueError("Dataset requires at least one column.")
        lengths = {key: len(value) for key, value in data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset columns have mismatched lengths: {lengths}")
        self.data = dict(data)
        self.length = next(iter(lengths.values()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dataset":
        return cls(data)

    def with_format(self, *_args: Any, **_kwargs: Any) -> "Dataset":
        return self

    def select(self, indices: Iterable[int]) -> "Dataset":
        index_list = list(indices)
        return Dataset(
            {key: self._take_many(value, index_list) for key, value in self.data.items()}
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        if isinstance(index, (slice, range)) or self._is_index_sequence(index):
            return {
                key: self._take_many(value, self._normalize_indices(index))
                for key, value in self.data.items()
            }
        return {key: value[index] for key, value in self.data.items()}

    def _normalize_indices(self, index) -> list[int]:
        if isinstance(index, slice):
            return list(range(*index.indices(self.length)))
        if isinstance(index, range):
            return list(index)
        return [int(item) for item in index]

    @staticmethod
    def _is_index_sequence(index) -> bool:
        return isinstance(index, Sequence) and not isinstance(index, (str, bytes))

    @staticmethod
    def _take_many(value, indices: list[int]):
        try:
            return value[indices]
        except (TypeError, IndexError):
            return [value[index] for index in indices]


__all__ = ["Dataset"]
