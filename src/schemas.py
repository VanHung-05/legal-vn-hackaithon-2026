"""Shared data structures for the legal retrieval pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Article:
    """A single legal article from the corpus."""
    aid: int
    law_id: str
    text: str
    text_truncated: str = ""
    char_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Question:
    """A legal question from train/test set."""
    qid: int
    question: str
    relevant_laws: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """A single retrieval result."""
    aid: int
    score: float
    law_id: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """Model output for one question."""
    qid: int
    relevant_laws: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
