from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    score: float
    section: str = ""


@dataclass
class TextChunk:
    """A piece of the source document, before it is embedded or scored."""

    text: str
    section: str = ""
