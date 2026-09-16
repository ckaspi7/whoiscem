from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    score: float
    section: str = ""
