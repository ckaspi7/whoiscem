# Lazy imports — import submodules explicitly to avoid cascade loading heavy deps at test time
from retrieval.types import ScoredChunk

__all__ = ["ScoredChunk"]
