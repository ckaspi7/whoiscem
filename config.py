"""Runtime configuration, resolved from the environment in one place.

Every external backend is selected here rather than at the call site, so the
same code path runs against an on-disk store on a laptop, a container under
docker-compose, and a managed service in production. Call sites depend on
``Settings``; the concrete client is built by ``retrieval.backends``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Loaded here so every caller sees the same environment, whether it came in
# through streamlit, the eval harness, a script, or a test. Real environment
# variables always win over .env, which keeps CI and Streamlit Cloud authoritative.
load_dotenv()

QdrantMode = Literal["embedded", "server", "cloud"]

QDRANT_MODES: tuple[str, ...] = ("embedded", "server", "cloud")

DEFAULT_QDRANT_MODE = "embedded"
DEFAULT_QDRANT_PATH = ".qdrant"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_RESUME_PATH = os.path.join("data", "resume.md")
# Chosen by measurement, not taste: at top-3 the full hybrid pipeline recalls
# 75% of golden references and plain dense search recalls 79%; at top-5 the
# pipeline reaches 92%. See eval/results/ablation-retrieval.json.
DEFAULT_RETRIEVAL_TOP_N = 5

# Defaults chosen from eval/results/ablation-retrieval.json, where plain dense
# search at k=5 reaches 100% recall on the golden references in 2.2 KB of
# context at 215 ms, while the full hybrid pipeline reaches 91.7% in 2.4 KB at
# 940 ms. The hybrid parts are kept and configurable: BM25 and reranking earn
# their keep on larger, more lexically varied corpora, and this corpus is one
# short document.
# "auto" hands the model the whole corpus while it still fits in a prompt, and
# selects once it does not. Measured, not assumed: on this 7 KB resume every
# increase in returned context improves grounding — faithfulness on the resume
# route is 0.79 at dense top-5 and 0.87 with the whole document — because the
# model draws on more than the one snippet a question references. Selection is
# a cost paid for a corpus that does not fit, and this one does.
RETRIEVAL_STRATEGIES: tuple[str, ...] = ("auto", "dense", "sparse", "rrf", "rrf_rerank")
DEFAULT_RETRIEVAL_STRATEGY = "auto"

# "classifier" is the original design: an LLM call classifies the query into
# one of five fixed labels, then a hardcoded switch picks exactly one tool.
# "tool_calling" binds the tools to the LLM directly (bind_tools) and lets it
# choose, call zero-to-many of them, and loop back with the results before
# answering — a real agent rather than a switch statement wearing one's
# docstrings. Kept alongside each other and both measured (see
# eval/results/) rather than one replacing the other on faith: the plan this
# project follows is explicit that demonstrating the comparison is worth more
# than either choice alone.
AGENT_MODES: tuple[str, ...] = ("classifier", "tool_calling")
# Defaults to the measured baseline until the comparison run says otherwise —
# same discipline as retrieval_strategy: no default changes on faith here.
DEFAULT_AGENT_MODE = "classifier"

# Which OpenAI chat model powers routing, condensation, and generation in both
# graphs. The faithfulness judge (guardrails/faithfulness_check.py) and RAGAS's
# own judge are deliberately not parameterised by this — they stay pinned to
# gpt-4o-mini so a model swap changes exactly the thing being measured, not
# the thing measuring it.
CHAT_MODELS: tuple[str, ...] = ("gpt-4o-mini", "gpt-6-luna")
# Defaults to the incumbent until a real before/after says otherwise — same
# discipline as retrieval_strategy and agent_mode.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

# Roughly 3k tokens; comfortably inside the window and cheap enough per turn.
CORPUS_FITS_CONTEXT_CHARS = 12_000

DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006"
DEFAULT_PHOENIX_PROJECT = "whoiscem"


class ConfigError(ValueError):
    """Raised when the environment describes a backend that cannot be built."""


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process.

    Attributes:
        qdrant_mode: ``embedded`` (on-disk, no server), ``server`` (host/port),
            or ``cloud`` (URL + API key).
        qdrant_path: Storage directory used by ``embedded`` mode. One process at
            a time may hold it — the client takes an exclusive lock.
        redis_url: Empty means "no Redis server"; session memory falls back to
            an in-process store rather than turning itself off.
        resume_path: Resume indexed on first run — Markdown or PDF.
        phoenix_endpoint: Trace collector. Defaults to a local Phoenix, which
            needs no account; point it at Phoenix Cloud to export there instead.
    """

    qdrant_mode: QdrantMode = DEFAULT_QDRANT_MODE
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_host: str = DEFAULT_QDRANT_HOST
    qdrant_port: int = DEFAULT_QDRANT_PORT
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    redis_url: str = ""
    resume_path: str = DEFAULT_RESUME_PATH
    retrieval_top_n: int = DEFAULT_RETRIEVAL_TOP_N
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY
    agent_mode: str = DEFAULT_AGENT_MODE
    chat_model: str = DEFAULT_CHAT_MODEL
    phoenix_endpoint: str = DEFAULT_PHOENIX_ENDPOINT
    phoenix_api_key: str = ""
    phoenix_project: str = DEFAULT_PHOENIX_PROJECT

    def __post_init__(self) -> None:
        if self.qdrant_mode not in QDRANT_MODES:
            raise ConfigError(
                f"QDRANT_MODE must be one of {', '.join(QDRANT_MODES)} — got {self.qdrant_mode!r}"
            )
        if self.retrieval_strategy not in RETRIEVAL_STRATEGIES:
            raise ConfigError(
                f"RETRIEVAL_STRATEGY must be one of {', '.join(RETRIEVAL_STRATEGIES)} "
                f"— got {self.retrieval_strategy!r}"
            )
        if self.agent_mode not in AGENT_MODES:
            raise ConfigError(f"AGENT_MODE must be one of {', '.join(AGENT_MODES)} — got {self.agent_mode!r}")
        if self.chat_model not in CHAT_MODELS:
            raise ConfigError(f"CHAT_MODEL must be one of {', '.join(CHAT_MODELS)} — got {self.chat_model!r}")
        if self.qdrant_mode == "cloud" and not self.qdrant_url:
            raise ConfigError("QDRANT_MODE=cloud requires QDRANT_URL (and usually QDRANT_API_KEY)")


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build ``Settings`` from environment variables (``os.environ`` by default)."""
    src = os.environ if env is None else env

    def get(key: str, default: str = "") -> str:
        return str(src.get(key, default)).strip()

    def _int(raw: str, name: str) -> int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer — got {raw!r}") from exc

    port = _int(get("QDRANT_PORT", str(DEFAULT_QDRANT_PORT)), "QDRANT_PORT")

    return Settings(
        qdrant_mode=get("QDRANT_MODE", DEFAULT_QDRANT_MODE).lower(),  # type: ignore[arg-type]
        qdrant_path=get("QDRANT_PATH", DEFAULT_QDRANT_PATH),
        qdrant_host=get("QDRANT_HOST", DEFAULT_QDRANT_HOST),
        qdrant_port=port,
        qdrant_url=get("QDRANT_URL"),
        qdrant_api_key=get("QDRANT_API_KEY"),
        redis_url=get("REDIS_URL"),
        resume_path=get("RESUME_PATH", DEFAULT_RESUME_PATH),
        retrieval_top_n=_int(get("RETRIEVAL_TOP_N", str(DEFAULT_RETRIEVAL_TOP_N)), "RETRIEVAL_TOP_N"),
        retrieval_strategy=get("RETRIEVAL_STRATEGY", DEFAULT_RETRIEVAL_STRATEGY).lower(),
        agent_mode=get("AGENT_MODE", DEFAULT_AGENT_MODE).lower(),
        chat_model=get("CHAT_MODEL", DEFAULT_CHAT_MODEL).lower(),
        phoenix_endpoint=get("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_PHOENIX_ENDPOINT),
        phoenix_api_key=get("PHOENIX_API_KEY"),
        phoenix_project=get("PHOENIX_PROJECT_NAME", DEFAULT_PHOENIX_PROJECT),
    )
