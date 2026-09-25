"""Observability: OpenTelemetry tracing to Arize Phoenix, and structured
logging with a session id correlation field (Phase 4.4).

Tracing is vendor-neutral: OpenInference semantic conventions over
OpenTelemetry. Phoenix is the collector, chosen by configuration the same way
the vector store and session memory are, and it runs locally with no account:

    pip install arize-phoenix && phoenix serve     # http://localhost:6006

Set PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY to export to Phoenix Cloud
instead. Tracing never blocks the app — if the collector is unreachable or the
packages are missing, this degrades to a no-op and says so once.

Logging: every module already does ``logger = logging.getLogger(__name__)``,
but nothing ever called ``logging.basicConfig`` — Python's logging module
silently falls back to WARNING-level, unstructured, uncorrelated output on
stderr, so a deployed instance's `logger.warning("Tool %s failed: %s", ...)`
calls (already scattered through tools/*.py and chatbot.py) went essentially
nowhere useful. ``setup_logging`` fixes the handler; ``set_session_id`` is the
correlation id the plan calls out as existing (chatbot.py resolves one every
turn) and never logged — a context var, not a parameter threaded through
every call site, so nothing outside chatbot.py's entry point needs to know it
exists.
"""

from __future__ import annotations

import contextvars
import json
import logging
import socket
from urllib.parse import urlparse

from config import Settings, load_settings

logger = logging.getLogger(__name__)

_configured = False
_status = "not configured"

_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")

# Every attribute a bare logging.LogRecord already carries, plus the two this
# module adds itself — anything else on a record came from a caller's own
# `extra={...}` and should be surfaced, not silently dropped.
_STANDARD_LOG_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "session_id",
}


def set_session_id(session_id: str) -> None:
    """Attach `session_id` to every log record emitted from here on, in this
    context. A context var rather than a global: Streamlit can run more than
    one session's code in the same process, and each must log its own id."""
    _session_id.set(session_id)


class _SessionIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _session_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Any `extra={...}` field a caller passes
    (e.g. node_latencies) is included automatically, not just the fixed set
    below — this is what lets a single `logger.info("turn complete", extra=
    {"node_latencies": ..., "route": ...})` call carry per-stage retrieval
    timing into the logs without a bespoke schema for it."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        session_id = getattr(record, "session_id", "")
        if session_id:
            payload["session_id"] = session_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_logging_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Attach a structured JSON handler to the root logger. Idempotent and
    safe to call from more than one entry point (chatbot.py, eval/run_eval.py)
    without doubling handlers or output. A separate flag from setup_tracing's
    _configured: the two are independent, and sharing one would make either
    call wrongly think it had already run the other's setup.
    """
    global _logging_configured
    if _logging_configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_SessionIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _logging_configured = True


def tracing_status() -> str:
    """Human-readable state, surfaced in the UI so it is never assumed."""
    return _status


def _is_local(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _reachable(endpoint: str, timeout: float = 0.4) -> bool:
    parsed = urlparse(endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_tracing(settings: Settings | None = None) -> bool:
    """Register the tracer provider and instrument LangChain/LangGraph.

    Call before the LangChain objects are constructed — the instrumentor patches
    the callback manager, so anything built earlier is never traced. Returns
    True when tracing is live. Safe to call more than once.
    """
    global _configured, _status
    if _configured:
        return _status.startswith("tracing")

    # Through Settings, which loads .env — reading os.environ here would see an
    # unset endpoint and silently fall back to localhost.
    cfg = settings or load_settings()
    endpoint = cfg.phoenix_endpoint
    project = cfg.phoenix_project

    # A local collector that is not running produces an endless retry log from
    # the exporter, which is what anyone cloning this repo sees by default.
    # Check the socket first and stay quiet if nothing is listening. Remote
    # endpoints are not probed: configuring one is a statement of intent.
    if _is_local(endpoint) and not _reachable(endpoint):
        _configured, _status = True, f"disabled (no collector at {endpoint})"
        logger.info("Tracing disabled: nothing listening at %s", endpoint)
        return False

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError as exc:
        _configured, _status = True, "disabled (tracing packages not installed)"
        logger.info("Tracing disabled: %s", exc)
        return False

    try:
        tracer_provider = register(
            project_name=project,
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            batch=True,
            auto_instrument=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    except Exception as exc:
        # An unreachable collector must never take the app down with it.
        _configured, _status = True, f"disabled ({type(exc).__name__})"
        logger.warning("Tracing disabled — could not reach %s: %s", endpoint, exc)
        return False

    where = "Phoenix Cloud" if "localhost" not in endpoint else "local Phoenix"
    _configured, _status = True, f"tracing to {where} — project {project}"
    logger.info("Tracing enabled: %s", _status)
    return True
