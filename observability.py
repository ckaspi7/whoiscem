"""OpenTelemetry tracing, exported to Arize Phoenix.

Instrumentation is vendor-neutral: OpenInference semantic conventions over
OpenTelemetry. Phoenix is the collector, chosen by configuration the same way
the vector store and session memory are, and it runs locally with no account:

    pip install arize-phoenix && phoenix serve     # http://localhost:6006

Set PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY to export to Phoenix Cloud
instead. Tracing never blocks the app — if the collector is unreachable or the
packages are missing, this degrades to a no-op and says so once.
"""

from __future__ import annotations

import logging
import socket
from urllib.parse import urlparse

from config import Settings, load_settings

logger = logging.getLogger(__name__)

_configured = False
_status = "not configured"


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
