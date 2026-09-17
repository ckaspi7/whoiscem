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

from config import Settings, load_settings

logger = logging.getLogger(__name__)

_configured = False
_status = "not configured"


def tracing_status() -> str:
    """Human-readable state, surfaced in the UI so it is never assumed."""
    return _status


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
