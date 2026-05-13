"""
OMEGA RESEARCH GRID — Structured Logging & OpenTelemetry Setup
Every system action emits a structured log + OTEL span.
Rationale: observability is non-negotiable in autonomous systems.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from backend.core.config import get_settings

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────────────────

AGENT_EXECUTIONS = Counter(
    "omega_agent_executions_total",
    "Total agent node executions",
    ["agent_type", "status"],
)
AGENT_LATENCY = Histogram(
    "omega_agent_latency_seconds",
    "Agent execution latency",
    ["agent_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)
TOKEN_USAGE = Counter(
    "omega_tokens_total",
    "Total tokens consumed",
    ["model", "agent_type"],
)
COST_USD = Counter(
    "omega_cost_usd_total",
    "Total API cost in USD",
    ["model"],
)
ACTIVE_SESSIONS = Gauge(
    "omega_active_sessions",
    "Currently active research sessions",
)
MEMORY_OPERATIONS = Counter(
    "omega_memory_operations_total",
    "Vector store read/write operations",
    ["operation", "store"],
)
SEARCH_REQUESTS = Counter(
    "omega_search_requests_total",
    "Total search API calls",
    ["provider"],
)
HALLUCINATIONS_DETECTED = Counter(
    "omega_hallucinations_detected_total",
    "Hallucinations flagged by Critic agent",
)


# ─────────────────────────────────────────────────────────────────────────────
# OPENTELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def setup_tracing() -> trace.Tracer:
    """Configure OTEL SDK with OTLP gRPC exporter."""
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTLOG CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure structlog for JSON-formatted structured logging."""
    log_level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **ctx: Any) -> structlog.BoundLogger:
    """Return a context-enriched logger."""
    return structlog.get_logger(name).bind(**ctx)
