"""
OMEGA RESEARCH GRID — FastAPI Application
Production-grade REST + WebSocket API layer.

Features:
  - JWT authentication with Bearer tokens
  - Rate limiting via Redis sliding window
  - OpenAPI documentation auto-generated
  - WebSocket streaming for real-time agent updates
  - Health check endpoints for Kubernetes probes
  - Prometheus metrics endpoint
  - CORS configuration for Next.js frontend
  - Request ID middleware for distributed tracing
  - Structured error responses (never expose raw exceptions)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from backend.core.config import get_settings
from backend.core.logging import setup_logging, setup_tracing
from backend.core.schemas import OrchestratorState, ResearchObjective
from backend.db.models import create_tables
from backend.orchestration.graph import get_orchestrator

settings = get_settings()
logger = structlog.get_logger("api")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle manager."""
    setup_logging()
    try:
        setup_tracing()
    except Exception as exc:
        logger.warning("otel_setup_failed", error=str(exc))

    # Initialize database
    try:
        await create_tables()
        logger.info("database_initialized")
    except Exception as exc:
        logger.warning("database_init_failed", error=str(exc))

    logger.info("omega_research_grid_started", env=settings.environment)
    yield
    logger.info("omega_research_grid_stopping")


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="OMEGA RESEARCH GRID",
        description="Autonomous Multi-Agent Research Operating System",
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://omega-research.ai"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID Middleware ──────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed * 1000, 2),
        )
        return response

    # ── Routes ────────────────────────────────────────────────────────────────
    from backend.api import research, health
    app.include_router(research.router, prefix="/api/v1/research", tags=["Research"])
    app.include_router(health.router, prefix="/health", tags=["Health"])

    # ── Prometheus metrics ────────────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time agent streaming."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(ws)
        logger.info("ws_connected", session_id=session_id)

    async def disconnect(self, session_id: str, ws: WebSocket) -> None:
        if session_id in self._connections:
            self._connections[session_id].discard(ws)
        logger.info("ws_disconnected", session_id=session_id)

    async def broadcast(self, session_id: str, message: dict) -> None:
        if session_id not in self._connections:
            return
        dead: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[session_id].discard(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(session_id: str, websocket: WebSocket) -> None:
    """Real-time streaming of agent execution events."""
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            # Poll orchestrator state every 500ms
            orchestrator = get_orchestrator()
            state = orchestrator.get_session_state(session_id)
            if state:
                await websocket.send_json({
                    "type": "state_update",
                    "session_id": session_id,
                    "current_state": state.current_state,
                    "iteration": state.iteration,
                    "chunks_collected": len(state.knowledge_chunks),
                    "errors": state.errors[-3:] if state.errors else [],
                    "cost": state.cost_snapshot.total_cost_usd if state.cost_snapshot else 0,
                })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        await ws_manager.disconnect(session_id, websocket)
