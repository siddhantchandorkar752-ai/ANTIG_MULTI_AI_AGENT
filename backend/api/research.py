"""
OMEGA RESEARCH GRID — Research API Router
Core REST endpoints for research session management.
"""
from __future__ import annotations

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Path, status
from pydantic import BaseModel, Field

from backend.core.schemas import (
    CostSnapshot,
    ExecutionState,
    OrchestratorState,
    ResearchObjective,
)
from backend.orchestration.graph import get_orchestrator

router = APIRouter()
logger = structlog.get_logger("research_api")

# In-memory session store (replace with Redis in production)
_sessions: dict[str, OrchestratorState] = {}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=2000, description="Research query")
    depth: int = Field(default=3, ge=1, le=5, description="Research depth (1=shallow, 5=exhaustive)")
    max_sources: int = Field(default=20, ge=5, le=100)
    cost_budget_usd: float = Field(default=2.0, ge=0.1, le=20.0)
    token_budget: int = Field(default=80_000, ge=10_000, le=500_000)


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    iteration: int
    chunks_collected: int
    errors: list[str]
    cost_usd: float
    confidence: float


class ReportResponse(BaseModel):
    session_id: str
    title: str
    executive_summary: str
    abstract: str
    introduction: str
    methodology: str
    core_analysis: str
    technical_breakdown: str
    key_findings: list[str]
    counterarguments: str
    risks: str
    future_outlook: str
    conclusion: str
    confidence_score: float
    critique_scores: dict
    markdown_content: str
    references_count: int
    iteration: int


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/start",
    response_model=ResearchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new research session",
    description="Launches an autonomous research session. Runs asynchronously.",
)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    objective = ResearchObjective(
        raw_query=request.query,
        depth=request.depth,
        max_sources=request.max_sources,
        cost_budget_usd=request.cost_budget_usd,
        token_budget=request.token_budget,
    )
    session_id = objective.session_id
    logger.info("research_start_requested", session_id=session_id, query=request.query[:100])

    async def _run() -> None:
        orchestrator = get_orchestrator()
        final_state = await orchestrator.run(objective)
        _sessions[session_id] = final_state

    background_tasks.add_task(_run)

    return ResearchResponse(
        session_id=session_id,
        status="accepted",
        message=f"Research session started. Connect to ws://host/ws/{session_id} for live updates.",
    )


@router.get(
    "/{session_id}/status",
    response_model=SessionStatusResponse,
    summary="Get session execution status",
)
async def get_session_status(
    session_id: Annotated[str, Path(description="Research session ID")],
) -> SessionStatusResponse:
    # Check active sessions first
    orchestrator = get_orchestrator()
    live_state = orchestrator.get_session_state(session_id)
    state = live_state or _sessions.get(session_id)

    if not state:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionStatusResponse(
        session_id=session_id,
        state=state.current_state,
        iteration=state.iteration,
        chunks_collected=len(state.knowledge_chunks),
        errors=state.errors[-5:],
        cost_usd=state.cost_snapshot.total_cost_usd if state.cost_snapshot else 0.0,
        confidence=state.report.confidence_score if state.report else 0.0,
    )


@router.get(
    "/{session_id}/report",
    response_model=ReportResponse,
    summary="Retrieve the final research report",
)
async def get_report(
    session_id: Annotated[str, Path(description="Research session ID")],
) -> ReportResponse:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found or still running")
    if not state.report:
        raise HTTPException(status_code=204, detail="Report not yet available")

    report = state.report
    return ReportResponse(
        session_id=session_id,
        title=report.title,
        executive_summary=report.executive_summary,
        abstract=report.abstract,
        introduction=report.introduction,
        methodology=report.methodology,
        core_analysis=report.core_analysis,
        technical_breakdown=report.technical_breakdown,
        key_findings=report.key_findings,
        counterarguments=report.counterarguments,
        risks=report.risks,
        future_outlook=report.future_outlook,
        conclusion=report.conclusion,
        confidence_score=report.confidence_score,
        critique_scores=report.critique_scores,
        markdown_content=report.markdown_content,
        references_count=len(report.references),
        iteration=report.iteration,
    )


@router.get(
    "/{session_id}/cost",
    summary="Get cost breakdown for a session",
)
async def get_cost(session_id: str) -> dict:
    state = _sessions.get(session_id)
    if not state or not state.cost_snapshot:
        raise HTTPException(status_code=404, detail="Session not found")
    snap = state.cost_snapshot
    return {
        "session_id": session_id,
        "total_tokens": snap.total_tokens,
        "total_cost_usd": round(snap.total_cost_usd, 6),
        "budget_remaining_usd": round(snap.budget_remaining_usd, 6),
        "budget_exhausted": snap.budget_exhausted,
        "agent_breakdown": snap.agent_breakdown,
    }


@router.get(
    "/{session_id}/audit",
    summary="Get governance audit log",
)
async def get_audit_log(session_id: str) -> dict:
    # In production, read from PostgreSQL audit_logs table
    return {
        "session_id": session_id,
        "message": "Audit log available in PostgreSQL audit_logs table",
    }


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a completed session",
)
async def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
