"""
OMEGA RESEARCH GRID — LangGraph Orchestration Engine
The central execution engine that coordinates all agents through a typed state graph.

Architecture decision: LangGraph over CrewAI/AutoGen
  - LangGraph: deterministic DAG, full state control, async-native, checkpointing
  - CrewAI: higher-level but less controllable, limited state inspection
  - AutoGen: conversation-based, harder to make deterministic, debugging is difficult
  - Semantic Kernel: Microsoft ecosystem lock-in, less pythonic

LangGraph wins because:
  1. Explicit state machine — every transition is observable
  2. Built-in checkpointing — resumable workflows after failures
  3. Async-first — all nodes can be awaited concurrently where dependencies allow
  4. Full graph control — can implement rollback, retry, conditional branching
  5. Native LangSmith integration — automatic trace capture

Graph topology:
  planner → search → memory_store → reader → memory_store → cost_check
          ↓ (parallel if needed)
          writer → critic → [revise loop | verifier] → finalizer

State machine transitions:
  IDLE → PLANNING → SEARCHING → READING → WRITING
  WRITING → CRITIQUING → REVISING (loop) or VERIFYING
  VERIFYING → FINALIZING → COMPLETED
  Any state → FAILED (on unrecoverable error)
  FAILED → RETRYING (governance-controlled)
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.cost_optimizer import CostOptimizerAgent
from backend.agents.critic import CriticAgent
from backend.agents.memory import MemoryAgent
from backend.agents.planner import PlannerAgent
from backend.agents.reader import ReaderAgent
from backend.agents.search import SearchAgent
from backend.agents.verifier import VerifierAgent
from backend.agents.writer import WriterAgent
from backend.core.config import get_settings
from backend.core.governance import ExecutionPolicy, GovernanceEngine
from backend.core.logging import ACTIVE_SESSIONS, get_logger
from backend.core.schemas import (
    ExecutionState,
    OrchestratorState,
    ResearchObjective,
)

settings = get_settings()
logger = get_logger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class OmegaOrchestrator:
    """
    The central orchestration engine of the OMEGA RESEARCH GRID.
    Manages LangGraph compilation, session lifecycle, and streaming.
    """

    def __init__(self):
        self._graphs: dict[str, CompiledStateGraph] = {}  # Per-session graphs
        self._active_sessions: dict[str, OrchestratorState] = {}

    def _build_graph(self, governance: GovernanceEngine) -> CompiledStateGraph:
        """
        Compile a new LangGraph for a research session.
        Each session gets its own graph with injected governance.
        """
        planner = PlannerAgent(governance)
        search = SearchAgent(governance)
        reader = ReaderAgent(governance)
        memory = MemoryAgent(governance)
        writer = WriterAgent(governance)
        critic = CriticAgent(governance)
        verifier = VerifierAgent(governance)
        cost_opt = CostOptimizerAgent(governance)

        graph = StateGraph(OrchestratorState)

        # ── Register all nodes ────────────────────────────────────────────────
        graph.add_node("planner", planner.plan)
        graph.add_node("search", search.search)
        graph.add_node("memory_store", memory.store)
        graph.add_node("reader", reader.read)
        graph.add_node("cost_check", cost_opt.optimize)
        graph.add_node("writer", writer.write)
        graph.add_node("critic", critic.critique)
        graph.add_node("verifier", verifier.verify)
        graph.add_node("finalizer", _finalize)

        # ── Define edges ──────────────────────────────────────────────────────
        graph.set_entry_point("planner")
        graph.add_edge("planner", "search")
        graph.add_edge("search", "memory_store")
        graph.add_edge("memory_store", "reader")
        graph.add_edge("reader", "cost_check")
        graph.add_edge("cost_check", "writer")
        graph.add_edge("writer", "critic")

        # ── Conditional routing from critic ───────────────────────────────────
        graph.add_conditional_edges(
            "critic",
            _route_after_critique,
            {
                "revise": "writer",      # Loop back for revision
                "verify": "verifier",    # Proceed to verification
                "finalize": "finalizer", # Budget exhausted — force finalize
            },
        )
        graph.add_edge("verifier", "finalizer")
        graph.add_edge("finalizer", END)

        return graph.compile()

    async def run(
        self,
        objective: ResearchObjective,
        on_event: Any = None,  # Callback for streaming events
    ) -> OrchestratorState:
        """
        Execute a complete research session.
        Returns the final OrchestratorState with report, critique, and cost data.
        """
        session_id = objective.session_id
        ACTIVE_SESSIONS.inc()
        logger.info("session_start", session_id=session_id, query=objective.raw_query[:100])

        # Per-session governance
        policy = ExecutionPolicy(
            max_cost_usd=objective.cost_budget_usd,
            max_tokens=objective.token_budget,
        )
        governance = GovernanceEngine(session_id=session_id, policy=policy)

        # Initialize memory connections
        memory = MemoryAgent(governance)
        await memory.initialize()

        # Build session graph
        graph = self._build_graph(governance)

        # Initial state
        initial_state = OrchestratorState(
            session_id=session_id,
            objective=objective,
            current_state=ExecutionState.IDLE,
            max_iterations=3,
        )
        self._active_sessions[session_id] = initial_state

        try:
            final_state: OrchestratorState = await graph.ainvoke(
                initial_state,
                config={
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": 20,
                },
            )

            # Attach cost metadata
            final_state.cost_snapshot = await governance.get_cost_snapshot()
            final_state.current_state = ExecutionState.COMPLETED

            logger.info(
                "session_complete",
                session_id=session_id,
                cost_usd=final_state.cost_snapshot.total_cost_usd,
                confidence=final_state.report.confidence_score if final_state.report else 0,
            )
            return final_state

        except Exception as exc:
            logger.error("session_failed", session_id=session_id, error=str(exc))
            initial_state.current_state = ExecutionState.FAILED
            initial_state.errors.append(str(exc))
            return initial_state
        finally:
            ACTIVE_SESSIONS.dec()
            self._active_sessions.pop(session_id, None)

    def get_active_sessions(self) -> list[str]:
        return list(self._active_sessions.keys())

    def get_session_state(self, session_id: str) -> OrchestratorState | None:
        return self._active_sessions.get(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def _finalize(state: OrchestratorState) -> OrchestratorState:
    """Terminal node — marks the session as complete."""
    state.current_state = ExecutionState.FINALIZING
    logger.info(
        "finalizing",
        session_id=state.session_id,
        has_report=state.report is not None,
        errors=len(state.errors),
    )
    return state


def _route_after_critique(
    state: OrchestratorState,
) -> Literal["revise", "verify", "finalize"]:
    """
    Conditional routing logic after the Critic agent runs.

    Routing rules:
      1. Budget exhausted → finalize immediately (cost protection)
      2. Score >= 0.82 OR iteration limit hit → verify (quality gate passed)
      3. Otherwise → revise (trigger another write-critique loop)
    """
    if state.cost_snapshot and state.cost_snapshot.budget_exhausted:
        logger.warning("routing_finalize_budget_exhausted")
        return "finalize"

    if state.critique is None:
        return "verify"

    if (
        state.critique.overall_score >= 0.82
        or state.iteration >= state.max_iterations - 1
    ):
        return "verify"

    return "revise"


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_orchestrator: OmegaOrchestrator | None = None


def get_orchestrator() -> OmegaOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OmegaOrchestrator()
    return _orchestrator
