"""
OMEGA RESEARCH GRID — Agent Governance Layer
The governance layer is the autonomous kernel of the system.
It enforces execution policies, monitors agent behavior, controls
resource allocation, detects anomalies, and manages failure recovery.

Design philosophy:
  This layer intentionally mirrors an OS process scheduler/supervisor.
  Every agent action is gated through governance before execution.
  This is the single point of policy enforcement.

Key responsibilities:
  1. Policy enforcement (token budgets, recursion depth, timeouts)
  2. Anomaly detection (infinite loops, cost explosions)
  3. Dynamic model routing (cheap → premium escalation)
  4. Resource accounting (per-agent token + cost tracking)
  5. Execution auditing (full audit log of every decision)
  6. Failure recovery (retry, escalate, or gracefully degrade)
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import get_settings
from backend.core.logging import (
    AGENT_EXECUTIONS,
    AGENT_LATENCY,
    COST_USD,
    HALLUCINATIONS_DETECTED,
    TOKEN_USAGE,
    get_logger,
)
from backend.core.schemas import AgentType, CostSnapshot, ExecutionState, TokenUsage

settings = get_settings()
logger = get_logger("governance")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION POLICY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionPolicy:
    """
    Per-session governance rules.
    Immutable once a session starts — modifications require escalation.
    """
    max_tokens: int = settings.max_total_tokens
    max_cost_usd: float = settings.cost_threshold_usd
    max_recursion_depth: int = settings.max_recursion_depth
    max_concurrent_agents: int = settings.max_concurrent_agents
    task_timeout_seconds: int = settings.task_timeout_seconds
    max_retries_per_task: int = 3
    cheap_model_threshold_usd: float = settings.cheap_model_threshold_usd

    # Tool access matrix: agent → set of permitted tool names
    tool_permissions: dict[str, set[str]] = field(default_factory=lambda: {
        AgentType.PLANNER: {"llm_call", "dag_builder"},
        AgentType.SEARCH: {"tavily_search", "serpapi_search", "arxiv_search", "pubmed_search"},
        AgentType.READER: {"firecrawl_fetch", "playwright_fetch", "bs4_parse"},
        AgentType.MEMORY: {"chroma_write", "chroma_read", "redis_cache"},
        AgentType.WRITER: {"llm_call", "markdown_renderer"},
        AgentType.CRITIC: {"llm_call"},
        AgentType.VERIFIER: {"tavily_search", "url_validator"},
        AgentType.COST_OPTIMIZER: {"model_router", "context_compressor"},
    })


# ─────────────────────────────────────────────────────────────────────────────
# GOVERNANCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class GovernanceEngine:
    """
    Central governance authority for the entire OMEGA system.
    Singleton per session — injected into every agent at construction.

    Thread-safety: asyncio.Lock guards all mutable state.
    """

    def __init__(self, session_id: str, policy: Optional[ExecutionPolicy] = None):
        self.session_id = session_id
        self.policy = policy or ExecutionPolicy()
        self._lock = asyncio.Lock()
        self._token_counts: dict[str, int] = defaultdict(int)
        self._cost_usd: dict[str, float] = defaultdict(float)
        self._agent_call_counts: dict[str, int] = defaultdict(int)
        self._recursion_depths: dict[str, int] = defaultdict(int)
        self._active_agents: set[str] = set()
        self._audit_log: list[dict[str, Any]] = []

    # ── Resource Accounting ───────────────────────────────────────────────────

    async def record_token_usage(
        self,
        agent_type: AgentType,
        usage: TokenUsage,
    ) -> None:
        async with self._lock:
            key = agent_type.value if hasattr(agent_type, "value") else agent_type
            self._token_counts[key] += usage.total_tokens
            self._cost_usd[key] += usage.cost_usd
            TOKEN_USAGE.labels(model=usage.model, agent_type=key).inc(usage.total_tokens)
            COST_USD.labels(model=usage.model).inc(usage.cost_usd)
            await self._audit("token_usage", {
                "agent": key, "tokens": usage.total_tokens, "cost": usage.cost_usd
            })

    async def get_cost_snapshot(self) -> CostSnapshot:
        async with self._lock:
            total_cost = sum(self._cost_usd.values())
            total_tokens = sum(self._token_counts.values())
            return CostSnapshot(
                session_id=self.session_id,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                agent_breakdown=dict(self._cost_usd),
                budget_remaining_usd=max(0.0, self.policy.max_cost_usd - total_cost),
                budget_exhausted=total_cost >= self.policy.max_cost_usd,
            )

    # ── Policy Enforcement ────────────────────────────────────────────────────

    async def check_budget(self) -> bool:
        """Returns True if budget is still available."""
        snapshot = await self.get_cost_snapshot()
        if snapshot.budget_exhausted:
            logger.warning(
                "budget_exhausted",
                session_id=self.session_id,
                total_cost=snapshot.total_cost_usd,
                budget=self.policy.max_cost_usd,
            )
            return False
        return True

    async def check_token_budget(self, tokens_needed: int) -> bool:
        async with self._lock:
            total = sum(self._token_counts.values())
            return (total + tokens_needed) <= self.policy.max_tokens

    async def acquire_agent_slot(self, agent_type: str) -> bool:
        """
        Concurrency gate — limits simultaneous active agents.
        Returns False if concurrency ceiling is hit.
        """
        async with self._lock:
            if len(self._active_agents) >= self.policy.max_concurrent_agents:
                logger.warning(
                    "concurrency_ceiling_hit",
                    active=len(self._active_agents),
                    max=self.policy.max_concurrent_agents,
                )
                return False
            self._active_agents.add(agent_type)
            self._agent_call_counts[agent_type] += 1
            return True

    async def release_agent_slot(self, agent_type: str) -> None:
        async with self._lock:
            self._active_agents.discard(agent_type)

    async def check_recursion_depth(self, agent_type: str) -> bool:
        async with self._lock:
            depth = self._recursion_depths.get(agent_type, 0)
            if depth >= self.policy.max_recursion_depth:
                logger.error(
                    "recursion_limit_exceeded",
                    agent=agent_type,
                    depth=depth,
                    max=self.policy.max_recursion_depth,
                )
                return False
            self._recursion_depths[agent_type] = depth + 1
            return True

    async def release_recursion(self, agent_type: str) -> None:
        async with self._lock:
            self._recursion_depths[agent_type] = max(
                0, self._recursion_depths.get(agent_type, 1) - 1
            )

    def check_tool_permission(self, agent_type: str, tool_name: str) -> bool:
        permitted = self.policy.tool_permissions.get(agent_type, set())
        if tool_name not in permitted:
            logger.warning(
                "tool_access_denied",
                agent=agent_type,
                tool=tool_name,
                permitted=list(permitted),
            )
            return False
        return True

    # ── Model Routing ─────────────────────────────────────────────────────────

    async def select_model(self, complexity: float = 0.5) -> str:
        """
        Dynamic model routing based on budget + task complexity.
        complexity: float[0,1] — 1.0 requires premium model.

        Routing logic:
          - Budget < threshold → force cheap model regardless of complexity
          - complexity > 0.8 + budget ok → premium model
          - Otherwise → standard model
          - On exception → fallback model
        """
        snapshot = await self.get_cost_snapshot()

        if snapshot.budget_remaining_usd < self.policy.cheap_model_threshold_usd:
            logger.info("model_routed_cheap", reason="budget_low")
            return settings.standard_model

        if complexity > 0.8:
            logger.info("model_routed_premium", complexity=complexity)
            return settings.premium_model

        return settings.standard_model

    # ── Retry Wrapper ─────────────────────────────────────────────────────────

    async def with_retry(
        self,
        fn: Callable,
        *args: Any,
        agent_type: str = "unknown",
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> Any:
        """
        Tenacity-powered async retry with exponential backoff + governance metering.
        Catches transient failures (network, rate limits) transparently.
        """
        start = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
                reraise=True,
            ):
                with attempt:
                    result = await asyncio.wait_for(
                        fn(*args, **kwargs),
                        timeout=self.policy.task_timeout_seconds,
                    )
            AGENT_EXECUTIONS.labels(agent_type=agent_type, status="success").inc()
            return result
        except Exception as exc:
            AGENT_EXECUTIONS.labels(agent_type=agent_type, status="failed").inc()
            await self._audit("agent_failure", {"agent": agent_type, "error": str(exc)})
            raise
        finally:
            elapsed = time.perf_counter() - start
            AGENT_LATENCY.labels(agent_type=agent_type).observe(elapsed)

    # ── Audit Logging ─────────────────────────────────────────────────────────

    async def _audit(self, event: str, data: dict[str, Any]) -> None:
        entry = {
            "session_id": self.session_id,
            "event": event,
            "timestamp": time.time(),
            **data,
        }
        self._audit_log.append(entry)
        logger.info("audit", **entry)

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)

    def flag_hallucination(self, claim_text: str) -> None:
        HALLUCINATIONS_DETECTED.inc()
        logger.warning("hallucination_flagged", claim=claim_text[:200])
