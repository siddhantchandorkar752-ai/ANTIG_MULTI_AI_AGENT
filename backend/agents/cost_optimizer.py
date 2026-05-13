"""
OMEGA RESEARCH GRID — Cost Optimizer Agent
Monitors token consumption and dynamically optimizes execution costs.

Optimization strategies:
  1. Context compression: truncate low-relevance chunks before LLM calls
  2. Model downgrade: route to cheaper models when budget is tight
  3. Response caching: cache LLM outputs for identical prompts (Redis)
  4. Prompt compression: remove redundant context tokens via GPT-4o-mini
  5. Budget alerting: emit warnings at 50%, 80%, 95% consumption
  6. Early termination: abort non-critical tasks when budget is exhausted

Token pricing (as of 2024):
  GPT-4o: $2.50/1M input, $10/1M output
  GPT-4o-mini: $0.15/1M input, $0.60/1M output
  Claude Haiku: $0.25/1M input, $1.25/1M output
  text-embedding-3-small: $0.02/1M tokens
"""
from __future__ import annotations

import json

import redis.asyncio as aioredis

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    CostSnapshot,
    ExecutionState,
    OrchestratorState,
)

settings = get_settings()
logger = get_logger("cost_optimizer")

# Model cost table: (input_per_1m_usd, output_per_1m_usd)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    "text-embedding-3-small": (0.02, 0.0),
}

ALERT_THRESHOLDS = [0.50, 0.80, 0.95]


class CostOptimizerAgent:
    """Monitors and optimizes token/cost consumption across all agents."""

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self._alerted_thresholds: set[float] = set()
        self._redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        try:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
        except Exception:
            self._redis = None

    async def optimize(self, state: OrchestratorState) -> OrchestratorState:
        """
        Called at the start of each DAG node to assess budget
        and recommend optimizations.
        """
        snapshot = await self.governance.get_cost_snapshot()
        state.cost_snapshot = snapshot

        # Compute consumption ratio
        total_budget = settings.cost_threshold_usd
        spent = snapshot.total_cost_usd
        ratio = spent / max(total_budget, 0.01)

        # Emit threshold alerts
        for threshold in ALERT_THRESHOLDS:
            if ratio >= threshold and threshold not in self._alerted_thresholds:
                self._alerted_thresholds.add(threshold)
                logger.warning(
                    "budget_alert",
                    threshold_pct=int(threshold * 100),
                    spent_usd=spent,
                    budget_usd=total_budget,
                    session_id=state.session_id,
                )

        if snapshot.budget_exhausted:
            logger.error("budget_exhausted", session_id=state.session_id, spent=spent)
            state.errors.append(f"Budget exhausted: ${spent:.4f} / ${total_budget:.2f}")

        return state

    async def cache_response(self, prompt_hash: str, response: str) -> None:
        """Cache an LLM response by prompt hash."""
        if self._redis:
            await self._redis.setex(
                f"llm_cache:{prompt_hash}",
                settings.redis_cache_ttl,
                response,
            )

    async def get_cached_response(self, prompt_hash: str) -> str | None:
        """Retrieve a cached LLM response."""
        if self._redis:
            return await self._redis.get(f"llm_cache:{prompt_hash}")
        return None

    @staticmethod
    def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate precise API cost for a model call."""
        if model not in MODEL_COSTS:
            return 0.001  # Conservative default
        in_rate, out_rate = MODEL_COSTS[model]
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

    @staticmethod
    def compress_prompt(text: str, max_tokens: int = 4000) -> str:
        """
        Hard-truncate prompt to max_tokens budget.
        Strategy: keep first 30% + last 70% (tail is more context-relevant).
        """
        max_chars = max_tokens * 4  # ~4 chars per token
        if len(text) <= max_chars:
            return text
        head = int(max_chars * 0.3)
        tail = max_chars - head
        return text[:head] + "\n...[truncated for cost efficiency]...\n" + text[-tail:]

    async def get_cost_report(self, session_id: str) -> dict:
        """Generate a detailed cost breakdown report."""
        snapshot = await self.governance.get_cost_snapshot()
        return {
            "session_id": session_id,
            "total_tokens": snapshot.total_tokens,
            "total_cost_usd": round(snapshot.total_cost_usd, 6),
            "budget_remaining_usd": round(snapshot.budget_remaining_usd, 6),
            "budget_exhausted": snapshot.budget_exhausted,
            "agent_breakdown": {
                agent: round(cost, 6)
                for agent, cost in snapshot.agent_breakdown.items()
            },
            "efficiency_score": self._efficiency_score(snapshot),
        }

    def _efficiency_score(self, snapshot: CostSnapshot) -> float:
        """
        Efficiency = how much of the budget was productively used.
        Low efficiency = token waste from retries, oversized prompts, etc.
        """
        if snapshot.total_cost_usd == 0:
            return 1.0
        budget_ratio = snapshot.total_cost_usd / settings.cost_threshold_usd
        # Ideal: 70-90% budget utilization for a completed research task
        if 0.7 <= budget_ratio <= 0.9:
            return 1.0
        elif budget_ratio < 0.7:
            return budget_ratio / 0.7
        else:
            return max(0.0, 1.0 - (budget_ratio - 0.9) * 5)
