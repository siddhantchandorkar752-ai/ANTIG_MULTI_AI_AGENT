"""
OMEGA RESEARCH GRID — Planner Agent
Responsible for decomposing research objectives into executable DAGs.

Design: Uses GPT-4o with structured output (JSON mode) to generate
the execution graph. The planner is the first and most critical agent —
a bad plan produces cascading failures downstream.

Key architectural decisions:
  1. Structured JSON output → forces deterministic DAG structure
  2. Recursive decomposition → handles complex nested research goals
  3. Dependency injection of GovernanceEngine → all plans are cost-aware
  4. Complexity estimation per subtask → drives model routing downstream
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    ExecutionDAG,
    ExecutionState,
    OrchestratorState,
    ResearchObjective,
    TaskNode,
    TokenUsage,
)

settings = get_settings()
logger = get_logger("planner_agent")

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent of the OMEGA RESEARCH GRID.
Your job is to decompose a research objective into an executable task graph.

Output ONLY valid JSON matching this schema:
{
  "title": "string — concise research title",
  "normalized_query": "string — cleaned research question",
  "domain": "string — research domain (e.g., AI, medicine, finance)",
  "estimated_tokens": number,
  "estimated_cost_usd": number,
  "tasks": [
    {
      "task_name": "string",
      "agent_type": "search|reader|memory|writer|critic|verifier",
      "description": "string — what this task does",
      "dependencies": ["task_name_1", "task_name_2"],
      "priority": 1-10,
      "complexity": 0.0-1.0,
      "estimated_tokens": number
    }
  ]
}

Rules:
- Create a logical dependency chain: search → read → write → critique → verify
- Every chain MUST end with a writer task followed by critic and verifier
- Keep task names snake_case and unique
- Complexity 1.0 = requires premium model, 0.0 = use cheap model
- Estimate tokens conservatively (overestimate by 20%)
- You MUST include at least: 2 search tasks, 2 reader tasks, 1 writer, 1 critic, 1 verifier
"""


class PlannerAgent:
    """
    The Planner Agent constructs the research execution DAG.
    It is the entry point of every research session.
    """

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self.llm = ChatOpenAI(
            model=settings.premium_model,
            temperature=0.1,  # Low temp for deterministic planning
            response_format={"type": "json_object"},
            api_key=settings.openai_api_key.get_secret_value(),
        )

    async def plan(self, state: OrchestratorState) -> OrchestratorState:
        """
        Entry point called by the LangGraph orchestrator.
        Returns enriched state with populated DAG.
        """
        logger.info("planner_start", session_id=state.session_id)
        state.current_state = ExecutionState.PLANNING

        if not await self.governance.check_budget():
            state.errors.append("Budget exhausted before planning")
            state.current_state = ExecutionState.FAILED
            return state

        if not await self.governance.acquire_agent_slot(AgentType.PLANNER):
            state.errors.append("Concurrency ceiling hit — planner could not acquire slot")
            state.current_state = ExecutionState.FAILED
            return state

        try:
            dag = await self.governance.with_retry(
                self._build_dag,
                state.objective,
                agent_type=AgentType.PLANNER,
            )
            state.dag = dag
            state.current_state = ExecutionState.SEARCHING
            logger.info(
                "planner_complete",
                session_id=state.session_id,
                node_count=len(dag.nodes),
                estimated_cost=dag.total_cost_usd,
            )
        except Exception as exc:
            logger.error("planner_failed", error=str(exc))
            state.errors.append(f"Planner failed: {exc}")
            state.current_state = ExecutionState.FAILED
        finally:
            await self.governance.release_agent_slot(AgentType.PLANNER)

        return state

    async def _build_dag(self, objective: ResearchObjective) -> ExecutionDAG:
        """Core planning logic — calls LLM and constructs typed DAG."""
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Research Objective: {objective.raw_query}
Research Depth: {objective.depth}/5
Max Sources: {objective.max_sources}
Token Budget: {objective.token_budget:,}
Cost Budget: ${objective.cost_budget_usd:.2f}

Generate the execution task graph now.
"""),
        ]

        response = await self.llm.ainvoke(messages)
        raw = response.content

        # Track token usage through governance
        usage = TokenUsage(
            prompt_tokens=response.usage_metadata.get("input_tokens", 0),
            completion_tokens=response.usage_metadata.get("output_tokens", 0),
            model=settings.premium_model,
            cost_usd=self._estimate_cost(
                response.usage_metadata.get("input_tokens", 0),
                response.usage_metadata.get("output_tokens", 0),
                settings.premium_model,
            ),
        )
        await self.governance.record_token_usage(AgentType.PLANNER, usage)

        plan = json.loads(raw)
        return self._parse_plan_to_dag(plan, objective)

    def _parse_plan_to_dag(
        self, plan: dict[str, Any], objective: ResearchObjective
    ) -> ExecutionDAG:
        """Convert LLM JSON output → typed ExecutionDAG."""
        # Enrich the objective with planner intelligence
        objective.normalized_query = plan.get("normalized_query", objective.raw_query)
        objective.domain = plan.get("domain", "general")

        dag = ExecutionDAG(
            session_id=objective.session_id,
            objective=objective,
            total_estimated_tokens=plan.get("estimated_tokens", 50_000),
            total_cost_usd=plan.get("estimated_cost_usd", 1.0),
        )

        # Map task_name → task_id for dependency resolution
        name_to_id: dict[str, str] = {}

        for task_spec in plan.get("tasks", []):
            node = TaskNode(
                session_id=objective.session_id,
                agent_type=AgentType(task_spec["agent_type"]),
                task_name=task_spec["task_name"],
                description=task_spec["description"],
                priority=task_spec.get("priority", 5),
                timeout_seconds=120,
                inputs={
                    "complexity": task_spec.get("complexity", 0.5),
                    "estimated_tokens": task_spec.get("estimated_tokens", 5000),
                },
            )
            name_to_id[task_spec["task_name"]] = node.task_id

            # Resolve dependency names → IDs
            node.dependencies = [
                name_to_id[dep]
                for dep in task_spec.get("dependencies", [])
                if dep in name_to_id
            ]
            dag.add_node(node)

        return dag

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Approximate cost estimation based on OpenAI pricing."""
        pricing = {
            "gpt-4o": (0.0025 / 1000, 0.01 / 1000),
            "gpt-4o-mini": (0.00015 / 1000, 0.0006 / 1000),
        }
        p_rate, c_rate = pricing.get(model, (0.001 / 1000, 0.003 / 1000))
        return prompt_tokens * p_rate + completion_tokens * c_rate
