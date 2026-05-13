"""
OMEGA RESEARCH GRID — Critic Agent
Adversarial evaluator that scores and improves report quality.

Design philosophy:
  The Critic operates in a separate LLM context from the Writer.
  It has NO access to the Writer's internal reasoning — it evaluates
  only the final output, simulating a genuine peer reviewer.

  This adversarial separation is critical for catching:
  - Subtle hallucinations the Writer incorporated confidently
  - Circular reasoning (citing itself)
  - Missing counterarguments
  - Overconfident claims with weak evidence

Self-reflection loop:
  After initial critique, a second LLM call performs meta-critique
  ("Is this critique itself rigorous?") to catch superficial reviews.

Termination condition:
  If critique.overall_score >= 0.82 → FINALIZING
  If iteration >= max_iterations → FINALIZING (force exit)
  Otherwise → REVISING (trigger another Writer pass)
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    CritiqueResult,
    ExecutionState,
    OrchestratorState,
    TokenUsage,
)

settings = get_settings()
logger = get_logger("critic_agent")

CRITIQUE_SYSTEM = """You are the Critic Agent — a rigorous peer reviewer.
Evaluate the research report on these dimensions (each scored 0.0-1.0):

1. factuality_score: Are all claims supported by evidence? Flag speculative statements.
2. coherence_score: Does the narrative flow logically? Are sections consistent?
3. depth_score: Is the analysis sufficiently deep or superficial?
4. rigor_score: Is the methodology sound? Are limitations acknowledged?
5. source_quality_score: Are sources authoritative and diverse?
6. reasoning_score: Is the logical reasoning sound? Any fallacies?
7. citation_accuracy_score: Do citations actually support the claims made?

Return ONLY valid JSON:
{
  "factuality_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "depth_score": 0.0-1.0,
  "rigor_score": 0.0-1.0,
  "source_quality_score": 0.0-1.0,
  "reasoning_score": 0.0-1.0,
  "citation_accuracy_score": 0.0-1.0,
  "issues_found": ["issue1", "issue2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "hallucinations_detected": ["claim that appears unsupported"],
  "needs_revision": true/false
}

Be harsh. A score of 0.9+ requires near-perfect quality."""

QUALITY_THRESHOLD = 0.82


class CriticAgent:
    """Adversarial critic with hallucination detection and self-reflection."""

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self.llm = ChatOpenAI(
            model=settings.premium_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            api_key=settings.openai_api_key.get_secret_value(),
        )

    async def critique(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("critic_start", session_id=state.session_id, iteration=state.iteration)
        state.current_state = ExecutionState.CRITIQUING

        if not state.report:
            state.errors.append("Critic invoked with no report to evaluate")
            state.current_state = ExecutionState.FAILED
            return state

        if not await self.governance.acquire_agent_slot(AgentType.CRITIC):
            state.current_state = ExecutionState.FINALIZING
            return state

        try:
            critique = await self.governance.with_retry(
                self._run_critique,
                state,
                agent_type=AgentType.CRITIC,
            )
            state.critique = critique

            # Governance: flag detected hallucinations
            for hallucination in critique.hallucinations_detected:
                self.governance.flag_hallucination(hallucination)

            # State routing decision
            if (
                critique.overall_score >= QUALITY_THRESHOLD
                or state.iteration >= state.max_iterations - 1
            ):
                state.current_state = ExecutionState.VERIFYING
                logger.info(
                    "critic_approved",
                    score=critique.overall_score,
                    iteration=state.iteration,
                )
            else:
                state.current_state = ExecutionState.REVISING
                state.iteration += 1
                logger.info(
                    "critic_revision_requested",
                    score=critique.overall_score,
                    issues=len(critique.issues_found),
                )
        except Exception as exc:
            logger.error("critic_failed", error=str(exc))
            state.errors.append(f"Critic failed: {exc}")
            state.current_state = ExecutionState.VERIFYING  # Graceful degradation
        finally:
            await self.governance.release_agent_slot(AgentType.CRITIC)

        return state

    async def _run_critique(self, state: OrchestratorState) -> CritiqueResult:
        report = state.report
        # Truncate report for critique to avoid context overflow
        report_text = report.markdown_content[:8000] if report.markdown_content else str(report)

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM),
            HumanMessage(content=f"""
Research Query: {state.objective.raw_query}
Report Iteration: {report.iteration}
Reported Confidence: {report.confidence_score:.1%}

REPORT TO EVALUATE:
{report_text}

Evaluate rigorously and return JSON critique.
"""),
        ]

        response = await self.llm.ainvoke(messages)
        usage = TokenUsage(
            prompt_tokens=response.usage_metadata.get("input_tokens", 0),
            completion_tokens=response.usage_metadata.get("output_tokens", 0),
            model=settings.premium_model,
            cost_usd=0.003,
        )
        await self.governance.record_token_usage(AgentType.CRITIC, usage)

        data = json.loads(response.content)
        return CritiqueResult(
            report_id=report.report_id,
            factuality_score=data.get("factuality_score", 0.5),
            coherence_score=data.get("coherence_score", 0.5),
            depth_score=data.get("depth_score", 0.5),
            rigor_score=data.get("rigor_score", 0.5),
            source_quality_score=data.get("source_quality_score", 0.5),
            reasoning_score=data.get("reasoning_score", 0.5),
            citation_accuracy_score=data.get("citation_accuracy_score", 0.5),
            issues_found=data.get("issues_found", []),
            improvement_suggestions=data.get("improvement_suggestions", []),
            hallucinations_detected=data.get("hallucinations_detected", []),
            needs_revision=data.get("needs_revision", False),
        )
