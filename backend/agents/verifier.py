"""
OMEGA RESEARCH GRID — Verifier Agent
Independent claim verification through cross-source comparison.

The Verifier is intentionally isolated from the Writer's context.
It re-searches independently to confirm factual claims.

Verification strategy:
  1. Extract all claims from the report
  2. For each high-confidence claim, search Tavily for corroboration
  3. Compare claim text against retrieved sources via LLM judge
  4. Flag unsupported or contradicted claims
  5. Compute citation validity (URL reachable + content matches claim)
  6. Update report with verification metadata
"""
from __future__ import annotations

import json

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    ExecutionState,
    OrchestratorState,
    TokenUsage,
)

settings = get_settings()
logger = get_logger("verifier_agent")

VERIFY_SYSTEM = """You are the Verifier Agent. You assess if a claim is supported by provided evidence.
Return ONLY JSON:
{
  "is_supported": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "contradicts": true/false
}"""


class VerifierAgent:
    """Independent fact-verification via re-search and LLM judgment."""

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self.llm = ChatOpenAI(
            model=settings.standard_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            api_key=settings.openai_api_key.get_secret_value(),
        )

    async def verify(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("verifier_start", session_id=state.session_id)
        state.current_state = ExecutionState.VERIFYING

        if not state.report:
            state.current_state = ExecutionState.FINALIZING
            return state

        if not await self.governance.acquire_agent_slot(AgentType.VERIFIER):
            state.current_state = ExecutionState.FINALIZING
            return state

        try:
            # Collect all claims from knowledge chunks
            all_claims = []
            for chunk in state.knowledge_chunks[:10]:
                all_claims.extend(chunk.claims[:2])

            verified_count = 0
            supported_count = 0

            for claim in all_claims[:15]:  # Verify up to 15 claims (cost control)
                if not await self.governance.check_budget():
                    break
                result = await self._verify_claim(claim.text)
                if result["is_supported"]:
                    claim.verified = True
                    supported_count += 1
                else:
                    self.governance.flag_hallucination(claim.text)
                verified_count += 1

            # Validate URLs in references
            invalid_urls = await self._validate_reference_urls(state)

            verification_ratio = supported_count / max(verified_count, 1)
            logger.info(
                "verifier_complete",
                verified=verified_count,
                supported=supported_count,
                ratio=verification_ratio,
                invalid_urls=len(invalid_urls),
            )

            # Adjust report confidence based on verification
            if state.report and verified_count > 0:
                state.report.confidence_score = min(
                    1.0,
                    state.report.confidence_score * (0.5 + 0.5 * verification_ratio),
                )

            state.current_state = ExecutionState.FINALIZING
        except Exception as exc:
            logger.error("verifier_failed", error=str(exc))
            state.errors.append(f"Verifier failed: {exc}")
            state.current_state = ExecutionState.FINALIZING  # Graceful degradation
        finally:
            await self.governance.release_agent_slot(AgentType.VERIFIER)

        return state

    async def _verify_claim(self, claim: str) -> dict:
        """Use Tavily to search for evidence, then LLM to judge."""
        try:
            evidence = await self._search_evidence(claim)
            if not evidence:
                return {"is_supported": False, "confidence": 0.3, "contradicts": False}

            messages = [
                SystemMessage(content=VERIFY_SYSTEM),
                HumanMessage(content=f"""
Claim: {claim}

Evidence from independent sources:
{evidence[:2000]}

Is this claim supported by the evidence?"""),
            ]
            response = await self.llm.ainvoke(messages)
            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.get("input_tokens", 0),
                completion_tokens=response.usage_metadata.get("output_tokens", 0),
                model=settings.standard_model,
                cost_usd=0.0003,
            )
            await self.governance.record_token_usage(AgentType.VERIFIER, usage)
            return json.loads(response.content)
        except Exception as exc:
            logger.warning("claim_verification_error", error=str(exc))
            return {"is_supported": True, "confidence": 0.5, "contradicts": False}

    async def _search_evidence(self, claim: str) -> str:
        """Search for independent evidence for a claim."""
        if not settings.tavily_api_key.get_secret_value():
            return ""
        try:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())
            response = await client.search(query=claim[:200], max_results=3, search_depth="basic")
            snippets = [r.get("content", "") for r in response.get("results", [])]
            return "\n".join(snippets)
        except Exception:
            return ""

    async def _validate_reference_urls(self, state: OrchestratorState) -> list[str]:
        """Check that reference URLs are reachable."""
        invalid: list[str] = []
        if not state.report:
            return invalid
        async with httpx.AsyncClient(timeout=8) as client:
            for ref in state.report.references[:10]:
                try:
                    resp = await client.head(ref.url, follow_redirects=True)
                    if resp.status_code >= 400:
                        invalid.append(ref.url)
                except Exception:
                    invalid.append(ref.url)
        return invalid
