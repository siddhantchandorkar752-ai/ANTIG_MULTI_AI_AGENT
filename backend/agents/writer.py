"""
OMEGA RESEARCH GRID — Writer Agent
Synthesizes research findings into professional structured reports.

Architecture:
  - LCEL chain: context_assembly | prompt | LLM | output_parser
  - Section-by-section generation to avoid context window overflows
  - Source citation injection via reference mapper
  - Confidence annotation based on source authority scores
  - Streaming output via async generator for real-time frontend updates

Report generation strategy:
  1. Compress and rank relevant context chunks
  2. Generate executive summary first (establishes coherence)
  3. Generate each section sequentially (maintains narrative flow)
  4. Inject citations and confidence scores post-generation
  5. Render final markdown + export PDF via WeasyPrint
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    ExecutionState,
    KnowledgeChunk,
    OrchestratorState,
    ResearchReport,
    Source,
    TokenUsage,
)

settings = get_settings()
logger = get_logger("writer_agent")

WRITER_SYSTEM = """You are the Writer Agent of the OMEGA RESEARCH GRID.
You synthesize research findings into professional, evidence-based reports.

Guidelines:
- Ground every claim in the provided source material
- Use precise, academic language
- Acknowledge uncertainty with hedging language where appropriate
- Structure content with clear headers and logical flow
- Include specific statistics, numbers, and quotes where available
- Flag speculative content explicitly with [INFERRED] marker
- Cite sources inline as [Source: domain.com]
"""

SECTION_PROMPTS = {
    "executive_summary": "Write a 3-paragraph executive summary of the research. Be concise and actionable.",
    "abstract": "Write a formal academic abstract (250 words) summarizing methodology and findings.",
    "introduction": "Write an introduction explaining the research question, its importance, and scope.",
    "methodology": "Describe the research methodology: sources consulted, analysis approach, limitations.",
    "core_analysis": "Write a comprehensive analysis of the main findings. Be thorough and analytical.",
    "technical_breakdown": "Provide a technical deep-dive into the key mechanisms, systems, or frameworks involved.",
    "key_findings": "List the 5-8 most important findings as clear, evidence-backed statements.",
    "counterarguments": "Present the strongest counterarguments and alternative perspectives objectively.",
    "risks": "Analyze key risks, limitations, and failure modes relevant to this topic.",
    "future_outlook": "Describe likely future developments, emerging trends, and open questions.",
    "conclusion": "Synthesize the analysis into a definitive conclusion with actionable insights.",
}


class WriterAgent:
    """Synthesizes research into structured professional reports."""

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self.premium_llm = ChatOpenAI(
            model=settings.premium_model,
            temperature=0.4,
            streaming=True,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        self.standard_llm = ChatOpenAI(
            model=settings.standard_model,
            temperature=0.4,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    async def write(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("writer_start", session_id=state.session_id, iteration=state.iteration)
        state.current_state = ExecutionState.WRITING

        if not await self.governance.check_budget():
            state.errors.append("Budget exhausted before writing")
            state.current_state = ExecutionState.FAILED
            return state

        if not await self.governance.acquire_agent_slot(AgentType.WRITER):
            return state

        try:
            # Select model based on remaining budget
            model = await self.governance.select_model(complexity=0.85)
            llm = self.premium_llm if model == settings.premium_model else self.standard_llm

            # Compress context
            from backend.agents.memory import MemoryAgent
            memory = MemoryAgent(self.governance)
            context_chunks = await memory.compress_context(
                state.knowledge_chunks,
                max_tokens=12_000,
            )

            context_text = self._build_context(context_chunks)
            objective = state.objective

            report = ResearchReport(
                session_id=state.session_id,
                title=f"Research Report: {objective.normalized_query or objective.raw_query}",
                iteration=state.iteration + 1,
                references=self._collect_references(state),
            )

            # Generate sections (async, sequential for coherence)
            if state.critique:
                improvement_hints = "\n".join(state.critique.improvement_suggestions[:3])
            else:
                improvement_hints = "First generation — focus on completeness and accuracy."

            base_context = f"""
Research Query: {objective.normalized_query or objective.raw_query}
Domain: {objective.domain or 'general'}
Previous Critique: {improvement_hints}

SOURCE MATERIAL:
{context_text}
"""

            for section_name, instruction in SECTION_PROMPTS.items():
                section_content = await self._generate_section(
                    llm=llm,
                    section=section_name,
                    instruction=instruction,
                    context=base_context,
                )
                if section_name == "key_findings":
                    report.key_findings = [
                        line.strip().lstrip("•-*").strip()
                        for line in section_content.split("\n")
                        if len(line.strip()) > 20
                    ][:8]
                else:
                    setattr(report, section_name, section_content)

            report.confidence_score = self._compute_confidence(context_chunks)
            report.markdown_content = self._render_markdown(report)
            state.report = report
            state.current_state = ExecutionState.CRITIQUING
            logger.info("writer_complete", sections=len(SECTION_PROMPTS))
        except Exception as exc:
            logger.error("writer_failed", error=str(exc))
            state.errors.append(f"Writer failed: {exc}")
            state.current_state = ExecutionState.FAILED
        finally:
            await self.governance.release_agent_slot(AgentType.WRITER)

        return state

    async def _generate_section(
        self, llm: ChatOpenAI, section: str, instruction: str, context: str
    ) -> str:
        """Generate a single report section via LCEL chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM),
            ("human", f"{context}\n\n---\nNow generate the {section.replace('_', ' ').title()} section.\n{instruction}"),
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            result = await chain.ainvoke({})
            return result
        except Exception as exc:
            logger.warning("section_generation_failed", section=section, error=str(exc))
            return f"[Section generation failed for {section}]"

    def _build_context(self, chunks: list[KnowledgeChunk]) -> str:
        sections = []
        for chunk in chunks:
            domain = chunk.metadata.get("domain", "unknown")
            authority = chunk.metadata.get("authority_score", 0.5)
            sections.append(
                f"[Source: {domain} | Authority: {authority:.1f}]\n{chunk.content}\n"
            )
        return "\n---\n".join(sections[:20])

    def _collect_references(self, state: OrchestratorState) -> list[Source]:
        refs: list[Source] = []
        seen: set[str] = set()
        for sr in state.search_results:
            for src in sr.sources:
                if src.source_id not in seen:
                    seen.add(src.source_id)
                    refs.append(src)
        return refs[:30]

    def _compute_confidence(self, chunks: list[KnowledgeChunk]) -> float:
        if not chunks:
            return 0.3
        avg_authority = sum(
            float(c.metadata.get("authority_score", 0.5)) for c in chunks
        ) / len(chunks)
        diversity = min(1.0, len({c.source_id for c in chunks}) / 5)
        return round(0.6 * avg_authority + 0.4 * diversity, 3)

    def _render_markdown(self, report: ResearchReport) -> str:
        md = f"# {report.title}\n\n"
        md += f"*Confidence: {report.confidence_score:.1%} | Iteration {report.iteration}*\n\n"
        md += "---\n\n"
        md += f"## Executive Summary\n{report.executive_summary}\n\n"
        md += f"## Abstract\n{report.abstract}\n\n"
        md += f"## Introduction\n{report.introduction}\n\n"
        md += f"## Research Methodology\n{report.methodology}\n\n"
        md += f"## Core Analysis\n{report.core_analysis}\n\n"
        md += f"## Technical Breakdown\n{report.technical_breakdown}\n\n"
        md += "## Key Findings\n"
        for i, f in enumerate(report.key_findings, 1):
            md += f"{i}. {f}\n"
        md += "\n"
        md += f"## Counterarguments\n{report.counterarguments}\n\n"
        md += f"## Risks & Limitations\n{report.risks}\n\n"
        md += f"## Future Outlook\n{report.future_outlook}\n\n"
        md += f"## Conclusion\n{report.conclusion}\n\n"
        md += "## References\n"
        for i, ref in enumerate(report.references[:20], 1):
            pub = ref.published_at.strftime("%Y-%m-%d") if ref.published_at else "n.d."
            md += f"{i}. [{ref.title}]({ref.url}) — {ref.domain or 'web'} ({pub})\n"
        return md
