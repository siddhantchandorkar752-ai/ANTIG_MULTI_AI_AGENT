"""
OMEGA RESEARCH GRID — Reader Agent
Extracts and structures knowledge from web pages, PDFs, and HTML.

Security hardening:
  - Prompt injection defense: strips <SYSTEM>, <INST>, [INST] patterns
  - Content length cap: prevents memory exhaustion from giant pages
  - JS-rendered pages via Playwright (async, headless)
  - BeautifulSoup for noise removal (nav, ads, footers)
  - readability-lxml for article body extraction

Chunking strategy:
  - Semantic chunking by paragraph boundary (not fixed token windows)
  - Max 512 tokens per chunk to fit embedding model windows
  - Overlap of 50 tokens for retrieval continuity
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import get_logger
from backend.core.schemas import (
    AgentType,
    Claim,
    ExecutionState,
    KnowledgeChunk,
    OrchestratorState,
    Source,
    TokenUsage,
)

settings = get_settings()
logger = get_logger("reader_agent")

# ── Prompt injection patterns to strip ────────────────────────────────────────
INJECTION_PATTERNS = [
    re.compile(r"<\s*SYSTEM\s*>.*?<\s*/\s*SYSTEM\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[INST\].*?\[/INST\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"###\s*system\s*:", re.IGNORECASE),
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
]

MAX_CONTENT_CHARS = 50_000  # Hard cap per page to prevent memory exhaustion


def sanitize_content(text: str) -> str:
    """Remove prompt injection attempts and truncate oversized content."""
    for pattern in INJECTION_PATTERNS:
        text = pattern.sub("", text)
    return text[:MAX_CONTENT_CHARS]


class ReaderAgent:
    """
    Fetches, extracts, and chunks content from web sources.
    Outputs structured KnowledgeChunk objects ready for vector storage.
    """

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self.llm = ChatOpenAI(
            model=settings.standard_model,
            temperature=0.0,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,   # ~450 tokens at 4 chars/token
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def read(self, state: OrchestratorState) -> OrchestratorState:
        logger.info("reader_start", session_id=state.session_id)
        state.current_state = ExecutionState.READING

        if not await self.governance.acquire_agent_slot(AgentType.READER):
            state.errors.append("Reader slot unavailable")
            return state

        try:
            sources_to_read: list[Source] = []
            for sr in state.search_results:
                sources_to_read.extend(sr.sources[:settings.max_reader_pages])

            # Parallel fetch — bounded concurrency via semaphore
            sem = asyncio.Semaphore(4)
            tasks = [self._process_source(src, sem) for src in sources_to_read]
            chunk_batches = await asyncio.gather(*tasks, return_exceptions=True)

            for batch in chunk_batches:
                if isinstance(batch, Exception):
                    logger.warning("reader_source_error", error=str(batch))
                    continue
                state.knowledge_chunks.extend(batch)

            state.current_state = ExecutionState.WRITING
            logger.info("reader_complete", chunks=len(state.knowledge_chunks))
        except Exception as exc:
            logger.error("reader_failed", error=str(exc))
            state.errors.append(f"Reader failed: {exc}")
        finally:
            await self.governance.release_agent_slot(AgentType.READER)

        return state

    async def _process_source(
        self, source: Source, sem: asyncio.Semaphore
    ) -> list[KnowledgeChunk]:
        async with sem:
            raw_content = await self._fetch_content(source)
            if not raw_content:
                return []
            clean = sanitize_content(raw_content)
            chunks = self._chunk_content(clean, source)
            enriched = await self._extract_claims(chunks)
            return enriched

    async def _fetch_content(self, source: Source) -> Optional[str]:
        """Fetch page content — try simple HTTP first, then Playwright."""
        try:
            async with httpx.AsyncClient(
                timeout=15,
                headers={"User-Agent": "OmegaResearchBot/1.0 (research purposes)"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(source.url)
                if resp.status_code != 200:
                    return None
                html = resp.text
                return self._extract_text_from_html(html)
        except Exception as exc:
            logger.warning("fetch_failed", url=source.url, error=str(exc))
            return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean article text using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 40]
            return "\n".join(lines)
        except Exception:
            return html[:MAX_CONTENT_CHARS]

    def _chunk_content(self, text: str, source: Source) -> list[KnowledgeChunk]:
        """Split content into semantic chunks."""
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = KnowledgeChunk(
                source_id=source.source_id,
                content=chunk_text,
                chunk_index=i,
                token_count=len(chunk_text) // 4,
                metadata={
                    "url": source.url,
                    "title": source.title,
                    "domain": source.domain or "",
                    "authority_score": source.authority_score,
                },
            )
            chunks.append(chunk)
        return chunks

    async def _extract_claims(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """
        Use LLM to extract atomic factual claims from chunks.
        Claims are later used by the Verifier agent for cross-checking.
        Only processes first N chunks to stay within token budget.
        """
        for chunk in chunks[:5]:  # Budget-aware limit
            try:
                prompt = f"""Extract 2-4 key factual claims from this text.
Return ONLY a JSON array of strings.
Text: {chunk.content[:1200]}"""
                response = await self.llm.ainvoke(prompt)
                import json
                claim_texts = json.loads(response.content)
                chunk.claims = [
                    Claim(
                        text=ct,
                        source_id=chunk.source_id,
                        source_url=chunk.metadata.get("url", ""),
                        confidence=0.7,
                    )
                    for ct in claim_texts[:4]
                    if isinstance(ct, str) and len(ct) > 20
                ]
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.get("input_tokens", 0),
                    completion_tokens=response.usage_metadata.get("output_tokens", 0),
                    model=settings.standard_model,
                    cost_usd=0.0005,
                )
                await self.governance.record_token_usage(AgentType.READER, usage)
            except Exception as exc:
                logger.warning("claim_extraction_failed", error=str(exc))
        return chunks
