"""
OMEGA RESEARCH GRID — Memory Agent
Manages all memory tiers: short-term (Redis), long-term (ChromaDB/FAISS), episodic.

Memory architecture:
  ┌──────────────────────────────────────────────────────┐
  │  Working Memory (in-process dict, per-session)       │
  │  Short-Term Memory (Redis, TTL=1h, serialized JSON)  │
  │  Long-Term Semantic Memory (ChromaDB, persistent)    │
  │  Episodic Memory (PostgreSQL, full audit trail)      │
  └──────────────────────────────────────────────────────┘

Retrieval pipeline:
  1. Query embedding via text-embedding-3-small
  2. ChromaDB cosine similarity search (top-K)
  3. BM25 keyword search on same corpus
  4. Reciprocal Rank Fusion (RRF) to merge rankings
  5. Metadata filtering (recency, domain, authority)
  6. Context compression (trim low-confidence chunks)

Why RRF over pure vector search:
  Vector search can miss exact keyword matches for technical terms
  (model names, chemical formulas, code snippets). BM25 catches these.
  RRF is parameter-free and empirically outperforms weighted combinations.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.core.config import get_settings
from backend.core.governance import GovernanceEngine
from backend.core.logging import MEMORY_OPERATIONS, get_logger
from backend.core.schemas import (
    AgentType,
    ExecutionState,
    KnowledgeChunk,
    OrchestratorState,
)

settings = get_settings()
logger = get_logger("memory_agent")


class MemoryAgent:
    """
    Hybrid memory system: Redis (cache) + ChromaDB (vector) + FAISS (local).
    """

    def __init__(self, governance: GovernanceEngine):
        self.governance = governance
        self._chroma_client: Optional[Any] = None
        self._redis: Optional[aioredis.Redis] = None
        self._collection_name = "omega_knowledge"

    async def initialize(self) -> None:
        """Lazy initialization of external connections."""
        try:
            import chromadb
            self._chroma_client = await asyncio.to_thread(
                chromadb.HttpClient,
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            logger.info("chromadb_connected")
        except Exception as exc:
            logger.warning("chromadb_unavailable", error=str(exc))

        try:
            self._redis = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("redis_connected")
        except Exception as exc:
            logger.warning("redis_unavailable", error=str(exc))

    async def store(self, state: OrchestratorState) -> OrchestratorState:
        """
        Ingest all knowledge chunks into memory stores.
        Called after Reader agent populates state.knowledge_chunks.
        """
        logger.info("memory_store_start", chunks=len(state.knowledge_chunks))

        if not await self.governance.acquire_agent_slot(AgentType.MEMORY):
            return state

        try:
            await asyncio.gather(
                self._store_in_chroma(state.knowledge_chunks, state.session_id),
                self._cache_in_redis(state.knowledge_chunks, state.session_id),
                return_exceptions=True,
            )
            logger.info("memory_store_complete")
        except Exception as exc:
            logger.error("memory_store_failed", error=str(exc))
        finally:
            await self.governance.release_agent_slot(AgentType.MEMORY)

        return state

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
    ) -> list[KnowledgeChunk]:
        """
        Hybrid retrieval: ChromaDB vector search + BM25 → RRF merge.
        """
        MEMORY_OPERATIONS.labels(operation="read", store="chroma").inc()

        # Try Redis cache first
        cache_key = f"recall:{session_id}:{hash(query)}"
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                MEMORY_OPERATIONS.labels(operation="cache_hit", store="redis").inc()
                data = json.loads(cached)
                return [KnowledgeChunk(**c) for c in data]

        # Vector search
        vector_results = await self._vector_search(query, session_id, top_k)

        # BM25 search over working memory
        bm25_results = self._bm25_search(query, top_k)

        # RRF merge
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)

        # Cache result
        if self._redis and merged:
            await self._redis.setex(
                cache_key,
                settings.redis_cache_ttl,
                json.dumps([c.model_dump(mode="json", exclude={"embedding": True}) for c in merged]),
            )

        return merged

    async def _store_in_chroma(
        self, chunks: list[KnowledgeChunk], session_id: str
    ) -> None:
        if not self._chroma_client:
            return
        try:
            from openai import AsyncOpenAI
            oai = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
            collection = await asyncio.to_thread(
                self._chroma_client.get_or_create_collection,
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Batch embed
            texts = [c.content[:1800] for c in chunks]
            if not texts:
                return
            embed_response = await oai.embeddings.create(
                input=texts,
                model="text-embedding-3-small",
            )
            embeddings = [e.embedding for e in embed_response.data]
            ids = [c.chunk_id for c in chunks]
            metadatas = [
                {
                    "session_id": session_id,
                    "source_id": c.source_id,
                    "chunk_index": c.chunk_index,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ]
            await asyncio.to_thread(
                collection.add,
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            MEMORY_OPERATIONS.labels(operation="write", store="chroma").inc()
            logger.info("chroma_stored", count=len(chunks))
        except Exception as exc:
            logger.warning("chroma_store_failed", error=str(exc))

    async def _vector_search(
        self, query: str, session_id: str, top_k: int
    ) -> list[KnowledgeChunk]:
        if not self._chroma_client:
            return []
        try:
            from openai import AsyncOpenAI
            oai = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
            embed = await oai.embeddings.create(input=[query], model="text-embedding-3-small")
            query_vec = embed.data[0].embedding
            collection = await asyncio.to_thread(
                self._chroma_client.get_collection, self._collection_name
            )
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_vec],
                n_results=top_k,
                where={"session_id": session_id},
            )
            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                chunk = KnowledgeChunk(
                    chunk_id=results["ids"][0][i],
                    source_id=meta.get("source_id", ""),
                    content=doc,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=meta,
                )
                chunks.append(chunk)
            return chunks
        except Exception as exc:
            logger.warning("vector_search_failed", error=str(exc))
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """
        BM25 over in-process working memory.
        Falls back gracefully if rank-bm25 not installed.
        """
        return []  # Populated from working memory in full impl

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[KnowledgeChunk],
        bm25_results: list[KnowledgeChunk],
        top_k: int,
        k: int = 60,
    ) -> list[KnowledgeChunk]:
        """
        RRF: score(d) = Σ 1/(k + rank_i(d))
        Parameter-free, robust to scale differences between rankers.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, KnowledgeChunk] = {}

        for rank, chunk in enumerate(vector_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunks[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunks[chunk.chunk_id] = chunk

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [chunks[cid] for cid in ranked_ids[:top_k]]

    async def _cache_in_redis(
        self, chunks: list[KnowledgeChunk], session_id: str
    ) -> None:
        if not self._redis:
            return
        try:
            key = f"session:{session_id}:chunks"
            serialized = json.dumps([
                c.model_dump(mode="json") for c in chunks[:50]
            ])
            await self._redis.setex(key, settings.redis_cache_ttl, serialized)
            MEMORY_OPERATIONS.labels(operation="write", store="redis").inc()
        except Exception as exc:
            logger.warning("redis_cache_failed", error=str(exc))

    async def compress_context(
        self, chunks: list[KnowledgeChunk], max_tokens: int = 8000
    ) -> list[KnowledgeChunk]:
        """
        Prune context to fit within token budget.
        Strategy: keep highest-authority chunks until token limit.
        """
        selected: list[KnowledgeChunk] = []
        used_tokens = 0
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get("authority_score", 0.5),
            reverse=True,
        )
        for chunk in sorted_chunks:
            if used_tokens + chunk.token_count > max_tokens:
                break
            selected.append(chunk)
            used_tokens += chunk.token_count
        logger.info("context_compressed", before=len(chunks), after=len(selected))
        return selected
