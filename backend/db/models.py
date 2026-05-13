"""
OMEGA RESEARCH GRID — Database Models (SQLAlchemy 2.0)
Relational schema for session persistence, audit logging, and report storage.

Schema design rationale:
  - research_sessions: lifecycle tracking, one per user query
  - research_reports: versioned report store (supports iteration history)
  - audit_logs: immutable governance audit trail (append-only)
  - agent_metrics: per-agent performance telemetry
  - sources: deduplicated source registry across sessions
  - claims: atomic fact claims for verification tracking

Scaling strategy:
  - Partition audit_logs by created_at (monthly partitions)
  - Index research_sessions.user_id + created_at for dashboard queries
  - JSONB columns for flexible payload storage (PostgreSQL-specific)
  - created_at indexed everywhere — all queries are time-bounded
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(50), default="IDLE", index=True)
    depth: Mapped[int] = mapped_column(Integer, default=3)
    token_budget: Mapped[int] = mapped_column(Integer, default=100_000)
    cost_budget_usd: Mapped[float] = mapped_column(Float, default=5.0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    final_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    reports: Mapped[list["ResearchReport"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    agent_metrics: Mapped[list["AgentMetric"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, default=1)
    executive_summary: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    introduction: Mapped[str] = mapped_column(Text, default="")
    methodology: Mapped[str] = mapped_column(Text, default="")
    core_analysis: Mapped[str] = mapped_column(Text, default="")
    technical_breakdown: Mapped[str] = mapped_column(Text, default="")
    key_findings: Mapped[dict] = mapped_column(JSONB, default=list)
    counterarguments: Mapped[str] = mapped_column(Text, default="")
    risks: Mapped[str] = mapped_column(Text, default="")
    future_outlook: Mapped[str] = mapped_column(Text, default="")
    conclusion: Mapped[str] = mapped_column(Text, default="")
    references: Mapped[dict] = mapped_column(JSONB, default=list)
    critique_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    markdown_content: Mapped[str] = mapped_column(Text, default="")
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ResearchSession"] = relationship(back_populates="reports")


class AuditLog(Base):
    """
    Immutable governance audit trail.
    Every agent decision, tool call, and state transition is recorded here.
    NEVER update or delete rows — this is an append-only log.
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    agent_type: Mapped[str | None] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    session: Mapped["ResearchSession"] = relationship(back_populates="audit_logs")


class AgentMetric(Base):
    """Per-agent performance metrics for monitoring and optimization."""
    __tablename__ = "agent_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True)
    agent_type: Mapped[str] = mapped_column(String(50), index=True)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    model_used: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ResearchSession"] = relationship(back_populates="agent_metrics")


class SourceRecord(Base):
    """Deduplicated source registry — shared across sessions."""
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[str] = mapped_column(Text, unique=True, index=True)
    url_fingerprint: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str | None] = mapped_column(String(255), index=True)
    source_type: Mapped[str] = mapped_column(String(50), default="web")
    authority_score: Mapped[float] = mapped_column(Float, default=0.5)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    access_count: Mapped[int] = mapped_column(Integer, default=1)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE ENGINE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        from backend.core.config import get_settings
        s = get_settings()
        _engine = create_async_engine(
            s.database_url,
            echo=s.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


async def get_db_session() -> AsyncSession:
    from sqlalchemy.ext.asyncio import async_sessionmaker
    engine = get_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


async def create_tables() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
