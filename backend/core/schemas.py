"""
OMEGA RESEARCH GRID — Canonical Data Schemas
Pydantic v2 models for all inter-agent messages, state objects, and artifacts.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentType(str, Enum):
    PLANNER = "planner"
    SEARCH = "search"
    READER = "reader"
    MEMORY = "memory"
    WRITER = "writer"
    CRITIC = "critic"
    VERIFIER = "verifier"
    COST_OPTIMIZER = "cost_optimizer"
    ORCHESTRATOR = "orchestrator"


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    SEARCHING = "SEARCHING"
    READING = "READING"
    RETRIEVING = "RETRIEVING"
    WRITING = "WRITING"
    CRITIQUING = "CRITIQUING"
    VERIFYING = "VERIFYING"
    REVISING = "REVISING"
    FINALIZING = "FINALIZING"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    ESCALATED = "ESCALATED"
    COMPLETED = "COMPLETED"


class MessageType(str, Enum):
    TASK_ASSIGN = "task_assign"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    STATUS_UPDATE = "status_update"
    ESCALATION = "escalation"
    HEARTBEAT = "heartbeat"


class SourceType(str, Enum):
    WEB = "web"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PDF = "pdf"


class OmegaBase(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_config = {"use_enum_values": True}


class AgentMessage(OmegaBase):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    sender: AgentType
    recipient: str  # AgentType or "broadcast"
    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    retry_count: int = Field(default=0, ge=0)
    ttl_seconds: int = Field(default=300, ge=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    parent_id: Optional[str] = None


class ResearchObjective(OmegaBase):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_query: str = Field(..., min_length=10)
    normalized_query: Optional[str] = None
    domain: Optional[str] = None
    depth: int = Field(default=3, ge=1, le=5)
    max_sources: int = Field(default=20, ge=5, le=100)
    language: str = "en"
    token_budget: int = Field(default=100_000)
    cost_budget_usd: float = Field(default=5.00)


class TaskNode(OmegaBase):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    agent_type: AgentType
    task_name: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    state: ExecutionState = ExecutionState.IDLE
    priority: int = Field(default=5, ge=1, le=10)
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=120)
    retry_count: int = 0
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExecutionDAG(OmegaBase):
    session_id: str
    objective: ResearchObjective
    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    entry_points: list[str] = Field(default_factory=list)
    state: ExecutionState = ExecutionState.IDLE
    total_estimated_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.task_id] = node
        if not node.dependencies:
            self.entry_points.append(node.task_id)

    def get_ready_nodes(self) -> list[TaskNode]:
        completed = {
            tid for tid, t in self.nodes.items()
            if t.state == ExecutionState.COMPLETED
        }
        return [
            t for t in self.nodes.values()
            if t.state == ExecutionState.IDLE
            and set(t.dependencies).issubset(completed)
        ]


class Source(OmegaBase):
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    title: str
    snippet: Optional[str] = None
    source_type: SourceType = SourceType.WEB
    domain: Optional[str] = None
    published_at: Optional[datetime] = None
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def composite_score(self) -> float:
        return (
            0.35 * self.relevance_score
            + 0.30 * self.authority_score
            + 0.20 * self.freshness_score
            + 0.15 * self.trust_score
        )


class SearchResult(OmegaBase):
    query: str
    expanded_queries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    total_found: int = 0
    search_providers_used: list[str] = Field(default_factory=list)


class Claim(OmegaBase):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_id: str
    source_url: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified: bool = False
    contradicts: list[str] = Field(default_factory=list)


class KnowledgeChunk(OmegaBase):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    content: str
    summary: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    token_count: int = 0
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchReport(OmegaBase):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    title: str
    executive_summary: str = ""
    abstract: str = ""
    introduction: str = ""
    methodology: str = ""
    core_analysis: str = ""
    technical_breakdown: str = ""
    key_findings: list[str] = Field(default_factory=list)
    counterarguments: str = ""
    risks: str = ""
    future_outlook: str = ""
    conclusion: str = ""
    references: list[Source] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    critique_scores: dict[str, float] = Field(default_factory=dict)
    iteration: int = 1
    markdown_content: str = ""


class CritiqueResult(OmegaBase):
    report_id: str
    factuality_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    depth_score: float = Field(ge=0.0, le=1.0)
    rigor_score: float = Field(ge=0.0, le=1.0)
    source_quality_score: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=0.0, le=1.0)
    citation_accuracy_score: float = Field(ge=0.0, le=1.0)
    issues_found: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    hallucinations_detected: list[str] = Field(default_factory=list)
    needs_revision: bool = False

    @property
    def overall_score(self) -> float:
        return (
            0.25 * self.factuality_score
            + 0.15 * self.coherence_score
            + 0.15 * self.depth_score
            + 0.15 * self.rigor_score
            + 0.15 * self.source_quality_score
            + 0.10 * self.reasoning_score
            + 0.05 * self.citation_accuracy_score
        )


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0


class CostSnapshot(OmegaBase):
    session_id: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    agent_breakdown: dict[str, float] = Field(default_factory=dict)
    budget_remaining_usd: float = 5.00
    budget_exhausted: bool = False


class OrchestratorState(OmegaBase):
    session_id: str
    objective: Optional[ResearchObjective] = None
    dag: Optional[ExecutionDAG] = None
    current_state: ExecutionState = ExecutionState.IDLE
    search_results: list[SearchResult] = Field(default_factory=list)
    knowledge_chunks: list[KnowledgeChunk] = Field(default_factory=list)
    report: Optional[ResearchReport] = None
    critique: Optional[CritiqueResult] = None
    cost_snapshot: Optional[CostSnapshot] = None
    iteration: int = 0
    max_iterations: int = 3
    errors: list[str] = Field(default_factory=list)
    agent_messages: list[AgentMessage] = Field(default_factory=list)
    memory_context: list[KnowledgeChunk] = Field(default_factory=list)
