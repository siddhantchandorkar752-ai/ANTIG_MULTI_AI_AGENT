"""
OMEGA RESEARCH GRID — Core Configuration
Central settings management with Pydantic v2 Settings.
All config is environment-variable driven for 12-factor compliance.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ───────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: SecretStr = Field(default="insecure-dev-key-change-in-prod")

    # ── AI Providers ──────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(default="")
    anthropic_api_key: SecretStr = Field(default="")
    google_api_key: SecretStr = Field(default="")

    # ── Model Selection ───────────────────────────────────────────────────────
    premium_model: str = "gpt-4o"
    standard_model: str = "gpt-4o-mini"
    fallback_model: str = "claude-3-haiku-20240307"

    # ── Search ────────────────────────────────────────────────────────────────
    tavily_api_key: SecretStr = Field(default="")
    serpapi_key: SecretStr = Field(default="")
    firecrawl_api_key: SecretStr = Field(default="")

    # ── Observability ─────────────────────────────────────────────────────────
    langchain_api_key: SecretStr = Field(default="")
    langchain_tracing_v2: bool = True
    langchain_project: str = "omega-research-grid"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "omega-research-grid"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://omega:omega@localhost:5432/omega_research"
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # ── Execution Limits ──────────────────────────────────────────────────────
    max_search_results: int = 20
    max_reader_pages: int = 10
    max_recursion_depth: int = 5
    max_total_tokens: int = 500_000
    token_budget_per_task: int = 100_000
    max_concurrent_agents: int = 8
    task_timeout_seconds: int = 300

    # ── Cost Thresholds ───────────────────────────────────────────────────────
    cost_threshold_usd: float = 5.00
    cheap_model_threshold_usd: float = 1.00

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings loader — cached after first call."""
    return Settings()
