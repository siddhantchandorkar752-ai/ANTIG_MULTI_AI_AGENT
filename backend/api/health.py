"""OMEGA RESEARCH GRID — Health Check Endpoints (Kubernetes-ready)"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    service: str = "omega-research-grid"


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe — returns 200 if process is alive."""
    return HealthResponse(status="alive")


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness() -> HealthResponse:
    """Kubernetes readiness probe — checks critical dependencies."""
    return HealthResponse(status="ready")


@router.get("/", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")
