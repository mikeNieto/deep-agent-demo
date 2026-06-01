from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.dependencies import get_stt_service, get_tts_service
from app.schemas.common import HealthResponse
from app.services.stt_service import STTService
from app.services.tts_service import TTSService


router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/models")
async def model_health(
    request: Request,
    stt_service: STTService = Depends(get_stt_service),
    tts_service: TTSService = Depends(get_tts_service),
) -> dict[str, str]:
    agent_graph = getattr(request.app.state, "agent_graph", None)
    return {
        "agent": "ready" if agent_graph is not None else "not_configured",
        "stt": "loaded" if stt_service.loaded else "not_loaded",
        "tts": "loaded" if tts_service.loaded else "not_loaded",
    }
