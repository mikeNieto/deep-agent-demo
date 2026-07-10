from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.live import router as live_router
from app.config import get_settings

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
INDEX_PATH = STATIC_DIR / "index.html"

app = FastAPI(title="Gemini Live Test")
app.include_router(live_router)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_PATH.read_text()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        factory=False,
    )
