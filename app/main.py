from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.routers.api import router as api_router
from app.routers.web import router as web_router

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)
app = FastAPI(title=settings.app_name, debug=settings.debug)

app.include_router(web_router)
app.include_router(api_router)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def log_startup() -> None:
    logger.info(
        "App started | name=%s | mode=%s | port=%s | upload_dir=%s | max_upload_mb=%s | max_video_seconds=%s | task_device=%s | task_num_workers=%s",
        settings.app_name,
        settings.predictor_mode,
        os.getenv("PORT", "8122"),
        settings.uploads_dir,
        settings.max_upload_mb,
        settings.max_video_seconds,
        settings.task_device,
        settings.task_num_workers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8122")), reload=True)
