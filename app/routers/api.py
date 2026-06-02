from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.prediction import PredictionResponse
from app.services.predictor import get_predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_virality(video: UploadFile = File(...)) -> PredictionResponse:
    settings = get_settings()
    predictor = get_predictor()
    started_at = perf_counter()

    filename = video.filename or ""
    suffix = Path(filename).suffix.lower()
    logger.info("Prediction request received | filename=%s | suffix=%s", filename, suffix or "<none>")
    if suffix not in settings.allowed_upload_suffixes:
        allowed = ", ".join(settings.allowed_upload_suffixes)
        logger.warning("Rejected upload format | filename=%s | suffix=%s", filename, suffix or "<none>")
        raise HTTPException(status_code=400, detail=f"Format tidak didukung. Gunakan: {allowed}")

    payload = await video.read()
    if not payload:
        logger.warning("Rejected empty upload | filename=%s", filename)
        raise HTTPException(status_code=400, detail="File video kosong.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        logger.warning(
            "Rejected oversized upload | filename=%s | size_bytes=%s | max_bytes=%s",
            filename,
            len(payload),
            max_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran file melebihi {settings.max_upload_mb} MB.",
        )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = settings.uploads_dir / f"{uuid4().hex}{suffix}"
    temp_path.write_bytes(payload)
    logger.info("Temporary upload stored | temp_path=%s | size_bytes=%s", temp_path, len(payload))

    try:
        result = predictor.predict(temp_path)
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Prediction finished | engine=%s | score=%.2f | elapsed_ms=%.1f | file=%s",
            result.engine,
            result.virality_score,
            elapsed_ms,
            filename,
        )
        return result
    except ValueError as exc:
        logger.warning("Prediction validation error | file=%s | error=%s", filename, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Prediction runtime error | file=%s | error=%s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction | file=%s", filename)
        raise HTTPException(status_code=500, detail="Prediksi gagal diproses.") from exc
    finally:
        await video.close()
        if not settings.keep_uploads and temp_path.exists():
            temp_path.unlink(missing_ok=True)
            logger.info("Temporary upload deleted | temp_path=%s", temp_path)


@router.get("/health")
async def health() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}
