from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _normalize_task_device(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "cpu": "cpu",
        "gpu": "gpu",
        "cuda": "gpu",
    }
    return aliases.get(raw, "auto")


def _default_num_workers() -> int:
    cpus = os.cpu_count() or 4
    # keep some headroom so default does not exceed system recommendation too easily
    return max(1, min(16, cpus - 2))


@dataclass(frozen=True)
class Settings:
    app_name: str
    debug: bool
    predictor_mode: str
    tribe_repo_id: str
    tribe_checkpoint_name: str
    tribe_cache_dir: Path
    tribe_device: str
    task_device: str
    task_num_workers: int
    max_upload_mb: int
    max_video_seconds: int
    uploads_dir: Path
    keep_uploads: bool
    allowed_upload_suffixes: tuple[str, ...]
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    allowed_suffixes = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    raw_repo_id = os.getenv("TRIBE_REPO_ID", "facebook/tribev2").strip()
    normalized_repo_id = raw_repo_id.replace("\\", "/")
    task_device = _normalize_task_device(os.getenv("TASK_DEVICE", os.getenv("TRIBE_DEVICE", "auto")))
    return Settings(
        app_name=os.getenv("APP_NAME", "TRIBE v2 Virality Predictor"),
        debug=_env_bool("DEBUG", False),
        predictor_mode=os.getenv("PREDICTOR_MODE", "auto").strip().lower(),
        tribe_repo_id=normalized_repo_id,
        tribe_checkpoint_name=os.getenv("TRIBE_CHECKPOINT_NAME", "best.ckpt"),
        tribe_cache_dir=Path(os.getenv("TRIBE_CACHE_DIR", str(root / ".cache" / "tribe"))),
        tribe_device=os.getenv("TRIBE_DEVICE", "auto").strip().lower(),
        task_device=task_device,
        task_num_workers=_env_int("TASK_NUM_WORKERS", _default_num_workers()),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 120),
        max_video_seconds=_env_int("MAX_VIDEO_SECONDS", 15),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(root / "data" / "uploads"))),
        keep_uploads=_env_bool("KEEP_UPLOADS", False),
        allowed_upload_suffixes=allowed_suffixes,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )
