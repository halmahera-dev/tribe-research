from __future__ import annotations

import logging
import os
import subprocess
import threading
from functools import lru_cache
from pathlib import Path, PurePosixPath
from time import perf_counter

from app.core.config import Settings, get_settings
from app.schemas.prediction import PredictionResponse
from app.services.scoring import build_heuristic_prediction, build_prediction_from_preds

logger = logging.getLogger(__name__)


class ViralityPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._model_error: Exception | None = None
        self._load_lock = threading.Lock()

    def predict(self, video_path: Path) -> PredictionResponse:
        mode = self.settings.predictor_mode
        if mode not in {"auto", "tribe", "heuristic"}:
            mode = "auto"
        logger.info("Predictor started | mode=%s | file=%s", mode, video_path)

        if mode == "heuristic":
            result = build_heuristic_prediction(video_path, self.settings.max_video_seconds)
            logger.info("Predictor finished with heuristic mode | file=%s | score=%.2f", video_path, result.virality_score)
            return result

        try:
            infer_started = perf_counter()
            model = self._ensure_model()
            events = model.get_events_dataframe(video_path=str(video_path))
            duration = self._extract_duration_seconds(events)
            logger.info("Event extraction complete | file=%s | duration_seconds=%.2f", video_path, duration)
            if duration > self.settings.max_video_seconds:
                raise ValueError(
                    f"Durasi video {duration:.2f}s melebihi batas {self.settings.max_video_seconds}s."
                )
            preds, segments = model.predict(events=events, verbose=False)
            result = build_prediction_from_preds(preds, segments, engine="tribev2")
            result.notes.append("Inference path: TRIBE v2 multimodal brain encoding model.")
            logger.info(
                "TRIBE inference success | file=%s | score=%.2f | elapsed_ms=%.1f",
                video_path,
                result.virality_score,
                (perf_counter() - infer_started) * 1000,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - we intentionally fallback in auto mode
            logger.exception("TRIBE inference failed | file=%s | mode=%s", video_path, mode)
            if mode == "tribe":
                if isinstance(exc, ValueError):
                    raise
                raise RuntimeError(
                    "Mode tribev2 gagal. Cek akses HuggingFace/token, dependency, dan GPU/CPU memory."
                ) from exc
            logger.warning("Switching to heuristic fallback | file=%s | reason=%s", video_path, exc.__class__.__name__)
            fallback = build_heuristic_prediction(video_path, self.settings.max_video_seconds)
            fallback.notes.append(f"Auto fallback reason: {exc.__class__.__name__}.")
            logger.info("Fallback success | file=%s | score=%.2f", video_path, fallback.virality_score)
            return fallback

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._model_error is not None:
            raise self._model_error

        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._model_error is not None:
                raise self._model_error

            try:
                from tribev2 import TribeModel
            except Exception as exc:  # noqa: BLE001
                self._model_error = RuntimeError(
                    "Package tribev2 belum terpasang. Jalankan pip install -r requirements.txt."
                )
                raise self._model_error from exc

            self.settings.tribe_cache_dir.mkdir(parents=True, exist_ok=True)
            runtime_device = self._resolve_runtime_device()
            self._patch_whisperx_cpu_compute_type(runtime_device)
            repo_id = self.settings.tribe_repo_id.strip().strip("\"'").replace("\\", "/")
            logger.info(
                "Loading TRIBE model from repo_id=%s | task_device=%s | resolved_device=%s | workers=%s",
                repo_id,
                self.settings.task_device,
                runtime_device,
                self.settings.task_num_workers,
            )
            checkpoint_dir = self._prepare_checkpoint_dir(repo_id)
            config_update = self._build_runtime_config_update(runtime_device)
            self._model = TribeModel.from_pretrained(
                checkpoint_dir,
                checkpoint_name=self.settings.tribe_checkpoint_name,
                cache_folder=str(self.settings.tribe_cache_dir),
                device=runtime_device,
                config_update=config_update,
            )
            logger.info("TRIBE model loaded successfully.")
            return self._model

    def _resolve_runtime_device(self) -> str:
        requested = self.settings.task_device
        try:
            import torch
        except Exception:
            logger.warning("PyTorch not available for device check; fallback to CPU.")
            return "cpu"

        has_cuda = torch.cuda.is_available()
        if requested == "cpu":
            return "cpu"
        if requested == "gpu":
            if has_cuda:
                return "cuda"
            logger.warning("TASK_DEVICE=gpu tapi CUDA tidak tersedia. Fallback ke CPU.")
            return "cpu"
        return "cuda" if has_cuda else "cpu"

    def _build_runtime_config_update(self, runtime_device: str) -> dict:
        """
        Force runtime-safe config for local inference with device autodetection.
        """
        small_batch = 1 if runtime_device == "cpu" else 4
        return {
            # LLM/text/audio/video/image feature extractors
            "data.text_feature.device": runtime_device,
            "data.audio_feature.device": runtime_device,
            "data.image_feature.image.device": runtime_device,
            "data.video_feature.image.device": runtime_device,
            # Reduce pressure when running on CPU, still usable on GPU.
            "data.text_feature.batch_size": small_batch,
            "data.image_feature.image.batch_size": small_batch,
            "data.video_feature.image.batch_size": small_batch,
            # Avoid excessive worker warning / freeze risk.
            "data.num_workers": self.settings.task_num_workers,
        }

    @staticmethod
    def _patch_whisperx_cpu_compute_type(runtime_device: str) -> None:
        """
        Monkeypatch tribev2's whisperx runner so CPU does not use float16.

        Upstream code sets:
          device = "cuda" if torch.cuda.is_available() else "cpu"
          compute_type = "float16"
        This crashes on CPU with faster-whisper:
          "Requested float16 compute type ... not supported"
        """
        try:
            from tribev2.eventstransforms import ExtractWordsFromAudio
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip whisperx patch (imports unavailable): %s", exc)
            return

        if getattr(ExtractWordsFromAudio, "_patched_for_cpu_compute_type", False):
            return

        original = ExtractWordsFromAudio._get_transcript_from_audio

        def _patched_get_transcript_from_audio(wav_filename: Path, language: str):
            import json
            import tempfile

            language_codes = dict(
                english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
            )
            if language not in language_codes:
                raise ValueError(f"Language {language} not supported")

            device = runtime_device
            compute_type = "float16" if device == "cuda" else "int8"
            batch_size = "16" if device == "cuda" else "4"

            with tempfile.TemporaryDirectory() as output_dir:
                logger.info(
                    "Running whisperx via uvx (patched) | device=%s | compute_type=%s | batch_size=%s",
                    device,
                    compute_type,
                    batch_size,
                )
                cmd = [
                    "uvx",
                    "whisperx",
                    str(wav_filename),
                    "--model",
                    "large-v3",
                    "--language",
                    language_codes[language],
                    "--device",
                    device,
                    "--compute_type",
                    compute_type,
                    "--batch_size",
                    batch_size,
                    "--align_model",
                    "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                    "--output_dir",
                    output_dir,
                    "--output_format",
                    "json",
                ]
                cmd = [c for c in cmd if c]
                env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    raise RuntimeError(f"whisperx failed:\n{result.stderr}")

                json_path = Path(output_dir) / f"{wav_filename.stem}.json"
                transcript = json.loads(json_path.read_text())

            words = []
            for i, segment in enumerate(transcript["segments"]):
                sentence = segment["text"].replace('"', "")
                for word in segment.get("words", []):
                    if "start" not in word:
                        continue
                    words.append(
                        {
                            "text": word["word"].replace('"', ""),
                            "start": word["start"],
                            "duration": word["end"] - word["start"],
                            "sequence_id": i,
                            "sentence": sentence,
                        }
                    )

            import pandas as pd

            return pd.DataFrame(words)

        ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_patched_get_transcript_from_audio)
        ExtractWordsFromAudio._patched_for_cpu_compute_type = True
        ExtractWordsFromAudio._original_get_transcript_from_audio = original
        logger.info("Applied whisperx CPU compute-type patch for tribev2.")

    def _prepare_checkpoint_dir(self, repo_id: str) -> str:
        """
        Prepare local checkpoint folder before calling TribeModel.from_pretrained.

        Why this exists:
        - Upstream tribev2 currently does `Path(checkpoint_dir)` and then `str(...)`.
        - On Windows this turns `facebook/tribev2` into `facebook\\tribev2`,
          which is an invalid Hugging Face repo id and triggers HFValidationError.
        - Workaround: download files first with hf_hub_download(repo_id=...), then
          pass a local folder path to from_pretrained so tribev2 goes through local path branch.
        """
        local_repo_dir = self.settings.tribe_cache_dir / "hf_repo" / repo_id.replace("/", "__")
        local_repo_dir.mkdir(parents=True, exist_ok=True)

        required_files = ("config.yaml", self.settings.tribe_checkpoint_name)
        missing = [name for name in required_files if not (local_repo_dir / name).exists()]
        if not missing:
            self._sanitize_windows_config_yaml(local_repo_dir / "config.yaml")
            logger.info("Using local TRIBE checkpoint cache at %s", local_repo_dir)
            return str(local_repo_dir)

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "huggingface_hub belum tersedia untuk download checkpoint TRIBE."
            ) from exc

        logger.info(
            "Downloading TRIBE checkpoint files from HF | repo_id=%s | target=%s | missing=%s",
            repo_id,
            local_repo_dir,
            ",".join(missing),
        )
        for filename in required_files:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_repo_dir),
            )
            logger.info("Downloaded/verified HF file | repo_id=%s | file=%s | local=%s", repo_id, filename, downloaded)
        self._sanitize_windows_config_yaml(local_repo_dir / "config.yaml")
        return str(local_repo_dir)

    @staticmethod
    def _sanitize_windows_config_yaml(config_path: Path) -> None:
        """
        Convert Windows-incompatible PyYAML PosixPath tags into plain POSIX strings.

        Upstream TRIBE config may include blocks like:
            folder: !!python/object/apply:pathlib.PosixPath
            - /
            - checkpoint
            - ...

        On Windows, yaml.UnsafeLoader fails while instantiating PosixPath.
        """
        if os.name != "nt" or not config_path.exists():
            return

        text = config_path.read_text(encoding="utf-8")
        marker = "!!python/object/apply:pathlib.PosixPath"

        lines = text.splitlines()
        rebuilt: list[str] = []
        changed = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            is_marker_line = marker in line
            is_folder_key = stripped.startswith("folder:")
            next_is_list = i + 1 < len(lines) and lines[i + 1].strip().startswith("- ")

            # Handle both:
            # 1) original marker form: folder: !!python/object/apply:pathlib.PosixPath
            # 2) previously broken partial form: folder:   + following list entries
            if not is_marker_line and not (is_folder_key and next_is_list):
                rebuilt.append(line)
                i += 1
                continue

            parts: list[str] = []
            j = i + 1
            while j < len(lines):
                sub = lines[j]
                sub_stripped = sub.strip()
                sub_indent = len(sub) - len(sub.lstrip(" "))

                # list item belongs to this folder value even if same indentation
                if sub_stripped.startswith("- ") and sub_indent >= indent:
                    value = sub_stripped[2:].strip()
                    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                    parts.append(value)
                    j += 1
                    continue
                break

            if parts:
                changed = True
                posix_path = PurePosixPath(*parts).as_posix()
                rebuilt.append(f'{" " * indent}folder: {posix_path}')
                logger.warning(
                    "Sanitized Windows-incompatible PosixPath/list path in config.yaml | key=folder | value=%s",
                    posix_path,
                )
                i = j
                continue

            if is_marker_line:
                changed = True
                # marker without list payload; keep plain folder key to avoid YAML custom tag crash
                rebuilt.append(line.replace(marker, "").rstrip())
                i += 1
                continue

            rebuilt.append(line)
            i += 1

        if changed:
            config_path.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")
            logger.info("Sanitized config.yaml for Windows compatibility | path=%s", config_path)

    @staticmethod
    def _extract_duration_seconds(events) -> float:
        try:
            if "duration" not in events.columns:
                return 0.0
            if "type" in events.columns:
                filtered = events[events["type"].isin(["Video", "Audio"])]
                if filtered.empty:
                    return 0.0
                row = filtered.iloc[0]
            else:
                row = events.iloc[0]
            start = float(row.get("start", 0.0))
            duration = float(row.get("duration", 0.0))
            return max(0.0, start + duration)
        except Exception:  # noqa: BLE001
            return 0.0


@lru_cache(maxsize=1)
def get_predictor() -> ViralityPredictor:
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Predictor singleton initialized | mode=%s", settings.predictor_mode)
    return ViralityPredictor(settings=settings)
