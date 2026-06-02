from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from app.schemas.prediction import (
    PredictionResponse,
    RegionHeatmapRow,
    ScoreBreakdown,
    TimelinePoint,
)

REGION_LABELS: tuple[str, ...] = (
    "Visual Cortex",
    "Motion Network",
    "Auditory Cortex",
    "Language Network",
    "Attention Network",
    "Memory Network",
    "Reward Circuit",
    "Control Network",
    "Social Cognition",
)


def _normalize_01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _extract_segment_seconds(segments: list[object] | None, expected: int) -> np.ndarray:
    if not segments:
        return np.arange(expected, dtype=np.float64)
    seconds: list[float] = []
    for idx, segment in enumerate(segments[:expected]):
        val = getattr(segment, "start", idx)
        try:
            seconds.append(float(val))
        except (TypeError, ValueError):
            seconds.append(float(idx))
    if len(seconds) < expected:
        seconds.extend(float(i) for i in range(len(seconds), expected))
    seconds_np = np.array(seconds, dtype=np.float64)
    if np.any(np.diff(seconds_np) < 0):
        return np.arange(expected, dtype=np.float64)
    return seconds_np


def _build_heatmap(preds: np.ndarray) -> list[RegionHeatmapRow]:
    n_regions = len(REGION_LABELS)
    n_vertices = preds.shape[1]
    vertex_indices = np.array_split(np.arange(n_vertices), n_regions)
    abs_preds = np.abs(preds)

    rows: list[RegionHeatmapRow] = []
    for label, region_idx in zip(REGION_LABELS, vertex_indices, strict=False):
        region_signal = np.mean(abs_preds[:, region_idx], axis=1)
        normalized_signal = _normalize_01(region_signal)
        rows.append(
            RegionHeatmapRow(
                region=label,
                values=[float(np.clip(v, 0.0, 1.0)) for v in normalized_signal],
            )
        )
    return rows


def _clip_100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _build_highlights(hook: float, hold: float, peak_second: float, consistency: float) -> list[str]:
    highlights: list[str] = []
    if hook >= 70:
        highlights.append(f"Hook opening kuat, puncak atensi di sekitar detik {peak_second:.1f}.")
    elif hook >= 50:
        highlights.append("Hook opening cukup baik, tapi masih bisa dibuat lebih cepat dan tegas.")
    else:
        highlights.append("Hook opening lemah; perkuat 2-3 detik awal agar stop-scroll meningkat.")

    if hold >= 70:
        highlights.append("Retention stabil sampai akhir klip.")
    elif hold >= 50:
        highlights.append("Retention menengah; ada beberapa titik drop yang perlu dirapikan.")
    else:
        highlights.append("Retention turun signifikan; pertimbangkan pacing lebih rapat.")

    if consistency >= 65:
        highlights.append("Intensitas antar detik relatif konsisten.")
    else:
        highlights.append("Intensitas fluktuatif; fokus narasi perlu dipadatkan.")

    return highlights


def build_prediction_from_preds(
    preds: np.ndarray,
    segments: list[object] | None,
    *,
    engine: str,
) -> PredictionResponse:
    if preds.ndim != 2 or preds.shape[0] < 2 or preds.shape[1] < 2:
        raise ValueError("Prediksi TRIBE tidak valid untuk dihitung skornya.")

    seconds = _extract_segment_seconds(segments, preds.shape[0])
    raw_intensity = np.mean(np.abs(preds), axis=1)
    intensity = _normalize_01(_smooth(raw_intensity, window=3))
    drift = _normalize_01(np.abs(np.diff(intensity, prepend=intensity[0])))

    early_window = max(2, min(5, intensity.size // 2 if intensity.size > 3 else intensity.size))
    late_window = max(2, early_window)

    early_signal = intensity[:early_window]
    late_signal = intensity[-late_window:]

    hook_score = _clip_100(100 * (0.7 * float(np.max(early_signal)) + 0.3 * float(np.mean(early_signal))))
    hold_ratio = float(np.mean(late_signal) / (np.mean(early_signal) + 1e-6))
    hold_rate = _clip_100(100 * np.clip(hold_ratio, 0.0, 1.2) / 1.2)
    consistency = _clip_100(100 * (1.0 - float(np.std(intensity))))

    peak_idx = int(np.argmax(intensity))
    peak_hook_second = float(seconds[peak_idx])

    virality_score = _clip_100(0.45 * hook_score + 0.35 * hold_rate + 0.20 * consistency)
    confidence = float(np.clip(0.55 + 0.35 * (preds.shape[0] / (preds.shape[0] + 12)), 0.0, 1.0))

    timeline = [
        TimelinePoint(
            second=float(sec),
            intensity=float(np.clip(intensity[i], 0.0, 1.0)),
            focus_drift=float(np.clip(drift[i], 0.0, 1.0)),
        )
        for i, sec in enumerate(seconds)
    ]

    heatmap = _build_heatmap(preds)
    highlights = _build_highlights(hook_score, hold_rate, peak_hook_second, consistency)

    return PredictionResponse(
        engine=engine,
        duration_seconds=float(np.max(seconds) + 1 if seconds.size else 0),
        virality_score=virality_score,
        hook_score=hook_score,
        hold_rate=hold_rate,
        peak_hook_second=peak_hook_second,
        confidence=confidence,
        score_breakdown=ScoreBreakdown(
            hook=hook_score,
            hold=hold_rate,
            consistency=consistency,
        ),
        timeline=timeline,
        brain_heatmap=heatmap,
        highlights=highlights,
        notes=[],
    )


def build_heuristic_prediction(video_path: Path, max_video_seconds: int) -> PredictionResponse:
    payload = video_path.read_bytes()
    if not payload:
        raise ValueError("File video kosong.")

    # Heuristic fallback path so the app remains usable when TRIBE cannot load.
    sampled = payload[: min(len(payload), 3_000_000)]
    arr = np.frombuffer(sampled, dtype=np.uint8).astype(np.float64)
    if arr.size < 32:
        arr = np.pad(arr, (0, 32 - arr.size), mode="edge")

    n_points = max(8, min(max_video_seconds, 24))
    chunks = np.array_split(arr, n_points)
    energy = np.array([np.std(chunk) for chunk in chunks], dtype=np.float64)
    dynamic = np.array([np.mean(np.abs(np.diff(chunk))) if chunk.size > 1 else 0 for chunk in chunks], dtype=np.float64)
    signal = _normalize_01(0.6 * energy + 0.4 * dynamic)

    # Expand into pseudo cortical vertices for a consistent heatmap pipeline.
    seed_int = int(hashlib.sha1(sampled[:4096]).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)
    vertices = 900
    base = rng.normal(0.0, 0.22, size=(signal.size, vertices))
    modulation = signal[:, None] * rng.uniform(0.6, 1.3, size=(signal.size, vertices))
    pseudo_preds = base + modulation

    result = build_prediction_from_preds(pseudo_preds, None, engine="heuristic")
    result.notes = [
        "Mode fallback aktif: model TRIBE belum tersedia di runtime ini.",
        "Score dipakai untuk prototyping UX/flow, bukan evaluasi ilmiah final.",
    ]
    return result

