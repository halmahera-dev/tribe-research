# TRIBE v2 Virality Predictor Design

## Goal
Membangun web app end-to-end yang meniru alur "virality predictor":
upload video singkat -> analisis multimodal -> tampilkan virality score, hook score, hold rate, peak hook timestamp, timeline, dan brain-region heatmap.

## Scope
- Backend: FastAPI.
- Frontend: server-side templates (Jinja2) + static CSS/JS.
- Prediction engine:
  - Primary: `tribev2` (`TribeModel.from_pretrained("facebook/tribev2")`).
  - Fallback: heuristic mode jika model tidak tersedia/akses gagal.
- Input: video `.mp4/.mov/.mkv/.avi/.webm`.
- Batas durasi: default 15 detik.

## Architecture
- `app/main.py`: bootstrap FastAPI + static mount + router.
- `app/routers/web.py`: render halaman template.
- `app/routers/api.py`: endpoint upload/predict + validation.
- `app/services/predictor.py`: orchestrator inference (`tribev2` atau fallback).
- `app/services/scoring.py`: kalkulasi metrik + timeline + heatmap.
- `app/schemas/prediction.py`: response schema.
- `app/templates/*` + `app/static/*`: UI.

## Data Flow
1. User upload file melalui UI.
2. API validasi format + size.
3. File disimpan sementara ke `data/uploads`.
4. `ViralityPredictor.predict()` menjalankan:
   - mode `tribev2`: load model lazily, ekstrak events, validasi durasi, inferensi, scoring.
   - mode `auto`: jika `tribev2` gagal maka fallback heuristic.
5. API kembalikan JSON untuk dirender ke score cards, timeline bars, dan heatmap cells.
6. File sementara dihapus (default).

## Scoring Logic
- `intensity(t)` = mean absolute activation per timestep dari `preds`.
- `hook_score` berbasis sinyal awal.
- `hold_rate` berbasis rasio sinyal akhir terhadap awal.
- `consistency` berbasis standar deviasi intensitas.
- `virality_score` = weighted sum (`hook`, `hold`, `consistency`).
- `peak_hook_second` = timestamp intensitas maksimum.
- Heatmap per region dibuat dari pembagian vertex menjadi beberapa kelompok region representatif.

## Error Handling
- 400: format file salah, video kosong, durasi melebihi batas.
- 413: file terlalu besar.
- 500: mode `tribev2` dipaksa tapi model/dependency tidak tersedia.
- `auto` mode fallback ke heuristic agar UX tetap berjalan.

## Testing Strategy
- Unit test untuk fungsi scoring dan output shape.
- Smoke test import FastAPI app.
- Manual test UI upload via browser.

## Non-Goals
- Menyamakan model internal Higgsfield secara 1:1.
- Menjamin akurasi ilmiah virality final untuk keputusan bisnis.
- Training ulang TRIBE v2.

