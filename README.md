# TRIBE v2 Virality Predictor (FastAPI + Templates)

Prototype end-to-end untuk membuat virality predictor ala Higgsfield, menggunakan `tribev2` sebagai engine utama dan FastAPI + Jinja templates untuk frontend.

## What You Get
- Upload video hook dari browser.
- Prediksi:
  - virality score
  - hook score
  - hold rate
  - peak hook timestamp
  - attention timeline
  - brain-region heatmap
- Engine mode:
  - `tribev2` (utama)
  - `auto` (tribev2 -> fallback heuristic)
  - `heuristic` (langsung fallback, cocok untuk prototyping UI)

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8122
```

Open: [http://127.0.0.1:8122](http://127.0.0.1:8122)

## Docker (No DB)
Jalankan full app via Docker Compose tanpa database:

```bash
docker compose up --build
```

Image Docker sudah preinstall spaCy model `en_core_web_lg`, jadi tidak ada download model NLP saat request prediksi pertama.

Open: [http://127.0.0.1:8122](http://127.0.0.1:8122)

Stop:
```bash
docker compose down
```

## Configuration
Set lewat environment variables:

```bash
set PREDICTOR_MODE=auto
set MAX_VIDEO_SECONDS=15
set MAX_UPLOAD_MB=120
set TRIBE_REPO_ID=facebook/tribev2
set TRIBE_CHECKPOINT_NAME=best.ckpt
set TRIBE_CACHE_DIR=.\.cache\tribe
set TRIBE_DEVICE=auto
set TASK_DEVICE=auto
set TASK_NUM_WORKERS=8
set KEEP_UPLOADS=false
set LOG_LEVEL=INFO
```

Keterangan:
- `TASK_DEVICE=auto|cpu|gpu` -> auto detect device runtime untuk semua extractor utama.
- `TASK_NUM_WORKERS` -> jumlah worker dataloader TRIBE (turunkan jika mesin terasa berat/freeze).

## API
- `POST /api/predict` (multipart/form-data, field: `video`)
- `GET /api/health`

## Project Structure
```text
app/
  core/config.py
  routers/api.py
  routers/web.py
  schemas/prediction.py
  services/predictor.py
  services/scoring.py
  templates/
  static/
docs/
  research-notes.md
  superpowers/specs/2026-05-15-virality-predictor-design.md
tests/
```

## Testing
```bash
pytest -q
```

## Notes on TRIBE v2
- Engine utama mengikuti quick-start resmi `tribev2`:
  - `TribeModel.from_pretrained("facebook/tribev2")`
  - `get_events_dataframe(video_path=...)`
  - `predict(events=...)`
- Jika akses/dependency model belum siap, app fallback ke heuristic mode agar flow tetap bisa dipakai untuk UX testing.

## Troubleshooting (Windows HFValidationError)
Jika muncul error:
`HFValidationError ... 'facebook\tribev2'`

Artinya package `tribev2` terkena bug konversi `Path()` di Windows (slash `/` berubah jadi `\` saat dipakai sebagai Hugging Face repo id).

Workaround sudah dipasang di service ini:
- file model (`config.yaml`, `best.ckpt`) didownload dulu via `hf_hub_download(repo_id="facebook/tribev2", ...)`
- lalu `from_pretrained()` dipanggil menggunakan local checkpoint directory.

Jadi untuk apply fix:
1. restart server penuh (jangan hot reload saja),
2. pastikan `TRIBE_REPO_ID=facebook/tribev2`.

## Licensing Reminder
`tribev2` dilisensikan `CC BY-NC 4.0` (non-commercial). Pastikan use-case kamu sesuai ketentuan lisensi sebelum dipakai untuk produksi komersial.
