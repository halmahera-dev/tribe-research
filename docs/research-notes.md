# Research Notes - Higgsfield Virality Predictor vs TRIBE v2

Tanggal riset: 2026-05-15

## Ringkasan Temuan
- Halaman resmi Higgsfield Virality Predictor menjelaskan alur:
  1) upload clip sampai 15 detik,
  2) model audience/brain response analysis,
  3) hasil berupa virality score, peak hook timestamp, hold rate, dan brain heatmap.
- Repo resmi `facebookresearch/tribev2` memang model multimodal brain-response prediction (video/audio/text -> fMRI response).
- API quick-start TRIBE v2:
  - `TribeModel.from_pretrained("facebook/tribev2")`
  - `get_events_dataframe(video_path=...)`
  - `predict(events=...)` -> `(preds, segments)`

## Validitas Klaim "Higgsfield pakai TRIBE v2"
- Klaim tersebut beredar di komunitas, tapi tidak ditemukan pernyataan resmi publik dari Higgsfield yang secara eksplisit mengonfirmasi stack model internalnya.
- Implementasi di repo ini didesain sebagai rekonstruksi teknis yang "plausible" menggunakan `tribev2`, bukan reverse-engineering resmi.

## Constraint Penting
- Lisensi `tribev2` adalah `CC BY-NC 4.0` (non-commercial).
- Notebook demo `tribev2` menyebut potensi kebutuhan akses model tertentu di HuggingFace (contoh: Llama-3.2 access).
- Karena itu, app menyediakan fallback heuristic agar flow produk tetap bisa dites saat model path primer belum siap.

