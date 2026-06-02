const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const fileName = document.getElementById("file-name");
const statusBox = document.getElementById("status-box");
const analyzeBtn = document.getElementById("analyze-btn");
const dropzone = document.getElementById("dropzone");

const viralityScore = document.getElementById("virality-score");
const hookScore = document.getElementById("hook-score");
const holdRate = document.getElementById("hold-rate");
const peakSecond = document.getElementById("peak-second");
const engineBadge = document.getElementById("engine-badge");
const timelineBars = document.getElementById("timeline-bars");
const heatmapWrap = document.getElementById("heatmap-wrap");
const highlightList = document.getElementById("highlight-list");
const noteList = document.getElementById("note-list");
const scoreCards = document.querySelectorAll(".score-grid article");

function setStatus(kind, text) {
  statusBox.className = `status-box status-${kind}`;
  statusBox.textContent = text;
}

function formatScore(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return "--";
  return `${v.toFixed(1)}`;
}

function clearList(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function renderTimeline(points) {
  clearList(timelineBars);
  points.forEach((point) => {
    const bar = document.createElement("div");
    bar.className = "timeline-bar";
    bar.style.height = `${Math.max(6, point.intensity * 120)}px`;
    bar.title = `t=${point.second.toFixed(1)}s | intensity=${point.intensity.toFixed(2)} | drift=${point.focus_drift.toFixed(2)}`;
    timelineBars.appendChild(bar);
  });
}

function colorFromValue(v) {
  const x = Math.min(1, Math.max(0, v));
  const hue = 205 - (x * 55);      // cyan -> blue-green
  const light = 20 + (x * 55);
  return `hsl(${hue} 92% ${light}%)`;
}

function renderHeatmap(rows) {
  clearList(heatmapWrap);
  rows.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "heatmap-row";

    const label = document.createElement("div");
    label.className = "heatmap-label";
    label.textContent = row.region;

    const cells = document.createElement("div");
    cells.className = "heatmap-cells";
    row.values.forEach((value) => {
      const cell = document.createElement("span");
      cell.className = "heatmap-cell";
      cell.style.background = colorFromValue(value);
      cell.title = `${row.region}: ${value.toFixed(2)}`;
      cells.appendChild(cell);
    });

    wrapper.append(label, cells);
    heatmapWrap.appendChild(wrapper);
  });
}

function renderList(node, items, emptyText) {
  clearList(node);
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    node.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    node.appendChild(li);
  });
}

function fillResult(result) {
  viralityScore.textContent = formatScore(result.virality_score);
  hookScore.textContent = formatScore(result.hook_score);
  holdRate.textContent = `${formatScore(result.hold_rate)}%`;
  peakSecond.textContent = `${formatScore(result.peak_hook_second)}s`;
  engineBadge.textContent = `Engine: ${result.engine} | Confidence: ${(result.confidence * 100).toFixed(1)}%`;
  applyScoreTone(result.virality_score);

  renderTimeline(result.timeline || []);
  renderHeatmap(result.brain_heatmap || []);
  renderList(highlightList, result.highlights, "Belum ada highlight.");
  renderList(noteList, result.notes, "Tidak ada catatan tambahan.");
}

function applyScoreTone(score) {
  scoreCards.forEach((card) => {
    card.style.borderColor = "";
    card.style.boxShadow = "";
  });

  if (typeof score !== "number" || Number.isNaN(score)) return;
  const color =
    score >= 75 ? "rgba(52, 211, 153, 0.65)" :
    score >= 55 ? "rgba(56, 189, 248, 0.6)" :
    "rgba(251, 113, 133, 0.6)";
  const glow =
    score >= 75 ? "rgba(52, 211, 153, 0.22)" :
    score >= 55 ? "rgba(56, 189, 248, 0.2)" :
    "rgba(251, 113, 133, 0.2)";

  scoreCards.forEach((card) => {
    card.style.borderColor = color;
    card.style.boxShadow = `inset 0 0 0 1px ${glow}`;
  });
}

function refreshFileName() {
  const file = input.files && input.files[0];
  fileName.textContent = file ? file.name : "Belum ada file dipilih";
}

input.addEventListener("change", refreshFileName);

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  if (!event.dataTransfer || !event.dataTransfer.files.length) return;
  input.files = event.dataTransfer.files;
  refreshFileName();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files && input.files[0];
  if (!file) {
    setStatus("error", "Pilih video dulu sebelum analyze.");
    return;
  }

  const body = new FormData();
  body.append("video", file);
  analyzeBtn.disabled = true;
  setStatus("loading", "Menganalisis video... ini mungkin butuh beberapa saat.");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediksi gagal.");
    }
    fillResult(data);
    setStatus("success", "Analisis selesai. Cek score, timeline, dan heatmap di bawah.");
  } catch (error) {
    setStatus("error", `Error: ${error.message}`);
  } finally {
    analyzeBtn.disabled = false;
  }
});
