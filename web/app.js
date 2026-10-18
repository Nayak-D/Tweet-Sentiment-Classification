const COLORS = { negative: "#f17679", neutral: "#8c7cf0", positive: "#42bd98" };
const LABELS = ["negative", "neutral", "positive"];
const MODEL_LABELS = {
  majority_baseline: "Majority baseline",
  multinomial_nb: "Naive Bayes",
  logistic_regression: "Logistic Regression",
  linear_svm: "Linear SVM",
};
const state = { summary: null, page: 1, totalPages: 1, searchTimer: null };
const $ = (id) => document.getElementById(id);
const number = (value) => new Intl.NumberFormat().format(value);

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  return { context, width: rect.width, height: rect.height };
}

function drawPerformanceChart() {
  if (!state.summary) return;
  const canvas = $("performance-chart");
  const { context: ctx, width, height } = setupCanvas(canvas);
  const metric = $("metric-selector").value;
  const validationKey = metric === "accuracy" ? "validation_accuracy" : "validation_macro_f1";
  const testKey = metric === "accuracy" ? "test_accuracy" : "test_macro_f1";
  const scores = state.summary.model_scores;
  const left = 46, right = 14, top = 14, bottom = 35;
  const chartHeight = height - top - bottom;
  const chartWidth = width - left - right;
  ctx.font = "10px DM Sans, sans-serif";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= 5; tick++) {
    const score = tick / 5;
    const y = top + chartHeight * (1 - score);
    ctx.strokeStyle = "#edf0f5";
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    ctx.fillStyle = "#9ba3b0";
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(score * 100)}%`, left - 8, y);
  }
  const groupWidth = chartWidth / scores.length;
  const barWidth = Math.min(23, groupWidth * 0.27);
  scores.forEach((row, index) => {
    const center = left + groupWidth * (index + 0.5);
    const values = [[row[validationKey], "#a191f2"], [row[testKey], "#55b9a0"]];
    values.forEach(([value, color], barIndex) => {
      const barHeight = Math.max(0, value * chartHeight);
      const x = center + (barIndex === 0 ? -barWidth - 2 : 2);
      const y = top + chartHeight - barHeight;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, [4, 4, 0, 0]);
      ctx.fill();
    });
    ctx.fillStyle = "#788196";
    ctx.textAlign = "center";
    ctx.font = "9px DM Sans, sans-serif";
    ctx.fillText(MODEL_LABELS[row.model] || row.model, center, height - 13);
  });
}

function drawDistribution() {
  if (!state.summary) return;
  const split = $("split-selector").value;
  const counts = state.summary.split_counts[split];
  const total = state.summary.split_sizes[split];
  $("split-total").textContent = number(total);
  const canvas = $("distribution-chart");
  const { context: ctx, width, height } = setupCanvas(canvas);
  const centerX = width / 2, centerY = height / 2, radius = Math.min(width, height) * 0.46;
  let start = -Math.PI / 2;
  LABELS.forEach((label) => {
    const portion = counts[label] / total;
    const end = start + portion * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, start, end);
    ctx.strokeStyle = COLORS[label];
    ctx.lineWidth = Math.max(14, radius * 0.26);
    ctx.lineCap = "butt";
    ctx.stroke();
    start = end;
  });
  $("distribution-legend").replaceChildren(...LABELS.map((label) => {
    const item = document.createElement("div");
    const dot = document.createElement("i");
    dot.style.backgroundColor = COLORS[label];
    const name = document.createElement("span");
    name.textContent = label[0].toUpperCase() + label.slice(1);
    const count = document.createElement("small");
    count.textContent = `${number(counts[label])} · ${(counts[label] / total * 100).toFixed(1)}%`;
    item.append(dot, name, count);
    return item;
  }));
}

function drawCrossValidation() {
  if (!state.summary) return;
  const canvas = $("cv-chart");
  const { context: ctx, width, height } = setupCanvas(canvas);
  const models = ["multinomial_nb", "logistic_regression", "linear_svm"];
  const colors = ["#83b6ec", "#9a88ec", "#55b9a0"];
  const left = 43, right = 12, top = 12, bottom = 31;
  const minValue = 0.55, maxValue = 0.70;
  const chartH = height - top - bottom;
  const chartW = width - left - right;
  ctx.font = "9px DM Sans, sans-serif";
  ctx.textBaseline = "middle";
  [0.55, 0.60, 0.65, 0.70].forEach((value) => {
    const y = top + (maxValue - value) / (maxValue - minValue) * chartH;
    ctx.strokeStyle = "#edf0f5";
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    ctx.fillStyle = "#9ba3b0";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(2), left - 7, y);
  });
  const groupW = chartW / models.length;
  models.forEach((model, index) => {
    const rows = state.summary.cross_validation.filter((row) => row.model === model);
    const best = rows.reduce((a, b) => (b.mean_macro_f1 > a.mean_macro_f1 ? b : a));
    const x = left + groupW * (index + 0.5);
    const y = top + (maxValue - best.mean_macro_f1) / (maxValue - minValue) * chartH;
    const barBottom = top + chartH;
    const barWidth = Math.min(37, groupW * 0.38);
    ctx.fillStyle = colors[index];
    ctx.beginPath(); ctx.roundRect(x - barWidth / 2, y, barWidth, barBottom - y, [4, 4, 0, 0]); ctx.fill();
    const spread = Math.max(best.std_macro_f1, 0.002);
    const topY = top + (maxValue - (best.mean_macro_f1 + spread)) / (maxValue - minValue) * chartH;
    const bottomY = top + (maxValue - (best.mean_macro_f1 - spread)) / (maxValue - minValue) * chartH;
    ctx.strokeStyle = "#47536a";
    ctx.beginPath(); ctx.moveTo(x, topY); ctx.lineTo(x, bottomY); ctx.moveTo(x - 5, topY); ctx.lineTo(x + 5, topY); ctx.moveTo(x - 5, bottomY); ctx.lineTo(x + 5, bottomY); ctx.stroke();
    ctx.fillStyle = "#788196"; ctx.textAlign = "center";
    ctx.fillText(MODEL_LABELS[model], x, height - 12);
  });
}

function drawConfusionMatrix() {
  const matrix = state.summary.confusion_matrix;
  const largest = Math.max(...matrix.flat());
  const grid = $("confusion-matrix");
  grid.replaceChildren();
  const blank = document.createElement("div"); grid.append(blank);
  LABELS.forEach((label) => {
    const heading = document.createElement("div"); heading.className = "matrix-cell matrix-head";
    heading.textContent = label.slice(0, 3).toUpperCase(); grid.append(heading);
  });
  LABELS.forEach((actual, row) => {
    const rowHeading = document.createElement("div"); rowHeading.className = "matrix-cell matrix-head";
    rowHeading.textContent = actual.slice(0, 3).toUpperCase(); grid.append(rowHeading);
    LABELS.forEach((predicted, col) => {
      const cell = document.createElement("div"); cell.className = "matrix-cell";
      const intensity = matrix[row][col] / largest;
      cell.style.backgroundColor = row === col
        ? `rgba(85, 185, 160, ${0.10 + intensity * 0.35})`
        : `rgba(140, 124, 240, ${0.04 + intensity * 0.14})`;
      cell.style.color = intensity > 0.55 ? "#273148" : "#6f7a8d";
      cell.textContent = number(matrix[row][col]);
      cell.title = `Actual ${actual}; predicted ${predicted}: ${number(matrix[row][col])}`;
      grid.append(cell);
    });
  });
}

function updateStats() {
  const total = Object.values(state.summary.split_sizes).reduce((sum, value) => sum + value, 0);
  const selected = state.summary.model_scores.find((row) => row.model === state.summary.selected_model);
  $("total-samples").textContent = number(total);
  $("train-size").textContent = number(state.summary.split_sizes.train);
  $("test-accuracy").textContent = `${(selected.test_accuracy * 100).toFixed(1)}%`;
  $("selected-model").textContent = MODEL_LABELS[state.summary.selected_model] || state.summary.selected_model;
}

async function loadExamples() {
  const split = $("example-split").value;
  const sentiment = $("example-label").value;
  const query = $("example-search").value.trim();
  const params = new URLSearchParams({ split, sentiment, q: query, page: String(state.page) });
  const list = $("example-list");
  list.innerHTML = '<div class="loading">Loading examples…</div>';
  try {
    const data = await getJSON(`/api/examples?${params.toString()}`);
    state.totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
    $("example-count").textContent = `${number(data.total)} matching examples · page ${state.page}/${state.totalPages}`;
    $("previous-page").disabled = state.page <= 1;
    $("next-page").disabled = state.page >= state.totalPages;
    if (data.examples.length === 0) {
      list.innerHTML = '<div class="empty-state">No matching tweets in this split.</div>';
      return;
    }
    list.replaceChildren(...data.examples.map((example) => {
      const row = document.createElement("div"); row.className = "example-row";
      const text = document.createElement("p"); text.textContent = example.text;
      const label = document.createElement("span");
      label.className = `label-chip ${example.sentiment}`;
      label.textContent = example.sentiment;
      row.append(text, label);
      return row;
    }));
  } catch (error) {
    list.innerHTML = "";
    const message = document.createElement("div"); message.className = "empty-state";
    message.textContent = error.message;
    list.append(message);
  }
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

async function submitPrediction(event) {
  event.preventDefault();
  const text = $("text-input").value.trim();
  $("prediction-error").textContent = "";
  $("prediction-result").hidden = true;
  if (!text) {
    $("prediction-error").textContent = "Enter some text first.";
    return;
  }
  const button = $("predict-form").querySelector("button");
  button.disabled = true;
  button.firstChild.textContent = "Analyzing ";
  try {
    const result = await getJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const card = $("prediction-result");
    const emoji = document.createElement("span");
    emoji.className = `result-emoji ${result.sentiment}`;
    emoji.textContent = { positive: "↑", neutral: "↔", negative: "↓" }[result.sentiment];
    const copy = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = result.sentiment;
    const note = document.createElement("small"); note.textContent = "Model prediction · not a certainty score";
    copy.append(title, note); card.replaceChildren(emoji, copy); card.hidden = false;
  } catch (error) {
    $("prediction-error").textContent = error.message;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Analyze text ";
  }
}

async function init() {
  try {
    state.summary = await getJSON("/api/summary");
    updateStats();
    drawPerformanceChart();
    drawDistribution();
    drawCrossValidation();
    drawConfusionMatrix();
  } catch (error) {
    showToast(`Dashboard data error: ${error.message}`);
  }
  await loadExamples();
}

$("metric-selector").addEventListener("change", drawPerformanceChart);
$("split-selector").addEventListener("change", drawDistribution);
$("predict-form").addEventListener("submit", submitPrediction);
$("text-input").addEventListener("input", (event) => {
  $("character-count").textContent = event.target.value.length;
});
["example-split", "example-label"].forEach((id) => $(id).addEventListener("change", () => {
  state.page = 1; loadExamples();
}));
$("example-search").addEventListener("input", () => {
  window.clearTimeout(state.searchTimer);
  state.page = 1;
  state.searchTimer = window.setTimeout(loadExamples, 250);
});
$("previous-page").addEventListener("click", () => {
  if (state.page > 1) { state.page -= 1; loadExamples(); }
});
$("next-page").addEventListener("click", () => {
  if (state.page < state.totalPages) { state.page += 1; loadExamples(); }
});
window.addEventListener("resize", () => {
  if (!state.summary) return;
  drawPerformanceChart(); drawDistribution(); drawCrossValidation();
});
init();
