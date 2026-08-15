// Search Studio — launch heuristic Hadamard searches, stream live progress
// over /ws/job/{id}, retune micromag SA mid-run, export verified results.
// Frames: {step,T,E,best_E,accepts} (micromag/tile SA), {step,f,best_f}
// (Williamson/GS/circulant SA), {iter,best_f,det_log10,is_hadamard}
// (maxdet ILS), plus matrix_png_b64 preview frames and a terminal
// {"type":"end"}.

import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { retintCanvas, fillPlusOne } from "/js/theme.js";

const ENGINES = ["maxdet", "micromag", "tile", "gerzon", "williamson", "gs", "circulant"];

let msgEl, statusEl, statsEl, previewCanvas, previewCtx;
let waveChart, tunePanel, cancelBtn, exportBtn;
let ws = null;
let currentJob = null;

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  n.append(...kids);
  return n;
}

function msg(text, kind = "") {
  msgEl.textContent = text;
  msgEl.className = `msg ${kind}`;
}

async function api(path, body) {
  const opts = body
    ? {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }
    : {};
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      detail = (await r.json()).detail || detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return r.json();
}

function statRow(k, v, cls = "") {
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, String(v)));
}

function drawPng(b64) {
  const img = new Image();
  img.onload = () => {
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
    retintCanvas(previewCanvas);
  };
  img.src = `data:image/png;base64,${b64}`;
}

function resetRun() {
  waveChart.clear();
  statsEl.replaceChildren();
  statusEl.textContent = "connecting…";
  exportBtn.style.display = "none";
  // all-+1 field until the first preview frame — also clears the previous
  // run's image (search engines generate their start internally, so the
  // server has nothing earlier to send)
  fillPlusOne(previewCanvas);
}

function handleFrame(d) {
  if (d.type === "snapshot") {
    statusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.matrix_png_b64) {
      drawPng(d.matrix_png_b64);
      return;
    }
    const energy = d.E ?? d.f;
    const best = d.best_E ?? d.best_f;
    if (energy !== undefined || best !== undefined || d.T !== undefined) {
      waveChart.push({ E: energy, BEST: best, T: d.T });
    }
    const bits = [];
    if (d.iter !== undefined) bits.push(`iter ${d.iter}`);
    if (d.step !== undefined) bits.push(`step ${d.step}`);
    if (best !== undefined) bits.push(`best ${typeof best === "number" ? best.toPrecision(4) : best}`);
    if (d.accepts !== undefined) bits.push(`accepts ${d.accepts}`);
    if (d.elapsed_s !== undefined) bits.push(`${d.elapsed_s.toFixed(1)}s`);
    if (d.is_hadamard) bits.push("HADAMARD");
    statusEl.textContent = bits.join(" · ");
    return;
  }
  if (d.type === "end") {
    statusEl.textContent = `job ${d.status}`;
    cancelBtn.disabled = true;
    finishRun();
    return;
  }
  if (d.type === "error") msg(d.error, "error");
}

async function finishRun() {
  if (!currentJob) return;
  try {
    const d = await api(`/api/search/${currentJob}`);
    const r = d.result || {};
    if (r.ok) {
      msg(`found Hadamard order ${r.order} via ${r.engine}`, "ok");
      if (r.png_b64) drawPng(r.png_b64);
      const s = r.stats || {};
      const rows = [
        statRow("order", r.order),
        statRow("is_hadamard", s.is_hadamard, s.is_hadamard ? "good" : "bad"),
        statRow("max_off", s.max_off),
        statRow("f", s.f),
      ];
      if (s.det_log10 !== undefined && s.det_log10 !== null)
        rows.push(statRow("det_log10", s.det_log10.toFixed(2)));
      const gz = r.gerzon;
      if (gz && gz.aligned) {
        const a = gz.aligned;
        rows.push(statRow("gerzon H₂", `${a.n_h2}/${a.n_cells}`, a.n_h2 === a.n_cells ? "good" : ""));
        rows.push(statRow("gerzon walls", a.n_wall));
      }
      statsEl.replaceChildren(...rows);
      exportBtn.style.display = "";
      exportBtn.disabled = false;
    } else {
      const best = r.best_f ?? r.best_E ?? r.info?.best_E ?? "?";
      msg(`budget done — no Hadamard yet (best ${best})`, "");
      statsEl.replaceChildren(
        statRow("ok", false),
        statRow("best", best),
        statRow("engine", r.engine || (d.params || {}).engine || "—")
      );
    }
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doLaunch() {
  const engine = document.getElementById("ss-engine").value;
  const mode = document.getElementById("ss-mode").value;
  const order = parseInt(document.getElementById("ss-order").value, 10);
  const budget_s = parseFloat(document.getElementById("ss-budget").value);
  const seedRaw = document.getElementById("ss-seed").value;
  const body = { engine, order, budget_s, mode };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);

  if (ws) ws.close();
  resetRun();
  msg("launching…");
  try {
    const { job_id } = await api("/api/search", body);
    currentJob = job_id;
    cancelBtn.disabled = false;
    msg(`job ${job_id} running`, "ok");
    tunePanel.classList.toggle("hidden", !(engine === "micromag" && mode === "sa"));
    ws = connect(`/ws/job/${job_id}`, { message: handleFrame });
  } catch (e) {
    msg(`launch failed: ${e.message}`, "error");
  }
}

async function doCancel() {
  if (!currentJob) return;
  try {
    if (ws) ws.send({ op: "cancel" });
    await api(`/api/search/${currentJob}/cancel`, {});
    msg(`cancel requested for ${currentJob}`);
  } catch (e) {
    msg(`cancel failed: ${e.message}`, "error");
  }
}

async function doExport() {
  if (!currentJob) return;
  try {
    const d = await api(`/api/search/${currentJob}/export`, {});
    msg(`exported → ${d.path}`, "ok");
    exportBtn.disabled = true;
  } catch (e) {
    msg(`export failed: ${e.message}`, "error");
  }
}

function sendTune() {
  if (!ws) return;
  ws.send({
    op: "set",
    cooling: parseFloat(document.getElementById("tune-cooling").value),
    lam_ex: parseFloat(document.getElementById("tune-lam-ex").value),
    lam_ani: parseFloat(document.getElementById("tune-lam-ani").value),
  });
}

function slider(id, label, min, max, step, value) {
  const val = el("span", { class: "slider-val", id: `${id}-val` }, String(value));
  const input = el("input", { id, type: "range", min, max, step, value });
  input.addEventListener("input", () => {
    val.textContent = input.value;
    sendTune();
  });
  return el("div", { class: "row slider-row" }, el("label", {}, label), input, val);
}

export function init(container) {
  const controls = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Launch search"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "engine"),
      el("select", { id: "ss-engine" }, ...ENGINES.map((e) => el("option", { value: e }, e)))
    ),
    el(
      "div",
      { class: "row" },
      el("label", {}, "mode"),
      el(
        "select",
        { id: "ss-mode" },
        el("option", { value: "ils" }, "ils"),
        el("option", { value: "sa" }, "sa")
      )
    ),
    el("div", { class: "row" }, el("label", {}, "order"), el("input", { id: "ss-order", type: "number", value: "64", min: "1" })),
    el("div", { class: "row" }, el("label", {}, "budget s"), el("input", { id: "ss-budget", type: "number", value: "30", min: "1", step: "1" })),
    el("div", { class: "row" }, el("label", {}, "seed"), el("input", { id: "ss-seed", type: "number", placeholder: "(random)" })),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "ss-launch" }, "Launch"),
      (cancelBtn = el("button", { class: "btn", id: "ss-cancel", disabled: true }, "Cancel")),
      (exportBtn = el("button", { class: "btn", id: "ss-export", style: "display:none" }, "Export to library"))
    ),
    (msgEl = el("div", { class: "msg" }))
  );

  tunePanel = el(
    "div",
    { class: "panel hidden", id: "ss-tune" },
    el("h2", {}, "Live retune (micromag SA)"),
    slider("tune-cooling", "cooling", "0.99", "0.99999", "0.00001", "0.999"),
    slider("tune-lam-ex", "lam_ex", "0", "1", "0.01", "0"),
    slider("tune-lam-ani", "lam_ani", "0", "1", "0.01", "0")
  );

  const waveCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  waveChart = makeStripChart(waveCanvas, null); // theme-derived colors

  previewCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  previewCtx = previewCanvas.getContext("2d");
  const previewCell = el(
    "div",
    { class: "sim-cell" },
    el("div", { class: "sim-label" }, "matrix (best)"),
    previewCanvas
  );

  // ONE strip chart (same convention as Micromag): E / BEST / T with
  // per-series toggles. Default E+T on; BEST is one click away.
  const WAVE_SERIES = ["E", "BEST", "T"];
  const WAVE_DEFAULT_ON = new Set(["E", "T"]);
  for (const n of WAVE_SERIES) waveChart.setVisible(n, WAVE_DEFAULT_ON.has(n));
  const waveRow = el(
    "div",
    { class: "btn-row layer-select" },
    ...WAVE_SERIES.map((name) => {
      const b = el("button", { class: "btn", "data-series": name }, `[${name.toUpperCase()}]`);
      b.addEventListener("click", () => {
        const on = !waveChart.isVisible(name);
        waveChart.setVisible(name, on);
        b.style.opacity = on ? "1" : "0.45";
      });
      b.style.opacity = WAVE_DEFAULT_ON.has(name) ? "1" : "0.45";
      return b;
    })
  );

  // ONE run panel: status on top, then matrix left + waveforms right
  // inside .run-viz (flex-wrap stacks on narrow windows).
  const runPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Run"),
    (statusEl = el("div", { class: "status-line" }, "idle")),
    el("table", { class: "stats" }, (statsEl = el("tbody"))),
    el(
      "div",
      { class: "run-viz" },
      previewCell,
      el("div", { class: "run-charts" }, waveRow, waveCanvas)
    )
  );

  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, controls, tunePanel),
      el("div", {}, runPanel)
    )
  );

  document.getElementById("ss-launch").addEventListener("click", doLaunch);
  cancelBtn.addEventListener("click", doCancel);
  exportBtn.addEventListener("click", doExport);
}

export function deactivate() {
  // close the job stream; the job itself keeps running server-side
  if (ws) ws.close();
  ws = null;
}

// cross-tab deep link from the Library tab: pre-fill the search order
window.addEventListener("hoa64:payload", (e) => {
  const d = e.detail || {};
  if (d.tab !== "search_studio" || !Number.isFinite(d.order)) return;
  const inp = document.getElementById("ss-order");
  if (inp) inp.value = d.order;
});
