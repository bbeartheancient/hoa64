// Noise Lab — DiT noise classifier training & analysis (Phase: routes_noise).
//   TRAIN    POST /api/noise/train → {job_id}, streams /ws/job/{id} frames
//            {epoch, loss, acc, val_acc, classes} onto ONE strip chart
//            ([LOSS][ACC][VAL_ACC] data-series toggles). [CANCEL] → ws
//            {op:"cancel"} + POST cancel; on end GET /api/search/{id}
//            shows val_acc / skipped classes.
//   ANALYZE  POST /api/noise/analyze {path} XOR {live_seconds[, live_source]}
//            → top class, per-class probabilities (horizontal bar list, fg
//            bars on a dim track, top class .good) and the log-mel spectrogram
//            as a retinted PNG canvas. live_source=mic records from the
//            default mic (trusted-local, like the other path-taking
//            endpoints); the RF sources (wifi/ble) capture local radio-counter
//            telemetry via rf_capture and render it to baseband — the response
//            then carries a `capture` stats block shown in the status line.
//            GET /api/noise/classes reports live_sources availability
//            (unavailable sources are disabled in the selector).
// Model status comes from GET /api/noise/classes on init.

import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { retintCanvas, themeColor } from "/js/theme.js";

let msgEl, modelStatusEl, tStatusEl, tStatsEl, cancelBtn, trainBtn;
let melCanvas, melCell, barsHost, sourceSel;
let tChart;
let ws = null;
let currentJob = null;
let lastAnalysis = null; // {top, probs} — re-rendered on themechange

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

function numVal(id, fallback) {
  const v = parseFloat(document.getElementById(id).value);
  return Number.isFinite(v) ? v : fallback;
}

function statRow(k, v, cls = "") {
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, String(v)));
}

function drawPng(canvas, b64) {
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    retintCanvas(canvas); // registers for themechange re-tints
  };
  img.src = `data:image/png;base64,${b64}`;
}

// ---- TRAIN ------------------------------------------------------------------

function handleTrainFrame(d) {
  if (d.type === "snapshot") {
    tStatusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleTrainFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.status_text) tStatusEl.textContent = d.status_text; // pre-epoch phase (dataset load)
    if (d.epoch !== undefined) {
      tChart.push({ LOSS: d.loss, ACC: d.acc, VAL_ACC: d.val_acc });
      const bits = [`epoch ${d.epoch}`];
      if (typeof d.loss === "number") bits.push(`loss ${d.loss.toFixed(4)}`);
      if (typeof d.acc === "number") bits.push(`acc ${(d.acc * 100).toFixed(1)}%`);
      if (typeof d.val_acc === "number") bits.push(`val ${(d.val_acc * 100).toFixed(1)}%`);
      tStatusEl.textContent = bits.join(" · ");
    }
    return;
  }
  if (d.type === "end") {
    tStatusEl.textContent = `job ${d.status}`;
    cancelBtn.disabled = true;
    trainBtn.disabled = false;
    finishTrain();
    return;
  }
  if (d.type === "error") msg(d.error, "error");
}

async function finishTrain() {
  if (!currentJob) return;
  try {
    const d = await api(`/api/search/${currentJob}`);
    const r = d.result || {};
    const rows = [];
    if (typeof r.val_acc === "number")
      rows.push(statRow("val_acc", `${(r.val_acc * 100).toFixed(1)}%`, r.val_acc >= 0.5 ? "good" : ""));
    if (r.epochs_ran !== undefined) rows.push(statRow("epochs_ran", r.epochs_ran));
    if (Array.isArray(r.skipped) && r.skipped.length)
      rows.push(statRow("skipped", r.skipped.join(", "), "bad"));
    if (r.optimizer) rows.push(statRow("optimizer", r.optimizer));
    if (r.out_path) rows.push(statRow("out_path", r.out_path, "dim"));
    tStatsEl.replaceChildren(...rows);
    msg(
      r.val_acc !== undefined
        ? `training complete — val_acc ${(r.val_acc * 100).toFixed(1)}%` +
            (r.skipped && r.skipped.length ? ` · skipped: ${r.skipped.join(", ")}` : "")
        : "training finished",
      "ok"
    );
    refreshClasses();
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doTrain() {
  const winRaw = document.getElementById("noi-t-windows").value;
  const body = {
    epochs: parseInt(document.getElementById("noi-t-epochs").value, 10),
    batch_size: parseInt(document.getElementById("noi-t-batch").value, 10),
    max_windows_per_class: winRaw === "" ? null : parseInt(winRaw, 10),
    budget_s: numVal("noi-t-budget", 1800),
    optimizer: document.getElementById("noi-t-opt").value,
  };
  if (ws) ws.close();
  tChart.clear();
  tStatsEl.replaceChildren();
  tStatusEl.textContent = "connecting…";
  msg("starting training… (first run downloads NOISEX-92)");
  try {
    const { job_id } = await api("/api/noise/train", body);
    currentJob = job_id;
    cancelBtn.disabled = false;
    trainBtn.disabled = true;
    msg(`job ${job_id} running`, "ok");
    ws = connect(`/ws/job/${job_id}`, { message: handleTrainFrame });
  } catch (e) {
    msg(`train failed: ${e.message}`, "error");
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

async function refreshClasses() {
  try {
    const d = await api("/api/noise/classes");
    const m = d.model || {};
    const synth = new Set(d.synth_classes || []);
    const nRec = (d.recorded_classes || []).length;
    const nSyn = synth.size;
    modelStatusEl.textContent = m.trained
      ? `model trained — ${m.path}`
      : "no trained model — train a model first";
    modelStatusEl.className = m.trained ? "status-line good" : "status-line dim";
    if (sourceSel && d.live_sources) {
      const prev = sourceSel.value || "mic";
      sourceSel.replaceChildren(
        ...Object.entries(d.live_sources).map(([name, s]) => {
          const opt = el(
            "option",
            { value: name },
            s.available
              ? name.toUpperCase() + (s.iface || s.dev ? ` (${s.iface || s.dev})` : "")
              : `${name.toUpperCase()} — ${s.reason || "unavailable"}`
          );
          if (!s.available) opt.disabled = true;
          return opt;
        })
      );
      sourceSel.value = d.live_sources[prev] && d.live_sources[prev].available ? prev : "mic";
    }
    const host = document.getElementById("noi-class-list");
    if (host) {
      host.replaceChildren(
        el("div", { class: "dim" },
          `${(d.classes || []).length} labels · ${nRec} NOISEX-92 · ${nSyn} RF synth (baseband envelope, not the carrier)`),
        el("div", { class: "btn-row" },
          ...(d.classes || []).map((name) =>
            el("span", {
              class: "btn btn-xs",
              style: synth.has(name) ? "opacity:1" : "opacity:0.6;cursor:default",
              title: synth.has(name) ? "synthesized RF baseband" : "NOISEX-92 recording",
            }, synth.has(name) ? `${name}*` : name)))
      );
    }
  } catch (e) {
    modelStatusEl.textContent = `classes fetch failed: ${e.message}`;
    modelStatusEl.className = "status-line bad";
  }
}

// ---- ANALYZE ----------------------------------------------------------------

function renderBars() {
  if (!barsHost) return;
  if (!lastAnalysis) {
    barsHost.replaceChildren(el("div", { class: "dim" }, "no analysis yet"));
    return;
  }
  const { top, probs } = lastAnalysis;
  const entries = Object.entries(probs || {}).sort((a, b) => b[1] - a[1]);
  const fg = themeColor("fg");
  const dim = themeColor("dim");
  barsHost.replaceChildren(
    ...entries.map(([name, p]) => {
      const pct = (p * 100).toFixed(1);
      const isTop = name === top;
      const track = el(
        "div",
        { style: `flex:1;height:10px;background:${dim};opacity:0.9` },
        el("div", { style: `height:100%;width:${pct}%;background:${fg}` })
      );
      return el(
        "div",
        { class: `row ${isTop ? "good" : ""}`, style: "gap:8px;align-items:center" },
        el("span", { style: "width:130px;overflow:hidden;text-overflow:ellipsis" }, name),
        track,
        el("span", { class: isTop ? "good" : "dim", style: "width:56px;text-align:right" }, `${pct}%`)
      );
    })
  );
}

async function doAnalyze(body) {
  msg("analyzing…");
  try {
    const d = await api("/api/noise/analyze", body);
    lastAnalysis = { top: d.top, probs: d.probs };
    if (d.mel_png_b64) {
      melCell.classList.remove("hidden");
      drawPng(melCanvas, d.mel_png_b64);
    }
    renderBars();
    const cap = d.capture;
    const capBit = cap
      ? ` · ${cap.source} ${cap.iface || cap.dev || ""}` +
        ` · ${cap.packets ?? cap.events ?? 0} ${cap.source === "wifi" ? "pkts" : "evts"}` +
        ` · ${Math.round((cap.duty || 0) * 100)}% duty` +
        (cap.note ? ` · ${cap.note}` : "")
      : "";
    msg(
      `top: ${d.top} (${((d.probs[d.top] || 0) * 100).toFixed(1)}%) · ` +
        `fs ${d.fs} Hz · ${Number(d.duration_s).toFixed(2)} s${capBit}`,
      "ok"
    );
  } catch (e) {
    msg(`analyze failed: ${e.message}`, "error");
  }
}

function doAnalyzePath() {
  const path = document.getElementById("noi-a-path").value.trim();
  if (!path) {
    msg("give a WAV path (or use CAPTURE)", "error");
    return;
  }
  doAnalyze({ path });
}

function doCapture() {
  const secs = numVal("noi-a-live", 3);
  const source = sourceSel ? sourceSel.value : "mic";
  msg(
    source === "mic"
      ? `capturing ${secs}s from the default mic…`
      : `capturing ${secs}s of ${source} radio telemetry…`
  );
  doAnalyze({ live_seconds: secs, live_source: source });
}

// ---- tab lifecycle ----------------------------------------------------------

function seriesToggleRow(chart, names, defaultOn) {
  return el(
    "div",
    { class: "btn-row layer-select" },
    ...names.map((name) => {
      const b = el("button", { class: "btn", "data-series": name }, `[${name}]`);
      b.addEventListener("click", () => {
        const on = !chart.isVisible(name);
        chart.setVisible(name, on);
        b.style.opacity = on ? "1" : "0.45";
      });
      b.style.opacity = defaultOn.has(name) ? "1" : "0.45";
      chart.setVisible(name, defaultOn.has(name));
      return b;
    })
  );
}

function labeledCell(label, canvas) {
  return el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, label), canvas);
}

export function init(container) {
  // ---- TRAIN panel
  const tChartCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  tChart = makeStripChart(tChartCanvas, null);
  const trainPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Train classifier"),
    el("div", { class: "row" }, el("label", {}, "epochs"), el("input", { id: "noi-t-epochs", type: "number", value: "8", min: "1", max: "200" })),
    el("div", { class: "row" }, el("label", {}, "batch size"), el("input", { id: "noi-t-batch", type: "number", value: "64", min: "8", max: "512", step: "8" })),
    el("div", { class: "row" }, el("label", {}, "max windows/class"), el("input", { id: "noi-t-windows", type: "number", placeholder: "(all)", min: "16", max: "5000" })),
    el("div", { class: "row" }, el("label", {}, "budget s"), el("input", { id: "noi-t-budget", type: "number", value: "1800", min: "10" })),
    el("div", { class: "row" }, el("label", {}, "optimizer"),
      el("select", { id: "noi-t-opt" },
        el("option", { value: "muon", selected: "selected" }, "MUON (Dion3)"),
        el("option", { value: "adamw" }, "ADAMW"))),
    el(
      "div",
      { class: "btn-row" },
      (trainBtn = el("button", { class: "btn", id: "noi-t-run" }, "Train")),
      (cancelBtn = el("button", { class: "btn", disabled: true }, "Cancel"))
    ),
    seriesToggleRow(tChart, ["LOSS", "ACC", "VAL_ACC"], new Set(["LOSS", "VAL_ACC"])),
    tChartCanvas,
    (tStatusEl = el("div", { class: "status-line" }, "idle")),
    el("table", { class: "stats" }, (tStatsEl = el("tbody")))
  );

  // ---- ANALYZE panel
  melCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  melCell = labeledCell("log-mel", melCanvas);
  melCell.classList.add("noi-mel-cell", "hidden");
  const analyzePanel = el(
    "div",
    { class: "panel noi-analyze" },
    el("h2", {}, "Analyze"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "wav path"),
      el("input", { id: "noi-a-path", type: "text", placeholder: "/path/to/file.wav", style: "flex:1" }),
      el("button", { class: "btn", id: "noi-a-run" }, "Analyze")
    ),
    el(
      "div",
      { class: "row" },
      el("label", {}, "live capture s"),
      el("input", { id: "noi-a-live", type: "number", value: "3", min: "0.5", max: "30", step: "0.5" }),
      (sourceSel = el("select", { id: "noi-a-source" }, el("option", { value: "mic" }, "MIC"))),
      el("button", { class: "btn", id: "noi-a-capture" }, "Capture")
    ),
    el("div", { class: "dim" }, "needs a trained model — * labels are synthesized RF baseband (BLE/WiFi/Zigbee/LoRa envelopes), not the carrier"),
    el("div", { class: "panel-row" }, melCell),
    (barsHost = el("div", {}, el("div", { class: "dim" }, "no analysis yet")))
  );

  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el(
        "div",
        {},
        el(
          "div",
          { class: "panel" },
          el("h2", {}, "NOISE LAB"),
          (modelStatusEl = el("div", { class: "status-line dim" }, "checking model…")),
          (msgEl = el("div", { class: "msg" })),
          el("div", { id: "noi-class-list" }, el("div", { class: "dim" }, "loading classes…"))
        ),
        trainPanel
      ),
      el("div", {}, analyzePanel)
    )
  );

  trainBtn.addEventListener("click", doTrain);
  cancelBtn.addEventListener("click", doCancel);
  document.getElementById("noi-a-run").addEventListener("click", doAnalyzePath);
  document.getElementById("noi-a-capture").addEventListener("click", doCapture);
  window.addEventListener("themechange", renderBars);

  refreshClasses();
}

export function deactivate() {
  // close the job stream; training keeps running server-side
  if (ws) ws.close();
  ws = null;
  window.removeEventListener("themechange", renderBars);
}
