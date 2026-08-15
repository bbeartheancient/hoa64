// Filter Lab — PCB RF filters (everythingRF digest): prototype synthesis,
// S-parameter sweep, hairpin/stepped/stub KiCad export, Hadamard SA.
//   DESIGN  POST /api/filter/design → metrics + S21/S11 + footprint preview
//   EVOLVE  POST /api/filter/evolve → SA job, IL/RL/REJ + E chart
//   KICAD   POST /api/filter/kicad  → .kicad_mod/.kicad_pcb + preview

import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { themeColor } from "/js/theme.js";
import { drawKicadPrims } from "/js/kicad_layers.js";

const LAYERS = { design: "DESIGN", evolve: "EVOLVE", kicad: "KICAD" };

let msgEl, layerBtns = {}, panels = {};
let layer = "design";
let lastPacked = null;          // last /design or /evolve result
let sCanvas, kCanvas;
let eWs = null, eJob = null, eChart, eCancelBtn, eStatusEl, eStatsEl;

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  n.append(...kids.map((k) => {
    if (k == null || k === false) return "";
    if (typeof k === "string" || k instanceof Node) return k;
    return String(k);
  }));
  return n;
}

function msg(text, kind = "") {
  msgEl.textContent = text;
  msgEl.className = `msg ${kind}`;
}

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { detail = (await r.json()).detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return r.json();
}

function numVal(id, fallback) {
  const v = parseFloat(document.getElementById(id).value);
  return Number.isFinite(v) ? v : fallback;
}

function statRow(k, v, cls = "") {
  let s = String(v);
  if (s.length > 140) s = s.slice(0, 140) + "…";
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, s));
}

function selectLayer(name) {
  layer = name;
  for (const [k, b] of Object.entries(layerBtns)) b.style.opacity = k === layer ? "1" : "0.45";
  for (const [k, p] of Object.entries(panels)) p.classList.toggle("hidden", k !== layer);
  if (name === "design") drawSweep();
  if (name === "kicad") drawPreview();
}

function syncKindForm() {
  const k = document.getElementById("flt-kind")?.value;
  const band = k === "bpf" || k === "bsf";
  document.getElementById("flt-row-fc")?.classList.toggle("hidden", band);
  document.getElementById("flt-row-band")?.classList.toggle("hidden", !band);
  document.getElementById("flt-row-ripple")?.classList.toggle(
    "hidden", document.getElementById("flt-proto")?.value !== "chebyshev"
  );
}

function designBody() {
  const kind = document.getElementById("flt-kind").value;
  const body = {
    kind,
    proto: document.getElementById("flt-proto").value,
    n: parseInt(document.getElementById("flt-n").value, 10),
    eps_r: numVal("flt-epsr", 4.4),
    h_mm: numVal("flt-h", 1.6),
    tan_delta: numVal("flt-tand", 0.02),
    ripple_db: numVal("flt-ripple", 0.1),
    n_sweep: 101,
  };
  if (kind === "bpf" || kind === "bsf") {
    body.f_lo_mhz = numVal("flt-flo", 2300);
    body.f_hi_mhz = numVal("flt-fhi", 2600);
    body.f_c_mhz = 0.5 * (body.f_lo_mhz + body.f_hi_mhz);
  } else {
    body.f_c_mhz = numVal("flt-fc", 2450);
  }
  return body;
}

function renderMetrics(hostId, m, params) {
  const host = document.getElementById(hostId);
  if (!host || !m) return;
  const rows = [
    statRow("IL (pass)", `${Number(m.il_db).toFixed(2)} dB`),
    statRow("IL worst", `${Number(m.il_max_db).toFixed(2)} dB`),
    statRow("RL (pass)", `${Number(m.rl_db).toFixed(1)} dB`, m.rl_db >= 10 ? "good" : ""),
    statRow("rejection", `${Number(m.rejection_db).toFixed(1)} dB`, m.rejection_db >= 40 ? "good" : ""),
    statRow("Q_u", Number(m.q_u).toFixed(0)),
    statRow("IL est", `${Number(m.il_est_db).toFixed(2)} dB  (4.343 Σg / ΔQ_u)`),
  ];
  if (params) {
    rows.push(
      statRow("prototype", `${params.proto} n=${params.n}`),
      statRow("W_50", `${Number(params.w50_mm).toFixed(3)} mm`),
      statRow("RL target", `≥ ${params.return_loss_target_db} dB`),
      statRow("rej target", `${params.rejection_target_dbc} dBc @ ${params.rejection_offset}`),
      statRow("launch", params.launch_note),
      statRow("source", "everythingRF Filter Digest 2025"),
    );
  }
  host.replaceChildren(...rows);
}

function drawSweep() {
  if (!sCanvas) return;
  const ctx = sCanvas.getContext("2d");
  const W = sCanvas.width, H = sCanvas.height;
  ctx.fillStyle = themeColor("bg");
  ctx.fillRect(0, 0, W, H);
  const sw = lastPacked && lastPacked.sweep;
  if (!sw || !sw.f_mhz || !sw.f_mhz.length) {
    ctx.fillStyle = themeColor("dim");
    ctx.font = "11px monospace";
    ctx.fillText("no sweep yet", 8, H / 2);
    return;
  }
  const padL = 40, padR = 10, padT = 12, padB = 24;
  const xs = sw.f_mhz, y21 = sw.s21_db, y11 = sw.s11_db;
  const ymin = Math.min(-60, ...y21, ...y11);
  const ymax = Math.max(5, ...y21, ...y11);
  const X = (i) => padL + (i / (xs.length - 1)) * (W - padL - padR);
  const Y = (v) => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - padT - padB);
  ctx.strokeStyle = themeColor("dim");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  // 0 dB and −10 / −40 guides
  ctx.setLineDash([3, 3]);
  for (const g of [0, -10, -40]) {
    if (g < ymin || g > ymax) continue;
    ctx.beginPath(); ctx.moveTo(padL, Y(g)); ctx.lineTo(W - padR, Y(g)); ctx.stroke();
  }
  ctx.setLineDash([]);
  const stroke = (arr, col) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    arr.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
  };
  stroke(y21, themeColor("fg"));
  stroke(y11, themeColor("accent") || themeColor("dim"));
  ctx.fillStyle = themeColor("dim");
  ctx.font = "10px monospace";
  ctx.fillText(`${xs[0].toFixed(0)} MHz`, padL, H - 6);
  ctx.fillText(`${xs[xs.length - 1].toFixed(0)} MHz`, W - 70, H - 6);
  ctx.fillText("S21", padL + 6, padT + 10);
  ctx.fillStyle = themeColor("accent") || themeColor("dim");
  ctx.fillText("S11", padL + 36, padT + 10);
}

function drawPreview(preview) {
  if (!kCanvas) return;
  drawKicadPrims(kCanvas, preview || (lastPacked && lastPacked.preview));
}

function showPacked(d, metricsHost) {
  lastPacked = d;
  renderMetrics(metricsHost, d.metrics, d.params);
  drawSweep();
  drawPreview(d.preview);
}

async function doDesign() {
  msg("synthesising filter…");
  try {
    const d = await api("/api/filter/design", designBody());
    showPacked(d, "flt-d-metrics");
    msg(`${d.design.kind} n=${d.design.n}  IL ${d.metrics.il_db.toFixed(2)} dB  RL ${d.metrics.rl_db.toFixed(1)} dB`, "ok");
  } catch (e) {
    msg(`design failed: ${e.message}`, "error");
  }
}

async function doKicad() {
  const btn = document.getElementById("flt-k-run");
  btn.disabled = true;
  try {
    const body = designBody();
    if (lastPacked && lastPacked.design) body.design = lastPacked.design;
    const d = await api("/api/filter/kicad", body);
    lastPacked = { ...(lastPacked || {}), preview: d.preview, params: d.params, metrics: d.metrics };
    renderMetrics("flt-k-metrics", d.metrics, d.params);
    drawPreview(d.preview);
    const host = document.getElementById("flt-k-files");
    host.replaceChildren(
      ...(d.files || []).map((name) =>
        el("div", { class: "row" },
          el("a", { class: "btn btn-xs", href: `/api/filter/kicad/${d.token}/${name}`, download: name }, name))
      ),
      el("div", { class: "dim" }, "one-shot links — click to download")
    );
    msg(`kicad export: ${(d.files || []).join(", ")}`, "ok");
  } catch (e) {
    msg(`kicad failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

function handleEvolveFrame(d) {
  if (d.type === "snapshot") {
    for (const m of d.history) handleEvolveFrame(m);
    return;
  }
  if (d.type === "progress") {
    eChart.push({ E: d.E, BEST_E: d.best_E, T: d.T });
    if (d.preview) drawPreview(d.preview);
    const bits = [];
    if (d.step !== undefined) bits.push(`step ${d.step}`);
    if (d.il_db !== undefined) bits.push(`IL ${Number(d.il_db).toFixed(2)} dB`);
    if (d.rl_db !== undefined) bits.push(`RL ${Number(d.rl_db).toFixed(1)} dB`);
    if (d.rejection_db !== undefined) bits.push(`rej ${Number(d.rejection_db).toFixed(1)} dB`);
    eStatusEl.textContent = bits.join(" · ") || "running";
    return;
  }
  if (d.type === "end") {
    eStatusEl.textContent = `job ${d.status}`;
    eCancelBtn.disabled = true;
    finishEvolve();
  }
  if (d.type === "error") msg(d.error, "error");
}

async function finishEvolve() {
  if (!eJob) return;
  try {
    const d = await api(`/api/search/${eJob}`);
    const r = d.result || {};
    showPacked(r, "flt-e-metrics");
    const rows = ["steps", "accepts", "best_E", "elapsed_s"].map((k) =>
      r[k] !== undefined ? statRow(k, typeof r[k] === "number" ? Number(r[k].toPrecision(4)) : r[k]) : null
    ).filter(Boolean);
    eStatsEl.replaceChildren(...rows);
    msg("filter evolution complete", "ok");
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doEvolve() {
  const body = { ...designBody(), hadamard_order: parseInt(document.getElementById("flt-e-order").value, 10),
    max_steps: parseInt(document.getElementById("flt-e-steps").value, 10), budget_s: 60 };
  if (eWs) eWs.close();
  eChart.clear();
  eStatsEl.replaceChildren();
  eStatusEl.textContent = "connecting…";
  msg("starting filter evolution…");
  try {
    const { job_id } = await api("/api/filter/evolve", body);
    eJob = job_id;
    eCancelBtn.disabled = false;
    msg(`job ${job_id} running`, "ok");
    eWs = connect(`/ws/job/${job_id}`, { message: handleEvolveFrame });
  } catch (e) {
    msg(`evolve failed: ${e.message}`, "error");
  }
}

async function doCancelEvolve() {
  if (!eJob) return;
  try {
    if (eWs) eWs.send({ op: "cancel" });
    await api(`/api/search/${eJob}/cancel`, {});
    msg(`cancel requested for ${eJob}`);
  } catch (e) {
    msg(`cancel failed: ${e.message}`, "error");
  }
}

function onThemechange() {
  drawSweep();
  drawPreview();
}

export function init(container) {
  layerBtns = {};
  const layerRow = el(
    "div", { class: "btn-row layer-select", id: "flt-layer-select" },
    ...Object.entries(LAYERS).map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name }, `[${label}]`);
      b.addEventListener("click", () => selectLayer(name));
      layerBtns[name] = b;
      return b;
    })
  );

  // DESIGN owns the shared form ids; EVOLVE/KICAD read them via designBody().
  sCanvas = el("canvas", { class: "sim-canvas kicad-canvas", width: "520", height: "280" });
  kCanvas = el("canvas", { class: "sim-canvas kicad-canvas", width: "384", height: "256" });

  panels.design = el(
    "div", {},
    el("div", { class: "panel" },
      el("h2", {}, "Prototype synthesis"),
      el("div", { class: "row" }, el("label", {}, "kind"),
        el("select", { id: "flt-kind" },
          el("option", { value: "lpf" }, "LOW-PASS (stepped-Z)"),
          el("option", { value: "hpf" }, "HIGH-PASS (gap + stub)"),
          el("option", { value: "bpf" }, "BAND-PASS (hairpin)"),
          el("option", { value: "bsf" }, "BAND-STOP (open stub)"))),
      el("div", { class: "row" }, el("label", {}, "prototype"),
        el("select", { id: "flt-proto" },
          el("option", { value: "butterworth" }, "BUTTERWORTH"),
          el("option", { value: "chebyshev" }, "CHEBYSHEV"))),
      el("div", { class: "row" }, el("label", {}, "order N"), el("input", { id: "flt-n", type: "number", value: "5", min: "1", max: "11", step: "1" })),
      el("div", { class: "row", id: "flt-row-fc" }, el("label", {}, "f_c MHz"), el("input", { id: "flt-fc", type: "number", value: "2450", step: "1" })),
      el("div", { class: "row hidden", id: "flt-row-band" },
        el("label", {}, "f_lo / f_hi"),
        el("input", { id: "flt-flo", type: "number", value: "2300", step: "1" }),
        el("input", { id: "flt-fhi", type: "number", value: "2600", step: "1" })),
      el("div", { class: "row hidden", id: "flt-row-ripple" }, el("label", {}, "ripple dB"), el("input", { id: "flt-ripple", type: "number", value: "0.1", step: "0.05", min: "0.01" })),
      el("div", { class: "row" }, el("label", {}, "εr / h mm"), el("input", { id: "flt-epsr", type: "number", value: "4.4", step: "0.1" }), el("input", { id: "flt-h", type: "number", value: "1.6", step: "0.1" })),
      el("div", { class: "row" }, el("label", {}, "tan δ"), el("input", { id: "flt-tand", type: "number", value: "0.02", step: "0.005" })),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "flt-d-run" }, "Synthesize")),
      el("div", { class: "dim" }, "Butterworth −20N dB/decade · RL ≥ 10 dB · 40 dBc @ 10 % from the edge")
    ),
    el("div", { class: "panel" },
      el("h2", {}, "S-parameters"),
      el("div", { class: "panel-row" }, el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "S21 / S11 (dB)"), sCanvas))
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Metrics (everythingRF digest)"),
      el("table", { class: "stats" }, el("tbody", { id: "flt-d-metrics" },
        el("tr", {}, el("td", { class: "dim" }, "synthesize to fill IL / RL / Q_u"))))
    )
  );

  const eChartCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  eChart = makeStripChart(eChartCanvas, null);
  panels.evolve = el(
    "div", {},
    el("div", { class: "panel" },
      el("h2", {}, "Hadamard SA (section lengths / widths)"),
      el("div", { class: "row" }, el("label", {}, "hadamard order"), el("input", { id: "flt-e-order", type: "number", value: "32", min: "4", step: "4" })),
      el("div", { class: "row" }, el("label", {}, "max steps"), el("input", { id: "flt-e-steps", type: "number", value: "200", min: "20", step: "20" })),
      el("div", { class: "btn-row" },
        el("button", { class: "btn", id: "flt-e-run" }, "Evolve"),
        (eCancelBtn = el("button", { class: "btn", disabled: true }, "Cancel")),
        el("button", { class: "btn", id: "flt-e-kicad" }, "Export KiCad")),
      el("div", { class: "dim" }, "uses the DESIGN form (kind / N / f). Energy = IL + RL≥10 + 40 dBc + DFM")
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Waveforms"),
      eChartCanvas
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Status"),
      (eStatusEl = el("div", { class: "status-line" }, "idle")),
      el("table", { class: "stats" }, (eStatsEl = el("tbody"))),
      el("table", { class: "stats" }, el("tbody", { id: "flt-e-metrics" }))
    )
  );

  panels.kicad = el(
    "div", {},
    el("div", { class: "panel" },
      el("h2", {}, "KiCad / PCB export"),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "flt-k-run" }, "Generate")),
      el("div", { class: "dim" }, "last synthesized (or evolved) design, else the DESIGN form")
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Preview"),
      el("div", { class: "panel-row" }, el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "KiCad copper"), kCanvas)),
      el("div", { class: "dim" }, "red F.Cu · blue B.Cu · (green In1 / orange In2 reserved)")
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Parameters"),
      el("table", { class: "stats" }, el("tbody", { id: "flt-k-metrics" }))
    ),
    el("div", { class: "panel" },
      el("h2", {}, "Files"),
      el("div", { id: "flt-k-files" }, el("div", { class: "dim" }, "no export yet"))
    )
  );

  const head = el("div", { class: "panel" },
    el("h2", {}, "FILTER LAB"),
    layerRow,
    (msgEl = el("div", { class: "msg" }))
  );
  container.replaceChildren(el("div", { class: "lab ant-cap" }, el("div", {}, head), el("div", {}, ...Object.values(panels))));
  selectLayer("design");

  document.getElementById("flt-kind").addEventListener("change", syncKindForm);
  document.getElementById("flt-proto").addEventListener("change", syncKindForm);
  document.getElementById("flt-d-run").addEventListener("click", doDesign);
  document.getElementById("flt-k-run").addEventListener("click", doKicad);
  document.getElementById("flt-e-run").addEventListener("click", doEvolve);
  document.getElementById("flt-e-kicad").addEventListener("click", () => { selectLayer("kicad"); doKicad(); });
  eCancelBtn.addEventListener("click", doCancelEvolve);
  syncKindForm();
  drawSweep();
  drawPreview();
  window.addEventListener("themechange", onThemechange);
}

export function deactivate() {
  if (eWs) eWs.close();
  eWs = null;
  window.removeEventListener("themechange", onThemechange);
}
