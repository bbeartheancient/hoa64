// Antenna Lab — antenna design / parts matching / FDTD field sim / wire
// evolution, all behind ONE viewport with a [DESIGN][PARTS][FIELDS][EVOLVE]
// layer selector (#ant-layer-select, same convention as #sim-layer-select).
//   DESIGN  POST /api/antenna/design → ranked candidate table (click a row
//           for its reasons/explain trace; patch/pifa/loop rows get a
//           [KICAD] footprint export). A successful run auto-fires the parts
//           query for the band.
//   PARTS   POST /api/antenna/parts → off-the-shelf part table with
//           datasheet / everythingRF links; coverage_note rows dimmed.
//   FIELDS  POST /api/antenna/fields → {job_id}, streams /ws/job/{id}
//           frames {step,t_s,e_xy_png_b64,e_xz_png_b64,ar_png_b64?,emax,
//           e_rms,e_rms_lo?,e_rms_hi?} onto two (three) heatmap canvases +
//           an E_RMS strip chart. [CANCEL] → ws {op:"cancel"} + POST cancel.
//   EVOLVE  POST /api/antenna/evolve → {job_id}, streams
//           {step,T,E,best_E,accepts,points,z_in,gain_dbi,s11_db}; the best
//           wire geometry polyline is drawn on a 2D canvas (x,y projection,
//           fg on bg, axes cross in dim), E/BEST_E/T ride a strip chart; the
//           final frame's pattern_png_b64 is shown retinted beside it.
//   SURVEY  POST /api/antenna/survey → terrain elevation profile along the
//           TX→RX great-circle path (filled dim polyline), LOS sight line
//           (fg), mast ticks, worst-clearance point (accent) on a 640×200
//           chart canvas + a link-budget stats table (verdict / clearance /
//           Fresnel ratio / diffraction loss / FSPL / received power); then
//           POST /api/antenna/survey/map → retinted terrain heatmap with a
//           mercator-projected path/markers overlay redrawn on themechange.
// Server PNG canvases go through retintCanvas (registers them for
// themechange re-tints); the wire canvas redraws on themechange itself.
// Panels are kept attached and toggled with .hidden so a running job's
// stream keeps updating its canvases while another layer is up (the strip
// chart drops its themechange listener on detached canvases).

import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { retintCanvas, themeColor } from "/js/theme.js";

const LAYERS = { design: "DESIGN", parts: "PARTS", fields: "FIELDS", evolve: "EVOLVE", kicad: "KICAD", smith: "SMITH", survey: "SURVEY" };
// entry.type → kicad design_type (first substring match wins)
const KICAD_MAP = [
  ["pifa", "meander_ifa"],
  ["meander", "meander_ifa"],
  ["ifa", "meander_ifa"],
  ["patch", "patch"],
  ["loop", "loop"],
];

let msgEl, layerBtns = {}, panels = {};
let layer = "design";
let lastBand = null; // {f_lo_mhz, f_hi_mhz, f_center_mhz} from the last design run

// fields job state
let fWs = null, fJob = null, fChart, fCancelBtn, fStatusEl, fStatsEl;
let fXyCanvas, fXzCanvas, fArCanvas, fArCell;
// evolve job state
let eWs = null, eJob = null, eChart, eCancelBtn, eStatusEl, eStatsEl;
let smithCanvas, smithReadout, lastSweep = null;
let svProfileCanvas, svMapCanvas, svStatsEl, svStatusEl;
let lastSurvey = null, lastMap = null; // redrawn on themechange
let lastEvolvePoints = null, lastDesignEntries = null;
let wireCanvas, patternCanvas, patternCell;
let lastPoints = null; // last evolve points — redrawn on themechange

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

function statRow(k, v, cls = "") {
  let s = String(v);
  if (s.length > 120) s = s.slice(0, 120) + "…"; // never dump blobs into the table
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, s));
}

// job-result tables: whitelist display keys (long strings/base64 payloads like
// resonance_note / pattern_png_b64 / points stay OUT of the DOM — they blew up
// the table width and pushed the fixed settings drawer into view). The full
// result is downloadable as JSON instead.
function resultStatRows(r, keys) {
  const rows = [];
  for (const k of keys) {
    if (r[k] === undefined || r[k] === null) continue;
    const v = r[k];
    if (typeof v !== "number" && typeof v !== "string" && typeof v !== "boolean") continue;
    rows.push(statRow(k, typeof v === "number" ? Number(v.toPrecision(4)) : v));
  }
  if (r.z_in && typeof r.z_in === "object") {
    const im = Number(r.z_in.im);
    rows.push(statRow("z_in", `${Number(r.z_in.re).toFixed(1)}${im < 0 ? "−" : "+"}j${Math.abs(im).toFixed(1)} Ω`));
  }
  if (r.terms && typeof r.terms === "object") {
    for (const [k, v] of Object.entries(r.terms)) {
      if (typeof v === "number") rows.push(statRow(k, Number(v.toPrecision(4))));
    }
  }
  return rows;
}

function resultDownloadRow(jobId, r) {
  const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = el("a", { class: "btn btn-xs", href: url, download: `antenna-${jobId}-result.json` }, "RESULT JSON");
  return el("tr", {}, el("td", { class: "k" }, "full result"), el("td", {}, a));
}

const FIELD_STAT_KEYS = ["steps_run", "stopped", "dt_s", "dx_m", "cells_per_lambda", "alpha_theory", "decay_measured", "resolution_warning"];
const EVOLVE_STAT_KEYS = ["steps", "accepts", "best_E", "elapsed_s", "gain_dbi", "s11_db", "seed_row"];

function fmtMm(m) {
  const mm = m * 1000;
  return `${mm >= 100 ? mm.toFixed(0) : mm.toFixed(1)}`;
}

function fmtDims(dims) {
  if (!dims) return "—";
  return Object.entries(dims)
    .map(([k, v]) => `${k} ${fmtMm(v)}`)
    .join("  ");
}

// ---- layer selector ---------------------------------------------------------

function selectLayer(name) {
  layer = name;
  for (const [k, b] of Object.entries(layerBtns)) {
    b.style.opacity = k === layer ? "1" : "0.45";
  }
  for (const [k, p] of Object.entries(panels)) {
    p.classList.toggle("hidden", k !== layer);
  }
}

// ---- DESIGN -----------------------------------------------------------------

function designBody() {
  const site = {};
  const sizeRaw = document.getElementById("ant-d-size").value;
  if (sizeRaw !== "") site.max_size_m = parseFloat(sizeRaw) / 100; // cm → m
  if (document.getElementById("ant-d-ground").checked) site.ground_plane = true;
  const rangeRaw = document.getElementById("ant-d-range").value;
  if (rangeRaw !== "") site.range_m = parseFloat(rangeRaw);
  const mounting = document.getElementById("ant-d-mounting").value;
  if (mounting !== "free") site.mounting = mounting;
  const pol = document.getElementById("ant-d-pol").value;
  if (pol !== "any") site.polarization = pol;
  return {
    f_lo_mhz: numVal("ant-d-flo", 2400),
    f_hi_mhz: numVal("ant-d-fhi", 2485),
    medium: document.getElementById("ant-d-medium").value,
    site,
  };
}

function kicadType(entryType) {
  const t = String(entryType).toLowerCase();
  for (const [sub, ktype] of KICAD_MAP) if (t.includes(sub)) return ktype;
  return null;
}

async function doKicad(ktype, btn, fMhz, opts, resultsHost) {
  btn.disabled = true;
  try {
    const body = { design_type: ktype, f_mhz: fMhz };
    if (opts && Object.keys(opts).length) body.opts = opts;
    const d = await api("/api/antenna/kicad", body);
    if (resultsHost) {
      // one-shot download links — each URL self-destructs after one GET
      resultsHost.replaceChildren(
        ...(d.files || []).map((name) =>
          el(
            "div",
            { class: "row" },
            el("a", { class: "btn btn-xs", href: `/api/antenna/kicad/${d.token}/${name}`, download: name }, name)
          )
        ),
        el("div", { class: "dim" }, "one-shot links — click to download")
      );
    } else {
      // auto-download per generated file
      for (const name of d.files || []) {
        const a = el("a", { href: `/api/antenna/kicad/${d.token}/${name}`, download: name });
        document.body.append(a);
        a.click();
        a.remove();
      }
    }
    msg(`kicad export: ${(d.files || []).join(", ") || "(no files)"}`, "ok");
  } catch (e) {
    msg(`kicad failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

async function doKicadPanel() {
  const ktype = document.getElementById("ant-k-type").value;
  const opts = {};
  if (ktype !== "loop") {
    opts.eps_r = numVal("ant-k-epsr", 4.4);
    opts.h_mm = numVal("ant-k-h", 1.6);
  }
  if (ktype === "patch") opts.feed = document.getElementById("ant-k-feed").value;
  if (ktype === "loop") opts.medium = document.getElementById("ant-k-medium").value;
  const btn = document.getElementById("ant-k-run");
  await doKicad(ktype, btn, numVal("ant-k-f", 2450), opts, document.getElementById("ant-k-results"));
}

function renderDesign(d) {
  const host = document.getElementById("ant-d-results");
  const head = el(
    "tr",
    {},
    ...["TYPE", "SCORE", "GAIN dBi", "BW %", "POL", "DIMS mm", "VIABLE"].map((h) => el("th", {}, h))
  );
  const rows = [head];
  for (const e of d.entries || []) {
    const des = e.design || {};
    const detail = el(
      "tr",
      { class: "hidden" },
      el(
        "td",
        { colspan: "7" },
        ...(e.reasons || []).map((r) => el("div", { class: "dim" }, `· ${r}`)),
        e.explain ? el("div", { class: "msg", style: "margin-top:4px" }, e.explain) : "",
        ...(des.notes ? [el("div", { class: "dim" }, des.notes)] : [])
      )
    );
    const ktype = kicadType(e.type);
    if (ktype) {
      const kb = el("button", { class: "btn btn-xs" }, "KICAD");
      kb.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (lastBand) doKicad(ktype, kb, lastBand.f_center_mhz);
      });
      detail.firstChild.append(el("div", { class: "btn-row", style: "margin-top:6px" }, kb));
    }
    const tr = el(
      "tr",
      { style: "cursor:pointer" },
      el("td", { class: "k" }, e.type),
      el("td", {}, e.score !== undefined ? Number(e.score).toFixed(2) : "?"),
      el("td", {}, des.gain_dbi !== undefined ? Number(des.gain_dbi).toFixed(1) : "?"),
      el("td", {}, des.bandwidth_frac !== undefined ? (des.bandwidth_frac * 100).toFixed(1) : "?"),
      el("td", {}, des.polarization || "?"),
      el("td", { class: "dim" }, fmtDims(des.dimensions_m)),
      el("td", { class: e.viable ? "good" : "bad" }, e.viable ? "YES" : "NO")
    );
    tr.addEventListener("click", () => detail.classList.toggle("hidden"));
    rows.push(tr, detail);
  }
  host.replaceChildren(el("table", { class: "stats" }, ...rows));
}

async function doDesign() {
  msg("scoring antenna types…");
  try {
    const d = await api("/api/antenna/design", designBody());
    lastDesignEntries = d.entries || [];
    lastBand = {
      f_lo_mhz: numVal("ant-d-flo", 2400),
      f_hi_mhz: numVal("ant-d-fhi", 2485),
      f_center_mhz: d.f_center_mhz,
    };
    renderDesign(d);
    const n = (d.entries || []).filter((e) => e.viable).length;
    msg(
      `${(d.entries || []).length} types scored · ${n} viable · ` +
        `f_center ${d.f_center_mhz} MHz · required BW ${(d.required_bw_frac * 100).toFixed(1)}%  → see PARTS tab`,
      "ok"
    );
    // keep the parts band in sync and pre-run the parts query
    document.getElementById("ant-p-flo").value = String(lastBand.f_lo_mhz);
    document.getElementById("ant-p-fhi").value = String(lastBand.f_hi_mhz);
    // prefill the SURVEY link params from this design run
    const svF = document.getElementById("ant-v-f");
    if (svF && d.f_center_mhz) svF.value = String(d.f_center_mhz);
    const g0 = lastDesignEntries[0]?.design?.gain_dbi;
    const svG = document.getElementById("ant-v-gtx");
    if (svG && g0 !== undefined) svG.value = String(g0);
    doParts(true);
  } catch (e) {
    msg(`design failed: ${e.message}`, "error");
  }
}

// ---- PARTS ------------------------------------------------------------------

async function doParts(quiet = false) {
  const body = {
    f_lo_mhz: numVal("ant-p-flo", lastBand ? lastBand.f_lo_mhz : 2400),
    f_hi_mhz: numVal("ant-p-fhi", lastBand ? lastBand.f_hi_mhz : 2485),
  };
  const gainRaw = document.getElementById("ant-p-gain").value;
  if (gainRaw !== "") body.gain_dbi_min = parseFloat(gainRaw);
  if (!quiet) msg("matching catalog parts…");
  try {
    const d = await api("/api/antenna/parts", body);
    const head = el(
      "tr",
      {},
      ...["PART", "MFR", "TYPE", "BAND MHz", "GAIN", "VSWR", "SIZE mm", "MOUNT", "LINKS"].map((h) =>
        el("th", {}, h)
      )
    );
    const rows = (d.matches || []).map((m) =>
      el(
        "tr",
        m.coverage_note ? { style: "opacity:0.55", title: m.coverage_note } : {},
        el("td", { class: "k" }, m.part),
        el("td", {}, m.mfr || "?"),
        el("td", { class: "dim" }, m.type || "?"),
        el("td", {}, `${m.freq_lo_mhz}–${m.freq_hi_mhz}`),
        el("td", {}, m.gain_dbi !== undefined ? `${m.gain_dbi} dBi` : "?"),
        el("td", {}, m.vswr !== undefined ? String(m.vswr) : "?"),
        el("td", { class: "dim" }, m.size_mm || "?"),
        el("td", {}, m.mount || "?"),
        el(
          "td",
          {},
          m.datasheet_url
            ? el("a", { class: "btn btn-xs", href: m.datasheet_url, target: "_blank", rel: "noopener" }, "DATASHEET")
            : "",
          " ",
          m.erf_url
            ? el("a", { class: "btn btn-xs", href: m.erf_url, target: "_blank", rel: "noopener" }, "EVERYTHINGRF")
            : ""
        )
      )
    );
    document
      .getElementById("ant-p-results")
      .replaceChildren(el("table", { class: "stats" }, head, ...rows));
    if (!quiet || !(d.matches || []).length)
      msg(`${(d.matches || []).length} parts matched ${body.f_lo_mhz}–${body.f_hi_mhz} MHz`, "ok");
  } catch (e) {
    if (!quiet) msg(`parts failed: ${e.message}`, "error");
  }
}

// ---- FIELDS -----------------------------------------------------------------

function handleFieldFrame(d) {
  if (d.type === "snapshot") {
    fStatusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleFieldFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.e_xy_png_b64) drawPng(fXyCanvas, d.e_xy_png_b64);
    if (d.e_xz_png_b64) drawPng(fXzCanvas, d.e_xz_png_b64);
    if (d.ar_png_b64) {
      fArCell.classList.remove("hidden");
      drawPng(fArCanvas, d.ar_png_b64);
    }
    if (d.e_rms !== undefined) {
      fChart.push({ E_RMS: d.e_rms, E_RMS_LO: d.e_rms_lo, E_RMS_HI: d.e_rms_hi });
    }
    const bits = [];
    if (d.step !== undefined) bits.push(`step ${d.step}`);
    if (d.t_s !== undefined) bits.push(`t ${Number(d.t_s).toExponential(2)}s`);
    if (d.emax !== undefined) bits.push(`Emax ${Number(d.emax).toExponential(2)}`);
    if (d.e_rms !== undefined) bits.push(`rms ${Number(d.e_rms).toExponential(2)}`);
    fStatusEl.textContent = bits.join(" · ") || "running";
    return;
  }
  if (d.type === "end") {
    fStatusEl.textContent = `job ${d.status}`;
    fCancelBtn.disabled = true;
    finishFields();
    return;
  }
  if (d.type === "error") msg(d.error, "error");
}

async function finishFields() {
  if (!fJob) return;
  try {
    const d = await api(`/api/search/${fJob}`);
    const r = d.result || {};
    fStatsEl.replaceChildren(...resultStatRows(r, FIELD_STAT_KEYS), resultDownloadRow(fJob, r));
    msg("field run complete", "ok");
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doFields() {
  const body = {
    f_mhz: numVal("ant-f-f", 150),
    medium: document.getElementById("ant-f-medium").value,
    interface: document.getElementById("ant-f-interface").checked,
    n: parseInt(document.getElementById("ant-f-n").value, 10),
    max_steps: parseInt(document.getElementById("ant-f-steps").value, 10),
    frame_every: 25,
    pol_viz: true,
    budget_s: 120,
  };
  if (fWs) fWs.close();
  fChart.clear();
  fStatsEl.replaceChildren();
  fArCell.classList.add("hidden");
  fStatusEl.textContent = "connecting…";
  msg("starting FDTD run…");
  try {
    const { job_id } = await api("/api/antenna/fields", body);
    fJob = job_id;
    fCancelBtn.disabled = false;
    msg(`job ${job_id} running`, "ok");
    fWs = connect(`/ws/job/${job_id}`, { message: handleFieldFrame });
  } catch (e) {
    msg(`fields failed: ${e.message}`, "error");
  }
}

async function doCancelFields() {
  if (!fJob) return;
  try {
    if (fWs) fWs.send({ op: "cancel" });
    await api(`/api/search/${fJob}/cancel`, {});
    msg(`cancel requested for ${fJob}`);
  } catch (e) {
    msg(`cancel failed: ${e.message}`, "error");
  }
}

// ---- EVOLVE -----------------------------------------------------------------

function drawWire() {
  if (!wireCanvas) return;
  const ctx = wireCanvas.getContext("2d");
  const w = wireCanvas.width;
  const h = wireCanvas.height;
  ctx.fillStyle = themeColor("bg");
  ctx.fillRect(0, 0, w, h);
  // axes cross (world origin) in dim, after fitting bounds
  const pts = lastPoints;
  let x0 = 0, y0 = 0, scale = 1;
  if (pts && pts.length) {
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const p of pts) {
      for (let a = 0; a < 2; a++) {
        if (p[a] < lo[a]) lo[a] = p[a];
        if (p[a] > hi[a]) hi[a] = p[a];
      }
    }
    for (let a = 0; a < 2; a++) {
      if (0 < lo[a]) lo[a] = 0;
      if (0 > hi[a]) hi[a] = 0;
    }
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-9);
    scale = (Math.min(w, h) - 24) / span;
    x0 = 12 - lo[0] * scale + (w - Math.min(w, h)) / 2;
    y0 = 12 - lo[1] * scale + (h - Math.min(w, h)) / 2;
  }
  ctx.strokeStyle = themeColor("dim");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - y0);
  ctx.lineTo(w, h - y0);
  ctx.moveTo(x0, 0);
  ctx.lineTo(x0, h);
  ctx.stroke();
  if (!pts || pts.length < 2) {
    ctx.fillStyle = themeColor("dim");
    ctx.font = "11px monospace";
    ctx.fillText("no geometry yet", 8, h / 2);
    return;
  }
  ctx.strokeStyle = themeColor("fg");
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = x0 + p[0] * scale;
    const y = h - (y0 + p[1] * scale);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function onThemechange() {
  drawWire();
  drawSmith();
  drawSurveyProfile();
  drawSurveyMapOverlay(); // the map PNG itself re-tints via retintCanvas
}

// ---- SMITH ------------------------------------------------------------------
// Γ-plane rendering: unit circle + constant-r circles + constant-x arcs, all
// derived from Γ = (z−1)/(z+1) on normalized z = Z/Z₀. The sweep trace comes
// from /api/antenna/smith (MoM Z_in per frequency), design markers from the
// last recommender run's z_in_ohm (serialized as Python complex strings).

function parseComplex(s) {
  // Python str(complex): "(73.1+42.5j)" / "70+0j" / "-3-4j"
  const m = String(s).replace(/[()j\s]/g, "").match(/^([+-]?[\d.eE+-]*?)([+-][\d.eE]+)?$/);
  if (!m) return null;
  const re = parseFloat(m[1] || "0");
  const im = m[2] ? parseFloat(m[2]) : 0;
  return Number.isFinite(re) && Number.isFinite(im) ? { re, im } : null;
}

function gammaOf(z, z0) {
  const den = { re: z.re + z0, im: z.im };
  const d = den.re * den.re + den.im * den.im || 1e-30;
  return {
    re: ((z.re - z0) * den.re + z.im * den.im) / d,
    im: (z.im * den.re - (z.re - z0) * den.im) / d,
  };
}

function drawSmith(hoverPt) {
  if (!smithCanvas) return;
  const ctx = smithCanvas.getContext("2d");
  const W = smithCanvas.width, H = smithCanvas.height;
  const fg = themeColor("fg"), bg = themeColor("bg"), dim = themeColor("dim");
  const accent = themeColor("accent") || fg;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 26;
  const P = (g) => [cx + g.re * R, cy - g.im * R];
  const circle = (x, y, r) => { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke(); };
  // grid (dim): unit circle, constant-r circles, constant-x arcs, real axis
  ctx.strokeStyle = dim;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.75;
  circle(cx, cy, R);
  for (const r of [0.2, 0.5, 1, 2, 5]) circle(cx + (r / (r + 1)) * R, cy, R / (r + 1));
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.clip();
  for (const x of [0.5, 1, 2, 5]) for (const s of [1, -1]) circle(cx + R, cy - (s * R) / x, R / x);
  ctx.beginPath();
  ctx.moveTo(cx - R, cy);
  ctx.lineTo(cx + R, cy);
  ctx.stroke();
  ctx.restore();
  ctx.globalAlpha = 1;
  // sweep trace (fg)
  const pts = lastSweep ? lastSweep.sweep.filter((p) => p.gamma) : [];
  if (pts.length) {
    ctx.strokeStyle = fg;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = P(p.gamma);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = fg;
    const dot = (p, r) => { const [x, y] = P(p.gamma); ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); };
    dot(pts[0], 4);                    // f_lo
    dot(pts[pts.length - 1], 4);       // f_hi
    dot(pts[Math.floor(pts.length / 2)], 3); // f_center
  }
  // design markers (accent squares) — Z_in of each ranked entry at f_center
  if (lastDesignEntries && document.getElementById("ant-s-designs")?.checked) {
    const z0 = lastSweep ? lastSweep.z0 : numVal("ant-s-z0", 50);
    ctx.fillStyle = accent;
    for (const e of lastDesignEntries) {
      const z = parseComplex(e.design?.z_in_ohm);
      if (!z) continue;
      const [x, y] = P(gammaOf(z, z0));
      ctx.fillRect(x - 3, y - 3, 6, 6);
    }
  }
  // hover marker
  if (hoverPt) {
    const [x, y] = P(hoverPt.gamma);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    circle(x, y, 7);
  }
}

async function doSmith() {
  const source = document.getElementById("ant-s-src").value;
  if (source === "wire" && !lastEvolvePoints) {
    msg("no evolved geometry yet — run EVOLVE first", "error");
    return;
  }
  msg("sweeping Z_in(f) via MoM…");
  try {
    const body = {
      f_lo_mhz: numVal("ant-s-flo", 2400),
      f_hi_mhz: numVal("ant-s-fhi", 2485),
      n_points: parseInt(document.getElementById("ant-s-n").value, 10),
      z0: numVal("ant-s-z0", 50),
      medium: document.getElementById("ant-s-medium").value,
      source,
    };
    if (source === "wire") body.points = lastEvolvePoints;
    lastSweep = await api("/api/antenna/smith", body);
    drawSmith();
    const ok = lastSweep.sweep.filter((p) => p.gamma).length;
    msg(`smith sweep: ${ok}/${lastSweep.sweep.length} points solved`, "ok");
  } catch (e) {
    msg(`smith failed: ${e.message}`, "error");
  }
}

function onSmithHover(ev) {
  if (!lastSweep) return;
  const rect = smithCanvas.getBoundingClientRect();
  const mx = ((ev.clientX - rect.left) / rect.width) * smithCanvas.width;
  const my = ((ev.clientY - rect.top) / rect.height) * smithCanvas.height;
  const W = smithCanvas.width, H = smithCanvas.height;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 26;
  let best = null, bestD = Infinity;
  for (const p of lastSweep.sweep) {
    if (!p.gamma) continue;
    const dx = cx + p.gamma.re * R - mx, dy = cy - p.gamma.im * R - my;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = p; }
  }
  if (best && bestD < (smithCanvas.width * 0.08) ** 2) {
    const g = best.gamma;
    const mag = Math.hypot(g.re, g.im);
    const ang = (Math.atan2(g.im, g.re) * 180) / Math.PI;
    smithReadout.textContent =
      `f ${best.f_mhz.toFixed(1)} MHz · Z ${best.z.re.toFixed(1)}${best.z.im < 0 ? "−" : "+"}j${Math.abs(best.z.im).toFixed(1)} Ω` +
      ` · Γ ${mag.toFixed(3)}∠${ang.toFixed(0)}° · S11 ${best.s11_db.toFixed(1)} dB`;
    drawSmith(best);
  } else {
    smithReadout.textContent = "hover the trace for f / Z / Γ / S11";
    drawSmith();
  }
}

function handleEvolveFrame(d) {
  if (d.type === "snapshot") {
    eStatusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleEvolveFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (Array.isArray(d.points)) {
      lastPoints = d.points;
      drawWire();
    }
    eChart.push({ E: d.E, BEST_E: d.best_E, T: d.T });
    if (d.pattern_png_b64) {
      patternCell.classList.remove("hidden");
      drawPng(patternCanvas, d.pattern_png_b64);
    }
    const bits = [];
    if (d.step !== undefined) bits.push(`step ${d.step}`);
    if (d.z_in) {
      const im = Number(d.z_in.im);
      bits.push(`Z_in ${Number(d.z_in.re).toFixed(1)}${im < 0 ? "−" : "+"}j${Math.abs(im).toFixed(1)} Ω`);
    }
    if (d.gain_dbi !== undefined) bits.push(`gain ${Number(d.gain_dbi).toFixed(2)} dBi`);
    if (d.s11_db !== undefined) bits.push(`S11 ${Number(d.s11_db).toFixed(1)} dB`);
    eStatusEl.textContent = bits.join(" · ") || "running";
    return;
  }
  if (d.type === "end") {
    eStatusEl.textContent = `job ${d.status}`;
    eCancelBtn.disabled = true;
    finishEvolve();
    return;
  }
  if (d.type === "error") msg(d.error, "error");
}

async function finishEvolve() {
  if (!eJob) return;
  try {
    const d = await api(`/api/search/${eJob}`);
    const r = d.result || {};
    if (Array.isArray(r.points)) lastEvolvePoints = r.points;
    eStatsEl.replaceChildren(...resultStatRows(r, EVOLVE_STAT_KEYS), resultDownloadRow(eJob, r));
    msg("evolution complete", "ok");
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doEvolve() {
  const body = {
    f_mhz: numVal("ant-e-f", 2450),
    medium: document.getElementById("ant-e-medium").value,
    topology: "meander",
    hadamard_order: parseInt(document.getElementById("ant-e-order").value, 10),
    max_steps: parseInt(document.getElementById("ant-e-steps").value, 10),
    T_start: 1.0,
    T_end: 0.01,
    cooling: 0.995,
    budget_s: 120,
  };
  if (eWs) eWs.close();
  eChart.clear();
  eStatsEl.replaceChildren();
  patternCell.classList.add("hidden");
  lastPoints = null;
  drawWire();
  eStatusEl.textContent = "connecting…";
  msg("starting wire evolution…");
  try {
    const { job_id } = await api("/api/antenna/evolve", body);
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

// ---- SURVEY -----------------------------------------------------------------
// Virtual site survey: terrain profile + LOS/Fresnel geometry + link budget
// (/api/antenna/survey) and a Terrarium-tile terrain heatmap with the path
// overlaid (/api/antenna/survey/map). Both canvases are plain 2D and redraw
// on themechange (the map PNG re-tints itself via retintCanvas; only the
// overlay needs the explicit redraw).

function drawSurveyProfile() {
  if (!svProfileCanvas) return;
  const ctx = svProfileCanvas.getContext("2d");
  const W = svProfileCanvas.width, H = svProfileCanvas.height;
  const fg = themeColor("fg"), bg = themeColor("bg"), dim = themeColor("dim");
  const accent = themeColor("accent") || fg;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  const s = lastSurvey;
  if (!s || !Array.isArray(s.dist_m) || s.dist_m.length < 2) {
    ctx.fillStyle = dim;
    ctx.font = "11px monospace";
    ctx.fillText("no survey yet", 8, H / 2);
    return;
  }
  const dist = s.dist_m, elev = s.elev_m;
  const bulge = Array.isArray(s.bulge_m) ? s.bulge_m : elev.map(() => 0);
  const ground = elev.map((e, i) => e + bulge[i]); // terrain + 4/3-earth bulge
  const line = Array.isArray(s.los_line_m) ? s.los_line_m : null;
  const D = s.path_m || dist[dist.length - 1] || 1;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < dist.length; i++) {
    for (const v of [ground[i], line ? line[i] : ground[i]]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (hi - lo < 1e-9) hi = lo + 1;
  const padL = 44, padR = 8, padT = 10, padB = 18;
  const xOf = (d) => padL + (d / D) * (W - padL - padR);
  const yOf = (v) => H - padB - ((v - lo) / (hi - lo)) * (H - padT - padB);
  // terrain silhouette (dim fill)
  ctx.globalAlpha = 0.55;
  ctx.fillStyle = dim;
  ctx.beginPath();
  ctx.moveTo(xOf(dist[0]), yOf(ground[0]));
  for (let i = 1; i < dist.length; i++) ctx.lineTo(xOf(dist[i]), yOf(ground[i]));
  ctx.lineTo(xOf(dist[dist.length - 1]), H - padB);
  ctx.lineTo(xOf(dist[0]), H - padB);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
  // antenna masts: vertical ticks from ground to phase centre at each end
  if (line) {
    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.2;
    for (const i of [0, dist.length - 1]) {
      ctx.beginPath();
      ctx.moveTo(xOf(dist[i]), yOf(ground[i]));
      ctx.lineTo(xOf(dist[i]), yOf(line[i]));
      ctx.stroke();
    }
    // TX→RX LOS sight line (fg)
    ctx.beginPath();
    ctx.moveTo(xOf(dist[0]), yOf(line[0]));
    for (let i = 1; i < dist.length; i++) ctx.lineTo(xOf(dist[i]), yOf(line[i]));
    ctx.stroke();
  }
  // worst-clearance point (accent) — interpolate the profile at that range
  if (typeof s.worst_point_dist_m === "number") {
    const dw = s.worst_point_dist_m;
    let i = 1;
    while (i < dist.length - 1 && dist[i] < dw) i++;
    const t = (dw - dist[i - 1]) / Math.max(dist[i] - dist[i - 1], 1e-9);
    const gw = ground[i - 1] + t * (ground[i] - ground[i - 1]);
    const x = xOf(dw), y = yOf(gw);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, line ? yOf(line[i - 1] + t * (line[i] - line[i - 1])) : y - 10);
    ctx.stroke();
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
  // axis labels (dim): distance km along x, elevation m along y
  ctx.fillStyle = dim;
  ctx.font = "10px monospace";
  ctx.fillText("0 km", padL, H - 5);
  const dMax = `${(D / 1000).toFixed(2)} km`;
  ctx.fillText(dMax, W - padR - ctx.measureText(dMax).width, H - 5);
  ctx.fillText(`${hi.toFixed(0)} m`, 4, padT + 4);
  ctx.fillText(`${lo.toFixed(0)} m`, 4, H - padB);
}

// tile-style mercator: lat → global y fraction, same as the tile servers
function mercY(lat) {
  const r = (lat * Math.PI) / 180;
  return (1 - Math.asinh(Math.tan(r)) / Math.PI) / 2;
}

function drawSurveyMap() {
  if (!svMapCanvas) return;
  const ctx = svMapCanvas.getContext("2d");
  if (!lastMap || !lastMap.map_png_b64) {
    ctx.fillStyle = themeColor("bg");
    ctx.fillRect(0, 0, svMapCanvas.width, svMapCanvas.height);
    ctx.fillStyle = themeColor("dim");
    ctx.font = "11px monospace";
    ctx.fillText("no map yet", 8, svMapCanvas.height / 2);
    return;
  }
  const img = new Image();
  img.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, svMapCanvas.width, svMapCanvas.height);
    ctx.drawImage(img, 0, 0, svMapCanvas.width, svMapCanvas.height);
    retintCanvas(svMapCanvas); // pristine src captured before the overlay
    drawSurveyMapOverlay();
  };
  img.src = `data:image/png;base64,${lastMap.map_png_b64}`;
}

function drawSurveyMapOverlay() {
  if (!svMapCanvas || !lastMap || !lastSurvey) return;
  const m = lastMap, s = lastSurvey;
  const W = svMapCanvas.width, H = svMapCanvas.height;
  const ctx = svMapCanvas.getContext("2d");
  const fg = themeColor("fg"), accent = themeColor("accent") || fg;
  const xOf = (lon) => ((lon - m.lon_lo) / (m.lon_hi - m.lon_lo)) * W; // linear in lon
  const yHi = mercY(m.lat_hi), yLo = mercY(m.lat_lo);
  const yOf = (lat) => ((mercY(lat) - yHi) / (yLo - yHi)) * H;
  const tx = [xOf(s.tx.lon), yOf(s.tx.lat)];
  const rx = [xOf(s.rx.lon), yOf(s.rx.lat)];
  // straight TX→RX path line (fg)
  ctx.strokeStyle = fg;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(tx[0], tx[1]);
  ctx.lineTo(rx[0], rx[1]);
  ctx.stroke();
  // endpoint markers + labels (accent squares, dim labels)
  ctx.font = "10px monospace";
  for (const [p, label, dx] of [[tx, "TX", 6], [rx, "RX", 6]]) {
    ctx.fillStyle = accent;
    ctx.fillRect(p[0] - 3, p[1] - 3, 6, 6);
    ctx.fillStyle = fg;
    const lx = Math.min(Math.max(p[0] + dx, 2), W - 22);
    const ly = Math.min(Math.max(p[1] - 5, 10), H - 4);
    ctx.fillText(label, lx, ly);
  }
}

function renderSurveyStats(s) {
  const lb = s.link_budget || {};
  const vcls = s.verdict === "LOS clear" ? "good" : s.verdict === "obstructed" ? "bad" : "dim";
  const rows = [
    statRow("verdict", s.verdict, vcls),
    statRow("path_km", (s.path_m / 1000).toFixed(3)),
    statRow("LOS", s.los ? "YES" : "NO", s.los ? "good" : "bad"),
    statRow("clearance_m", Number(s.clearance_m).toFixed(1)),
    statRow("min fresnel clearance", Number(s.min_fresnel_clearance).toFixed(3)),
  ];
  if (typeof s.worst_point_dist_m === "number")
    rows.push(statRow("worst point km", (s.worst_point_dist_m / 1000).toFixed(3)));
  rows.push(statRow("diffraction_loss_db", Number(s.diffraction_loss_db).toFixed(1)));
  if (lb.fspl_db !== undefined) rows.push(statRow("fspl_db", Number(lb.fspl_db).toFixed(1)));
  if (s.received_dbw !== undefined)
    rows.push(statRow("received", `${Number(s.received_dbw).toFixed(1)} dBW (${(s.received_dbw + 30).toFixed(1)} dBm)`));
  svStatsEl.replaceChildren(...rows);
}

async function doSurvey() {
  const body = {
    tx: {
      lat: numVal("ant-v-txlat", 46.6),
      lon: numVal("ant-v-txlon", 8.0),
      h_m: numVal("ant-v-txh", 15),
    },
    rx: {
      lat: numVal("ant-v-rxlat", 46.62),
      lon: numVal("ant-v-rxlon", 8.02),
      h_m: numVal("ant-v-rxh", 15),
    },
    f_mhz: numVal("ant-v-f", lastBand ? lastBand.f_center_mhz : 2450),
    p_tx_dbw: numVal("ant-v-ptx", 0),
    g_tx_dbi: numVal("ant-v-gtx", 2.15),
    g_rx_dbi: numVal("ant-v-grx", 2.15),
    medium: document.getElementById("ant-v-medium").value,
    n: 200,
  };
  msg("surveying… (fetching SRTM terrain tiles)");
  svStatusEl.textContent = "running…";
  try {
    const s = await api("/api/antenna/survey", body);
    lastSurvey = s;
    drawSurveyProfile();
    renderSurveyStats(s);
    svStatusEl.textContent = `${s.verdict} · path ${(s.path_m / 1000).toFixed(2)} km · f ${s.f_mhz} MHz`;
    msg(`survey: ${s.verdict}`, s.verdict === "obstructed" ? "error" : "ok");
    try {
      lastMap = await api("/api/antenna/survey/map", { tx: body.tx, rx: body.rx, zoom: 11 });
      drawSurveyMap();
    } catch (e2) {
      lastMap = null;
      msg(`survey ok, map failed: ${e2.message}`, "error");
    }
  } catch (e) {
    svStatusEl.textContent = "failed";
    msg(`survey failed: ${e.message}`, "error");
  }
}

// ---- tab lifecycle ------------------------------------------------------------

function mediumSelect(id) {
  return el(
    "select",
    { id },
    el("option", { value: "air" }, "AIR"),
    el("option", { value: "water" }, "WATER"),
    el("option", { value: "water_sea" }, "WATER-SEA")
  );
}

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
  layerBtns = {};
  const layerRow = el(
    "div",
    { class: "btn-row layer-select", id: "ant-layer-select" },
    ...Object.entries(LAYERS).map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name }, `[${label}]`);
      b.addEventListener("click", () => selectLayer(name));
      layerBtns[name] = b;
      return b;
    })
  );

  // ---- DESIGN panel
  panels.design = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Design recommendation"),
      el("div", { class: "row" }, el("label", {}, "f lo MHz"), el("input", { id: "ant-d-flo", type: "number", value: "2400", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "f hi MHz"), el("input", { id: "ant-d-fhi", type: "number", value: "2485", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-d-medium")),
      el("div", { class: "row" }, el("label", {}, "max size cm"), el("input", { id: "ant-d-size", type: "number", placeholder: "(any)", min: "0" })),
      el("div", { class: "row" }, el("label", {}, "ground plane"), el("input", { id: "ant-d-ground", type: "checkbox" })),
      el("div", { class: "row" }, el("label", {}, "range m"), el("input", { id: "ant-d-range", type: "number", placeholder: "(any)", min: "0" })),
      el(
        "div",
        { class: "row" },
        el("label", {}, "mounting"),
        el(
          "select",
          { id: "ant-d-mounting" },
          el("option", { value: "free" }, "FREE"),
          el("option", { value: "pcb" }, "PCB"),
          el("option", { value: "ground" }, "GROUND")
        )
      ),
      el(
        "div",
        { class: "row" },
        el("label", {}, "polarization"),
        el(
          "select",
          { id: "ant-d-pol" },
          el("option", { value: "any" }, "ANY"),
          el("option", { value: "linear" }, "LINEAR"),
          el("option", { value: "circular" }, "CIRCULAR")
        )
      ),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-d-run" }, "Recommend"))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Ranked candidates (click a row for the trace)"),
      el("div", { id: "ant-d-results" }, el("div", { class: "dim" }, "no design run yet"))
    )
  );

  // ---- PARTS panel
  panels.parts = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Catalog match"),
      el("div", { class: "row" }, el("label", {}, "f lo MHz"), el("input", { id: "ant-p-flo", type: "number", value: "2400", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "f hi MHz"), el("input", { id: "ant-p-fhi", type: "number", value: "2485", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "gain min dBi"), el("input", { id: "ant-p-gain", type: "number", placeholder: "(any)", step: "0.5" })),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-p-run" }, "Find parts")),
      el("div", { class: "status-line" }, "follows the last design-run band")
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Matching parts"),
      el("div", { id: "ant-p-results" }, el("div", { class: "dim" }, "no parts query yet"))
    )
  );

  // ---- KICAD panel
  panels.kicad = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "KiCad / PCB export"),
      el(
        "div",
        { class: "row" },
        el("label", {}, "design type"),
        el(
          "select",
          { id: "ant-k-type" },
          el("option", { value: "patch" }, "PATCH"),
          el("option", { value: "meander_ifa" }, "MEANDER IFA"),
          el("option", { value: "loop" }, "LOOP")
        )
      ),
      el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ant-k-f", type: "number", value: "2450", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "substrate εr"), el("input", { id: "ant-k-epsr", type: "number", value: "4.4", step: "0.1", min: "1" })),
      el("div", { class: "row" }, el("label", {}, "substrate h mm"), el("input", { id: "ant-k-h", type: "number", value: "1.6", step: "0.1", min: "0.1" })),
      el(
        "div",
        { class: "row" },
        el("label", {}, "feed (patch)"),
        el(
          "select",
          { id: "ant-k-feed" },
          el("option", { value: "inset" }, "INSET"),
          el("option", { value: "edge" }, "EDGE")
        )
      ),
      el("div", { class: "row" }, el("label", {}, "medium (loop)"), mediumSelect("ant-k-medium")),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-k-run" }, "Generate"))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Files (.kicad_mod / .kicad_pcb)"),
      el("div", { id: "ant-k-results" }, el("div", { class: "dim" }, "no export yet"))
    )
  );

  // ---- FIELDS panel
  fXyCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  fXzCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  fArCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  fArCell = labeledCell("axial ratio", fArCanvas);
  fArCell.classList.add("hidden");
  const fChartCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  fChart = makeStripChart(fChartCanvas, null);
  panels.fields = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "FDTD field sim"),
      el(
        "div",
        { class: "row" },
        el("label", {}, "f MHz"),
        el("input", { id: "ant-f-f", type: "number", value: "150", step: "1" }),
        el("span", { class: "dim" }, "FDTD needs lower f for big boxes")
      ),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-f-medium")),
      el("div", { class: "row" }, el("label", {}, "interface air↔water"), el("input", { id: "ant-f-interface", type: "checkbox" })),
      el("div", { class: "row" }, el("label", {}, "grid n"), el("input", { id: "ant-f-n", type: "number", value: "48", min: "16", step: "8" })),
      el("div", { class: "row" }, el("label", {}, "max steps"), el("input", { id: "ant-f-steps", type: "number", value: "400", min: "50", step: "50" })),
      el(
        "div",
        { class: "btn-row" },
        el("button", { class: "btn", id: "ant-f-run" }, "Run"),
        (fCancelBtn = el("button", { class: "btn", disabled: true }, "Cancel"))
      )
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Fields"),
      el(
        "div",
        { class: "panel-row" },
        labeledCell("|E| XY", fXyCanvas),
        labeledCell("|E| XZ", fXzCanvas),
        fArCell
      )
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Waveforms"),
      seriesToggleRow(fChart, ["E_RMS", "E_RMS_LO", "E_RMS_HI"], new Set(["E_RMS"])),
      fChartCanvas
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Status"),
      (fStatusEl = el("div", { class: "status-line" }, "idle")),
      el("table", { class: "stats" }, (fStatsEl = el("tbody")))
    )
  );

  // ---- EVOLVE panel
  wireCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  patternCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  patternCell = labeledCell("final pattern", patternCanvas);
  patternCell.classList.add("hidden");
  const eChartCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  eChart = makeStripChart(eChartCanvas, null);
  panels.evolve = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Wire evolution"),
      el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ant-e-f", type: "number", value: "2450", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-e-medium")),
      el("div", { class: "row" }, el("label", {}, "hadamard order"), el("input", { id: "ant-e-order", type: "number", value: "64", min: "4", step: "4" })),
      el("div", { class: "row" }, el("label", {}, "max steps"), el("input", { id: "ant-e-steps", type: "number", value: "2000", min: "100", step: "100" })),
      el(
        "div",
        { class: "btn-row" },
        el("button", { class: "btn", id: "ant-e-run" }, "Evolve"),
        (eCancelBtn = el("button", { class: "btn", disabled: true }, "Cancel"))
      )
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Geometry"),
      el("div", { class: "panel-row" }, labeledCell("best wire (x,y)", wireCanvas), patternCell)
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Waveforms"),
      seriesToggleRow(eChart, ["E", "BEST_E", "T"], new Set(["E", "T"])),
      eChartCanvas
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Status"),
      (eStatusEl = el("div", { class: "status-line" }, "idle")),
      el("table", { class: "stats" }, (eStatsEl = el("tbody")))
    )
  );

  // ---- SMITH panel
  smithCanvas = el("canvas", { class: "sim-canvas smith-canvas", width: "384", height: "384" });
  panels.smith = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Smith chart"),
      el("div", { class: "row" }, el("label", {}, "f lo MHz"), el("input", { id: "ant-s-flo", type: "number", value: "2400", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "f hi MHz"), el("input", { id: "ant-s-fhi", type: "number", value: "2485", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "points"), el("input", { id: "ant-s-n", type: "number", value: "21", min: "3", max: "41", step: "2" })),
      el("div", { class: "row" }, el("label", {}, "Z0 Ω"), el("input", { id: "ant-s-z0", type: "number", value: "50", min: "1", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-s-medium")),
      el(
        "div",
        { class: "row" },
        el("label", {}, "source"),
        el(
          "select",
          { id: "ant-s-src" },
          el("option", { value: "dipole" }, "DIPOLE (textbook λ/2)"),
          el("option", { value: "wire" }, "LAST EVOLVE GEOMETRY")
        )
      ),
      el(
        "div",
        { class: "row" },
        el("label", {}, "design markers"),
        el("input", { id: "ant-s-designs", type: "checkbox", checked: true })
      ),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-s-run" }, "Sweep")),
      (smithReadout = el("div", { class: "status-line" }, "hover the trace for f / Z / Γ / S11"))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Γ plane (Z_in vs frequency, MoM)"),
      el("div", { class: "panel-row" }, labeledCell("smith chart", smithCanvas)),
      el("div", { class: "dim" }, "markers: ● f_lo / f_center / f_hi — squares: ranked designs at f_center")
    )
  );
  smithCanvas.addEventListener("mousemove", onSmithHover);
  smithCanvas.addEventListener("mouseleave", () => {
    smithReadout.textContent = "hover the trace for f / Z / Γ / S11";
    drawSmith();
  });
  drawSmith();

  // ---- SURVEY panel
  svProfileCanvas = el("canvas", { class: "chart", width: "640", height: "200" });
  svMapCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  panels.survey = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Site survey (SRTM terrain + LOS/Fresnel)"),
      el("div", { class: "row" }, el("label", {}, "tx lat"), el("input", { id: "ant-v-txlat", type: "number", value: "46.6", step: "0.001" })),
      el("div", { class: "row" }, el("label", {}, "tx lon"), el("input", { id: "ant-v-txlon", type: "number", value: "8.0", step: "0.001" })),
      el("div", { class: "row" }, el("label", {}, "tx h m"), el("input", { id: "ant-v-txh", type: "number", value: "15", min: "0", max: "500" })),
      el("div", { class: "row" }, el("label", {}, "rx lat"), el("input", { id: "ant-v-rxlat", type: "number", value: "46.62", step: "0.001" })),
      el("div", { class: "row" }, el("label", {}, "rx lon"), el("input", { id: "ant-v-rxlon", type: "number", value: "8.02", step: "0.001" })),
      el("div", { class: "row" }, el("label", {}, "rx h m"), el("input", { id: "ant-v-rxh", type: "number", value: "15", min: "0", max: "500" })),
      el(
        "div",
        { class: "row" },
        el("label", {}, "f MHz"),
        el("input", { id: "ant-v-f", type: "number", value: "2450", step: "1" }),
        el("span", { class: "dim" }, "follows the last design run")
      ),
      el("div", { class: "row" }, el("label", {}, "tx power dBW"), el("input", { id: "ant-v-ptx", type: "number", value: "0", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "g tx dBi"), el("input", { id: "ant-v-gtx", type: "number", value: "2.15", step: "0.1" })),
      el("div", { class: "row" }, el("label", {}, "g rx dBi"), el("input", { id: "ant-v-grx", type: "number", value: "2.15", step: "0.1" })),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-v-medium")),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-v-run" }, "Survey")),
      (svStatusEl = el("div", { class: "status-line" }, "idle")),
      el("div", { class: "dim" }, "terrain: AWS Open Data SRTM Terrarium tiles, no key")
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Link"),
      el("table", { class: "stats" }, (svStatsEl = el("tbody")))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Terrain profile"),
      svProfileCanvas
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Map"),
      el("div", { class: "panel-row" }, labeledCell("terrain map + path", svMapCanvas))
    )
  );
  drawSurveyProfile();
  drawSurveyMap();

  const head = el(
    "div",
    { class: "panel" },
    el("h2", {}, "ANTENNA LAB"),
    layerRow,
    (msgEl = el("div", { class: "msg" }))
  );

  // all panels stay attached (toggled via .hidden) so a running stream
  // keeps rendering into its canvases while another layer is selected
  container.replaceChildren(el("div", { class: "lab ant-cap" }, el("div", {}, head), el("div", {}, ...Object.values(panels))));
  selectLayer("design");

  document.getElementById("ant-d-run").addEventListener("click", doDesign);
  document.getElementById("ant-p-run").addEventListener("click", () => doParts(false));
  document.getElementById("ant-k-run").addEventListener("click", doKicadPanel);
  document.getElementById("ant-s-run").addEventListener("click", doSmith);
  document.getElementById("ant-v-run").addEventListener("click", doSurvey);
  document.getElementById("ant-f-run").addEventListener("click", doFields);
  fCancelBtn.addEventListener("click", doCancelFields);
  document.getElementById("ant-e-run").addEventListener("click", doEvolve);
  eCancelBtn.addEventListener("click", doCancelEvolve);

  window.addEventListener("themechange", onThemechange);
}

export function deactivate() {
  // close both job streams; the jobs themselves keep running server-side
  if (fWs) fWs.close();
  if (eWs) eWs.close();
  fWs = eWs = null;
  window.removeEventListener("themechange", onThemechange);
}
