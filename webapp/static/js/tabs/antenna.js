// Antenna Lab — antenna design / parts matching / FDTD field sim / wire
// evolution, all behind ONE viewport with a [DESIGN][PARTS][FIELDS][EVOLVE]
// layer selector (#ant-layer-select, same convention as #sim-layer-select).
//   DESIGN  POST /api/antenna/design → ranked candidate table
//   PARTS   POST /api/antenna/parts → off-the-shelf part table
//   FIELDS  POST /api/antenna/fields → 3-D field-slice viewer
//           (Three.js orthogonal planes) + E_RMS strip chart
//   EVOLVE  POST /api/antenna/evolve → wire geometry SA + E/BEST_E/T chart
//   KICAD   PCB footprint/board export (MIFA, RF_Antenna.pretty lib,
//           evolved walks) + JLCPCB design_params + canvas preview
//   SMITH   Γ-plane MoM Z_in(f) sweep with interactive hover readout
//   ARRAY   POST /api/antenna/array → polar pattern plot + beam metrics
// Site survey has moved to the Terrain tab (terrain.js — [SURVEY] button).
// Server PNG canvases go through retintCanvas (registers them for
// themechange re-tints); the wire canvas redraws on themechange itself.
// Panels are kept attached and toggled with .hidden so a running job's
// stream keeps updating its canvases while another layer is up.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { retintCanvas, fillPlusOne, themeColor } from "/js/theme.js";
import { drawKicadPrims } from "/js/kicad_layers.js";

const LAYERS = { design: "DESIGN", parts: "PARTS", fields: "FIELDS", evolve: "EVOLVE", kicad: "KICAD", smith: "SMITH", array: "ARRAY" };
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
let f3Renderer, f3Scene, f3Camera, f3Controls, f3Container;
let f3PlaneXY, f3PlaneXZ, f3TexXY, f3TexXZ, f3Raf = null;
let f3Cell = null;
// evolve job state
let eWs = null, eJob = null, eChart, eCancelBtn, eStatusEl, eStatsEl;
let smithCanvas, smithReadout, lastSweep = null;
let arrayCanvas, arrayStatsEl, lastArray = null;
let lastEvolvePoints = null, lastDesignEntries = null;
let wireCanvas, patternCanvas, patternCell;
let lastPoints = null; // last evolve points — redrawn on themechange
let kicadCanvas, lastKicadPreview = null, lastKicadPreviewBoard = null;
let kicadPreviewMode = "footprint", lastEvolveKind = "wire";

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
    if (Array.isArray(k)) return k.join("×");
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
const EVOLVE_STAT_KEYS = ["steps", "accepts", "best_E", "elapsed_s", "gain_dbi", "s11_db", "kind", "seed_row"];

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
  if (name === "fields" && f3Cell) initFieldsThree(f3Cell);
  if (name === "kicad") {
    loadKicadLibrary();
    drawKicadPreview();
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
    lastKicadPreview = d.preview || null;
    lastKicadPreviewBoard = d.preview_board || null;
    if (!lastKicadPreviewBoard) kicadPreviewMode = "footprint";
    drawKicadPreview();
    syncKicadPreviewBtns();
    if (d.params) renderKicadParams(d.params);
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
  if (ktype === "lib") {
    const sel = document.getElementById("ant-k-lib");
    if (!sel || !sel.value) {
      msg("pick a KiCad RF_Antenna library footprint", "error");
      return;
    }
    opts.lib_name = sel.value;
  }
  if (ktype === "evolved") {
    if (!lastEvolvePoints) {
      msg("evolve a geometry first (EVOLVE tab)", "error");
      return;
    }
    opts.points = lastEvolvePoints;
  }
  const btn = document.getElementById("ant-k-run");
  await doKicad(ktype, btn, numVal("ant-k-f", 2450), opts, document.getElementById("ant-k-results"));
}

function renderKicadParams(p) {
  const host = document.getElementById("ant-k-params");
  if (!host || !p) return;
  const rows = [
    statRow("λ₀", `${Number(p.lambda0_mm).toFixed(2)} mm`),
    statRow("λ/4", `${Number(p.L_quarter_mm).toFixed(2)} mm`),
    statRow("λ/2", `${Number(p.L_half_mm).toFixed(2)} mm`),
    statRow("L_q eff", `${Number(p.L_q_eff_mm).toFixed(2)} mm`),
    statRow("Z₀ / η₀", `${p.z0_ohm} Ω / ${Number(p.eta0_ohm).toFixed(1)} Ω`),
    statRow("W_50", `${Number(p.w50_mm).toFixed(3)} mm`),
    statRow("RL target", `≥ ${p.return_loss_target_db} dB (${Math.round((p.return_loss_frac_accepted || 0.9) * 100)} % accepted)`),
    statRow("MIFA trace", `${p.mifa_trace_mil} mil (${Number(p.mifa_trace_mm).toFixed(3)} mm)`),
    statRow("MIFA bbox", `${Number(p.mifa_bbox_mm[0]).toFixed(2)} × ${Number(p.mifa_bbox_mm[1]).toFixed(2)} mm`),
    statRow("GND extent", `${Number(p.gnd_extent_mm).toFixed(2)} mm (6h)`),
    statRow("keep-out", p.keepout_under_radiator ? "under radiator" : "—"),
    statRow("matching", p.matching || "π-network"),
    statRow("min trace", `${Number(p.min_trace_mm).toFixed(3)} mm (5 mil)`),
    statRow("GND rule", p.ground_plane_note || "—"),
    statRow("solder mask", p.solder_mask_note || "—"),
    statRow("source", p.source || "JLCPCB"),
  ];
  host.replaceChildren(...rows);
}

function syncKicadForm() {
  const t = document.getElementById("ant-k-type")?.value;
  const hide = (id, on) => {
    const n = document.getElementById(id);
    if (n) n.classList.toggle("hidden", !!on);
  };
  hide("ant-k-row-feed", t !== "patch");
  hide("ant-k-row-medium", t !== "loop");
  hide("ant-k-row-lib", t !== "lib");
  hide("ant-k-row-sub", t === "loop");
  if (t === "lib") loadKicadLibrary();
}

function syncKicadPreviewBtns() {
  const boardBtn = document.getElementById("ant-k-prev-board");
  if (boardBtn) {
    boardBtn.disabled = !lastKicadPreviewBoard;
    boardBtn.style.opacity = kicadPreviewMode === "board" ? "1" : "0.45";
  }
  const fpBtn = document.getElementById("ant-k-prev-fp");
  if (fpBtn) fpBtn.style.opacity = kicadPreviewMode === "footprint" ? "1" : "0.45";
}

async function loadKicadLibrary() {
  const sel = document.getElementById("ant-k-lib");
  if (!sel || sel.dataset.loaded === "1") return;
  try {
    const d = await api("/api/antenna/kicad/library");
    const fps = d.footprints || [];
    if (!fps.length) {
      sel.replaceChildren(el("option", { value: "" }, "(no RF_Antenna.pretty)"));
      return;
    }
    sel.replaceChildren(...fps.map((f) => el("option", { value: f.name }, f.name)));
    const ti = fps.find((f) => f.name.includes("SWRA117D")) || fps[0];
    sel.value = ti.name;
    sel.dataset.loaded = "1";
  } catch (e) {
    sel.replaceChildren(el("option", { value: "" }, "(library unavailable)"));
  }
}

function drawKicadPreview() {
  if (!kicadCanvas) return;
  const prev = (kicadPreviewMode === "board" && lastKicadPreviewBoard)
    ? lastKicadPreviewBoard : lastKicadPreview;
  const cap = prev && prev.bbox
    ? `${kicadPreviewMode}  ${(prev.bbox.xmax - prev.bbox.xmin).toFixed(2)} × ${(prev.bbox.ymax - prev.bbox.ymin).toFixed(2)} mm`
    : "";
  drawKicadPrims(kicadCanvas, prev, { caption: cap });
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
        if (!lastBand) return;
        const typeEl = document.getElementById("ant-k-type");
        const fEl = document.getElementById("ant-k-f");
        if (typeEl) typeEl.value = ktype;
        if (fEl) fEl.value = String(lastBand.f_center_mhz);
        syncKicadForm();
        selectLayer("kicad");
        doKicadPanel();
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
    const matches = d.matches || [];
    const rows = matches.map((m) =>
      el(
        "tr",
        m.coverage_note ? { style: "opacity:0.55", title: m.coverage_note } : {},
        el("td", { class: "k" }, m.part),
        el("td", {}, m.mfr || "?"),
        el("td", { class: "dim" }, m.type || "?"),
        el("td", {},
          `${m.freq_lo_mhz}–${m.freq_hi_mhz}` +
          (m.coverage_frac != null && m.coverage_frac < 1
            ? ` (${Math.round(m.coverage_frac * 100)}%)`
            : "")),
        el("td", {}, m.gain_dbi !== undefined ? `${m.gain_dbi} dBi` : "?"),
        el("td", {}, m.vswr !== undefined ? String(m.vswr) : "?"),
        el("td", { class: "dim" }, Array.isArray(m.size_mm) ? m.size_mm.join("×") : (m.size_mm || "?")),
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
    const host = document.getElementById("ant-p-results");
    if (!matches.length) {
      host.replaceChildren(el("div", { class: "dim" },
        `no catalog parts overlap ${body.f_lo_mhz}–${body.f_hi_mhz} MHz` +
        (body.gain_dbi_min != null ? ` @ ≥ ${body.gain_dbi_min} dBi` : "")));
    } else {
      host.replaceChildren(el("table", { class: "stats" }, head, ...rows));
    }
    const cov = d.coverage === "overlap"
      ? `${matches.length} overlap ${body.f_lo_mhz}–${body.f_hi_mhz} MHz (none cover the full span)`
      : `${matches.length} parts cover ${body.f_lo_mhz}–${body.f_hi_mhz} MHz`;
    if (!quiet || !matches.length) msg(cov, "ok");
  } catch (e) {
    msg(`parts failed: ${e.message}`, "error");
  }
}

// ---- FIELDS -----------------------------------------------------------------

function f3Render() {
  if (!f3Renderer || !f3Scene || !f3Camera) return;
  if (f3Raf) return; // throttle to rAF
  f3Raf = requestAnimationFrame(() => { f3Raf = null; f3Renderer.render(f3Scene, f3Camera); });
}

function initFieldsThree(container) {
  if (f3Renderer) return;
  const w = 256;
  f3Renderer = new THREE.WebGLRenderer({ antialias: true });
  f3Renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  f3Renderer.setSize(w, w, false);
  f3Renderer.setClearColor(new THREE.Color(themeColor("bg")));
  f3Renderer.domElement.classList.add("sim-canvas");
  container.appendChild(f3Renderer.domElement);
  f3Container = container;

  f3Scene = new THREE.Scene();
  f3Camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  f3Camera.position.set(0.8, 0.6, 2.2);
  f3Controls = new OrbitControls(f3Camera, f3Renderer.domElement);
  f3Controls.addEventListener("change", f3Render);

  f3TexXY = new THREE.CanvasTexture(el("canvas", { width: "128", height: "128" }));
  f3TexXZ = new THREE.CanvasTexture(el("canvas", { width: "128", height: "128" }));
  const planeMatXY = new THREE.MeshBasicMaterial({ map: f3TexXY, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
  const planeMatXZ = new THREE.MeshBasicMaterial({ map: f3TexXZ, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
  f3PlaneXY = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMatXY);
  f3PlaneXZ = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMatXZ);
  f3PlaneXZ.rotation.x = -Math.PI / 2; // XZ → horizontal
  f3Scene.add(f3PlaneXY);
  f3Scene.add(f3PlaneXZ);
  f3Scene.add(new THREE.AxesHelper(1.0));
  window.addEventListener("themechange", f3Theme);
  f3Render();
}

function f3Theme() {
  if (!f3Renderer) return;
  f3Renderer.setClearColor(new THREE.Color(themeColor("bg")));
  f3Render();
}

function disposeFieldsThree() {
  if (f3Raf) cancelAnimationFrame(f3Raf);
  f3Raf = null;
  window.removeEventListener("themechange", f3Theme);
  if (f3Controls) f3Controls.dispose();
  if (f3Renderer) {
    if (f3TexXY) f3TexXY.dispose();
    if (f3TexXZ) f3TexXZ.dispose();
    f3Scene && f3Scene.children.slice().forEach((o) => { f3Scene.remove(o); o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); });
    f3Renderer.dispose();
    f3Renderer.domElement.remove();
  }
  f3Renderer = f3Scene = f3Camera = f3Controls = f3Container = f3PlaneXY = f3PlaneXZ = f3TexXY = f3TexXZ = null;
}

function updateFieldTex(name, b64) {
  const img = new Image();
  img.onload = () => {
    const tex = name === "xy" ? f3TexXY : f3TexXZ;
    if (!tex) return;
    const w = tex.source.data.width, h = tex.source.data.height;
    const ctx = tex.source.data.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    tex.needsUpdate = true;
    f3Render();
  };
  img.src = `data:image/png;base64,${b64}`;
}

function handleFieldFrame(d) {
  if (d.type === "snapshot") {
    fStatusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleFieldFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.e_xy_png_b64) updateFieldTex("xy", d.e_xy_png_b64);
    if (d.e_xz_png_b64) updateFieldTex("xz", d.e_xz_png_b64);
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
  if (f3Cell) initFieldsThree(f3Cell);
  // both field planes start as an all-+1 field, not transparent, until the
  // first FDTD frame (step frame_every) lands via updateFieldTex
  for (const tex of [f3TexXY, f3TexXZ]) {
    if (!tex) continue;
    fillPlusOne(tex.source.data);
    tex.needsUpdate = true;
  }
  f3Render();
  fChart.clear();
  fStatsEl.replaceChildren();
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
  drawKicadPreview();
  drawArray();
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

// ---- ARRAY ------------------------------------------------------------------
// Polar pattern of total_db (element × AF, fg) with af_db underneath (dim),
// floored at −40 dB; θ runs −90..90° off broadside so the plot is the upper
// semicircle (θ=0 up). Guide rays mark the steered beam and the first
// grating lobe when metrics.grating_lobe_deg is set. Redraws on themechange.

const ARRAY_FLOOR_DB = -40;

function arrayPolar(canvas, thetaDeg, db, R) {
  const cx = canvas.width / 2, cy = canvas.height - 26;
  const a = (thetaDeg * Math.PI) / 180;
  const r = Math.max(0, Math.min(1, (db - ARRAY_FLOOR_DB) / -ARRAY_FLOOR_DB));
  return [cx + r * R * Math.sin(a), cy - r * R * Math.cos(a)];
}

function drawArray() {
  if (!arrayCanvas) return;
  const ctx = arrayCanvas.getContext("2d");
  const W = arrayCanvas.width, H = arrayCanvas.height;
  const fg = themeColor("fg"), bg = themeColor("bg"), dim = themeColor("dim");
  const accent = themeColor("accent") || fg;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  const cx = W / 2, cy = H - 26, R = Math.min(W / 2, H - 40) - 14;
  // dB rings (0/−10/−20/−30/−40) and 15° spokes, dim
  ctx.strokeStyle = dim;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.6;
  for (const db of [0, -10, -20, -30, -40]) {
    ctx.beginPath();
    ctx.arc(cx, cy, (db / ARRAY_FLOOR_DB) * R, Math.PI, 2 * Math.PI);
    ctx.stroke();
  }
  for (let deg = -90; deg <= 90; deg += 15) {
    const [x, y] = arrayPolar(arrayCanvas, deg, 0, R);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = dim;
  ctx.font = "10px monospace";
  ctx.fillText("0 dB", cx + 3, cy - R + 10);
  ctx.fillText("−40", cx + 3, cy - 4);
  ctx.fillText("0°", cx - 4, cy - R - 4);
  ctx.fillText("−90°", cx - R + 2, cy - 12);
  ctx.fillText("+90°", cx + R - 26, cy - 12);
  if (!lastArray) return;
  // guide rays: steered beam (accent) and first grating lobe (dim, dashed)
  const guide = (deg, color) => {
    const [x, y] = arrayPolar(arrayCanvas, deg, 0, R);
    ctx.strokeStyle = color;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (lastArray.steer_deg) guide(lastArray.steer_deg, accent);
  const gl = lastArray.metrics?.grating_lobe_deg;
  if (gl !== null && gl !== undefined) guide(gl, dim);
  // traces: af_db dim, total_db fg
  const trace = (key, color, width) => {
    const th = lastArray.theta_deg, db = lastArray[key];
    if (!th || !db) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    th.forEach((t, i) => {
      const [x, y] = arrayPolar(arrayCanvas, t, db[i], R);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
  trace("af_db", dim, 1);
  trace("total_db", fg, 2);
}

function syncArrayForm() {
  const isLam = document.getElementById("ant-a-smode").value === "lambda";
  document.getElementById("ant-a-dl").classList.toggle("hidden", !isLam);
  document.getElementById("ant-a-dmm").classList.toggle("hidden", isLam);
  const cheb = document.getElementById("ant-a-taper").value === "chebyshev";
  document.getElementById("ant-a-sll").disabled = !cheb;
}

async function doArray() {
  msg("designing array…");
  try {
    const body = {
      n: parseInt(document.getElementById("ant-a-n").value, 10),
      f_mhz: numVal("ant-a-f", 2450),
      taper: document.getElementById("ant-a-taper").value,
      sll_db: numVal("ant-a-sll", 30),
      steer_deg: numVal("ant-a-steer", 0),
      element: document.getElementById("ant-a-element").value,
      medium: document.getElementById("ant-a-medium").value,
    };
    if (document.getElementById("ant-a-smode").value === "lambda") body.d_lambda = numVal("ant-a-dl", 0.5);
    else body.d_mm = numVal("ant-a-dmm", 61);
    lastArray = await api("/api/antenna/array", body);
    drawArray();
    const m = lastArray.metrics;
    arrayStatsEl.replaceChildren(
      statRow("hpbw_deg", m.hpbw_deg === null ? "—" : Number(m.hpbw_deg.toFixed(2))),
      statRow("sll_db", m.sll_db === null ? "−∞ (no sidelobes)" : Number(m.sll_db.toFixed(2))),
      statRow("grating_lobe_deg", m.grating_lobe_deg === null ? "—" : Number(m.grating_lobe_deg.toFixed(1))),
      statRow("directivity_dbi", Number(m.directivity_dbi.toFixed(2))),
      statRow("aperture_mm", lastArray.aperture_mm),
      statRow("d_mm", Number(lastArray.d_mm.toFixed(2))),
      statRow("d_lambda", Number(lastArray.d_lambda.toFixed(3)))
    );
    msg(`array: ${lastArray.notes}`, "ok");
  } catch (e) {
    msg(`array failed: ${e.message}`, "error");
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
      lastEvolvePoints = d.points;
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
    if (r.kind) lastEvolveKind = r.kind;
    eStatsEl.replaceChildren(...resultStatRows(r, EVOLVE_STAT_KEYS), resultDownloadRow(eJob, r));
    msg("evolution complete", "ok");
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

async function doEvolve() {
  const topo = document.getElementById("ant-e-topo")?.value || "meander";
  lastEvolveKind = topo === "pcb" ? "pcb" : "wire";
  const body = {
    f_mhz: numVal("ant-e-f", 2450),
    medium: document.getElementById("ant-e-medium").value,
    topology: topo,
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
  // the pattern streams during the run now — show an all-+1 field from
  // job start instead of hiding the cell until the final frame
  patternCell.classList.remove("hidden");
  fillPlusOne(patternCanvas);
  lastPoints = null;
  lastEvolvePoints = null;
  drawWire();
  eStatusEl.textContent = "connecting…";
  msg(topo === "pcb" ? "starting PCB evolution…" : "starting wire evolution…");
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

async function doEvolveKicad() {
  if (!lastEvolvePoints) {
    msg("evolve a geometry first", "error");
    return;
  }
  const btn = document.getElementById("ant-e-kicad");
  selectLayer("kicad");
  const typeEl = document.getElementById("ant-k-type");
  if (typeEl) typeEl.value = "evolved";
  syncKicadForm();
  await doKicad("evolved", btn, numVal("ant-e-f", 2450), {
    points: lastEvolvePoints,
    eps_r: numVal("ant-k-epsr", 4.4),
    h_mm: numVal("ant-k-h", 1.6),
  }, document.getElementById("ant-k-results"));
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
  kicadCanvas = el("canvas", { class: "sim-canvas kicad-canvas", width: "384", height: "384" });
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
          el("option", { value: "mifa" }, "MIFA (JLCPCB)"),
          el("option", { value: "loop" }, "LOOP"),
          el("option", { value: "lib" }, "LIBRARY (RF_Antenna)"),
          el("option", { value: "evolved" }, "LAST EVOLVED")
        )
      ),
      el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ant-k-f", type: "number", value: "2450", step: "1" })),
      el(
        "div",
        { class: "row", id: "ant-k-row-sub" },
        el("label", {}, "substrate εr / h mm"),
        el("input", { id: "ant-k-epsr", type: "number", value: "4.4", step: "0.1", min: "1" }),
        el("input", { id: "ant-k-h", type: "number", value: "1.6", step: "0.1", min: "0.1" })
      ),
      el(
        "div",
        { class: "row", id: "ant-k-row-feed" },
        el("label", {}, "feed (patch)"),
        el(
          "select",
          { id: "ant-k-feed" },
          el("option", { value: "inset" }, "INSET"),
          el("option", { value: "edge" }, "EDGE")
        )
      ),
      el("div", { class: "row", id: "ant-k-row-medium" }, el("label", {}, "medium (loop)"), mediumSelect("ant-k-medium")),
      el(
        "div",
        { class: "row hidden", id: "ant-k-row-lib" },
        el("label", {}, "library fp"),
        el("select", { id: "ant-k-lib" }, el("option", { value: "" }, "(loading…)"))
      ),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-k-run" }, "Generate"))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Preview"),
      el(
        "div",
        { class: "btn-row layer-select" },
        el("button", { class: "btn", id: "ant-k-prev-fp" }, "[FOOTPRINT]"),
        el("button", { class: "btn", id: "ant-k-prev-board", disabled: true }, "[BOARD]")
      ),
      el("div", { class: "panel-row" }, labeledCell("component / PCB", kicadCanvas)),
      el("div", { class: "dim" }, "KiCad layers (not themed): red F.Cu · blue B.Cu · green In1 · orange In2")
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "JLCPCB design parameters"),
      el("table", { class: "stats" }, el("tbody", { id: "ant-k-params" },
        el("tr", {}, el("td", { class: "dim" }, "generate to fill λ/4, 50 Ω, RL ≥ 10 dB, 20 mil MIFA, 6h GND"))))
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Files (.kicad_mod / .kicad_pcb)"),
      el("div", { id: "ant-k-results" }, el("div", { class: "dim" }, "no export yet"))
    )
  );

  // ---- FIELDS panel
  const fChartCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  fChart = makeStripChart(fChartCanvas, null);
  f3Cell = el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "|E| slices (drag to orbit)"));
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
      el("h2", {}, "Fields (3D)"),
      el("div", { class: "panel-row" }, f3Cell)
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
      el("h2", {}, "Wire / PCB evolution"),
      el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ant-e-f", type: "number", value: "2450", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-e-medium")),
      el(
        "div",
        { class: "row" },
        el("label", {}, "topology"),
        el(
          "select",
          { id: "ant-e-topo" },
          el("option", { value: "meander" }, "WIRE (λ/2)"),
          el("option", { value: "pcb" }, "PCB (λ/4 MIFA)")
        )
      ),
      el("div", { class: "row" }, el("label", {}, "hadamard order"), el("input", { id: "ant-e-order", type: "number", value: "64", min: "4", step: "4" })),
      el("div", { class: "row" }, el("label", {}, "max steps"), el("input", { id: "ant-e-steps", type: "number", value: "2000", min: "100", step: "100" })),
      el(
        "div",
        { class: "btn-row" },
        el("button", { class: "btn", id: "ant-e-run" }, "Evolve"),
        (eCancelBtn = el("button", { class: "btn", disabled: true }, "Cancel")),
        el("button", { class: "btn", id: "ant-e-kicad" }, "Export KiCad")
      )
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Geometry"),
      el("div", { class: "panel-row" }, labeledCell("best walk (x,y)", wireCanvas), patternCell)
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

  // ---- ARRAY panel
  arrayCanvas = el("canvas", { class: "sim-canvas smith-canvas", width: "384", height: "300" });
  panels.array = el(
    "div",
    {},
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Phased array"),
      el("div", { class: "row" }, el("label", {}, "elements N"), el("input", { id: "ant-a-n", type: "number", value: "8", min: "1", max: "64", step: "1" })),
      el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ant-a-f", type: "number", value: "2450", step: "1" })),
      el(
        "div",
        { class: "row" },
        el("label", {}, "spacing"),
        el(
          "select",
          { id: "ant-a-smode" },
          el("option", { value: "lambda" }, "D IN λ"),
          el("option", { value: "mm" }, "D IN MM")
        ),
        el("input", { id: "ant-a-dl", type: "number", value: "0.5", min: "0.1", max: "2", step: "0.05" }),
        el("input", { id: "ant-a-dmm", class: "hidden", type: "number", value: "61", min: "0.1", step: "0.5" })
      ),
      el(
        "div",
        { class: "row" },
        el("label", {}, "taper"),
        el(
          "select",
          { id: "ant-a-taper" },
          el("option", { value: "uniform" }, "UNIFORM"),
          el("option", { value: "binomial" }, "BINOMIAL"),
          el("option", { value: "chebyshev" }, "CHEBYSHEV")
        ),
        el("input", { id: "ant-a-sll", type: "number", value: "30", min: "5", max: "80", step: "5", disabled: true })
      ),
      el("div", { class: "row" }, el("label", {}, "steer deg"), el("input", { id: "ant-a-steer", type: "number", value: "0", min: "-80", max: "80", step: "5" })),
      el(
        "div",
        { class: "row" },
        el("label", {}, "element"),
        el(
          "select",
          { id: "ant-a-element" },
          ...["dipole", "helix", "loop", "meander", "monopole", "patch", "pifa", "slot", "yagi"].map((k) =>
            el("option", { value: k }, k.toUpperCase())
          )
        )
      ),
      el("div", { class: "row" }, el("label", {}, "medium"), mediumSelect("ant-a-medium")),
      el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ant-a-run" }, "Design")),
      el("div", { class: "dim" }, "SLL dB applies to the CHEBYSHEV taper only")
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Polar pattern (dB, floor −40)"),
      el("div", { class: "panel-row" }, labeledCell("total (fg) + array factor (dim)", arrayCanvas)),
      el("div", { class: "dim" }, "guides: dashed accent = steer, dashed dim = grating lobe")
    ),
    el(
      "div",
      { class: "panel" },
      el("h2", {}, "Beam metrics"),
      el("table", { class: "stats" }, (arrayStatsEl = el("tbody", {},
        el("tr", {}, el("td", { class: "dim" }, "design an array to fill"))))
      )
    )
  );
  drawArray();

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
  document.getElementById("ant-k-type").addEventListener("change", syncKicadForm);
  document.getElementById("ant-k-prev-fp").addEventListener("click", () => {
    kicadPreviewMode = "footprint";
    syncKicadPreviewBtns();
    drawKicadPreview();
  });
  document.getElementById("ant-k-prev-board").addEventListener("click", () => {
    if (!lastKicadPreviewBoard) return;
    kicadPreviewMode = "board";
    syncKicadPreviewBtns();
    drawKicadPreview();
  });
  document.getElementById("ant-s-run").addEventListener("click", doSmith);
  document.getElementById("ant-a-run").addEventListener("click", doArray);
  document.getElementById("ant-a-smode").addEventListener("change", syncArrayForm);
  document.getElementById("ant-a-taper").addEventListener("change", syncArrayForm);
  document.getElementById("ant-f-run").addEventListener("click", doFields);
  fCancelBtn.addEventListener("click", doCancelFields);
  document.getElementById("ant-e-run").addEventListener("click", doEvolve);
  document.getElementById("ant-e-kicad").addEventListener("click", doEvolveKicad);
  eCancelBtn.addEventListener("click", doCancelEvolve);
  syncKicadForm();
  drawKicadPreview();

  window.addEventListener("themechange", onThemechange);
}

export function deactivate() {
  if (fWs) fWs.close();
  if (eWs) eWs.close();
  fWs = eWs = null;
  disposeFieldsThree();
  window.removeEventListener("themechange", onThemechange);
}
