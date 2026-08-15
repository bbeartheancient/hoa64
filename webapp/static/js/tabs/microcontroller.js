// Microcontroller Lab — LED matrix designer, WiFi mesh field (ALPHA),
// edge engine export (Phase: routes_mcu).
//   LED   paint a W×H frame grid (WS2812 GRB + serpentine server-side),
//         frame list + PLAY animation, [FIRMWARE] → one-shot downloads,
//         [SEND FRAME] → POST /api/mcu/push to the board's /frame.
//   MESH  ESP-NOW RSSI tomography (ALPHA): [FIRMWARE] per-node sketch,
//         [COLLECT] pulls the n×n link matrix, circle-link viz,
//         [CALIBRATE] baseline + delta links, [EXPORT JSON] handoff
//         artifact (mesh_field.json) for the external spatialxr project.
//   EDGE  engine × target template export via POST /api/mcu/export.

import { themeColor } from "/js/theme.js";

const LAYERS = { led: "LED", mesh: "MESH (ALPHA)", edge: "EDGE" };

// LED swatches are the literal device colors (like the KiCad copper
// palette) — not themed.
const SWATCHES = [
  ["#ff2020", "red"], ["#20ff20", "green"], ["#2040ff", "blue"],
  ["#ffe020", "yellow"], ["#20e0ff", "cyan"], ["#ff20e0", "magenta"],
  ["#ff9020", "orange"], ["#ffffff", "white"], ["#000000", "off"],
];

let msgEl, layerBtns = {};
let layer = "led";
let selectLayer = () => {};

// ---- LED state
let ledCanvas, frames = [], curFrame = 0, paint = SWATCHES[0][0];
let eraseMode = false, strokeErase = false, playTimer = null, painting = false;
let ledNotesEl, ledFilesEl;

// ---- MESH state
let meshCanvas, lastMesh = null, baseline = null;

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

function intVal(id, fallback) {
  const v = parseInt(document.getElementById(id).value, 10);
  return Number.isFinite(v) ? v : fallback;
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function fileLinks(host, d, prefix) {
  host.replaceChildren(
    ...(d.files || []).map((name) =>
      el("div", { class: "row" },
        el("a", { class: "btn btn-xs", href: `${prefix}/${d.token}/${name}`, download: name }, name))),
    el("div", { class: "dim" }, "one-shot links — click to download")
  );
}

// ---- LED layer ---------------------------------------------------------------

function gridWH() {
  return [clamp(intVal("mcu-w", 16), 1, 64), clamp(intVal("mcu-h", 16), 1, 64)];
}

function blankFrame() {
  const [w, h] = gridWH();
  return Array.from({ length: w * h }, () => [0, 0, 0]);
}

function resetFrames() {
  frames = [blankFrame()];
  curFrame = 0;
  renderFrameList();
  drawLed();
}

function hexToRgb(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

function drawLed() {
  if (!ledCanvas || !frames.length) return;
  const [w, h] = gridWH();
  const ctx = ledCanvas.getContext("2d");
  const S = ledCanvas.width;
  const cw = S / w, ch = S / h;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, S, S);
  const px = frames[curFrame];
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const [pr, pg, pb] = px[r * w + c];
      if (pr || pg || pb) {
        ctx.fillStyle = `rgb(${pr},${pg},${pb})`;
        ctx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
      }
    }
  ctx.strokeStyle = themeColor("dim");
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  for (let c = 1; c < w; c++) { ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, S); }
  for (let r = 1; r < h; r++) { ctx.moveTo(0, r * ch); ctx.lineTo(S, r * ch); }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function paintAt(ev) {
  const [w, h] = gridWH();
  const rect = ledCanvas.getBoundingClientRect();
  const c = Math.floor(((ev.clientX - rect.left) / rect.width) * w);
  const r = Math.floor(((ev.clientY - rect.top) / rect.height) * h);
  if (r < 0 || r >= h || c < 0 || c >= w) return;
  frames[curFrame][r * w + c] = eraseMode || strokeErase ? [0, 0, 0] : hexToRgb(paint);
  drawLed();
}

function renderFrameList() {
  const host = document.getElementById("mcu-frames");
  if (!host) return;
  host.replaceChildren(...frames.map((_, i) => {
    const b = el("button", { class: "btn btn-xs", style: `opacity:${i === curFrame ? 1 : 0.45}` }, `F${i}`);
    b.addEventListener("click", () => { curFrame = i; renderFrameList(); drawLed(); });
    return b;
  }));
}

function stopPlay() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  const b = document.getElementById("mcu-play");
  if (b) b.textContent = "[PLAY]";
}

function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  if (frames.length < 2) { msg("add a second frame to animate", "error"); return; }
  const delay = clamp(intVal("mcu-delay", 200), 30, 5000);
  playTimer = setInterval(() => {
    curFrame = (curFrame + 1) % frames.length;
    renderFrameList();
    drawLed();
  }, delay);
  document.getElementById("mcu-play").textContent = "[STOP]";
}

function ledBody() {
  const [w, h] = gridWH();
  const pinRaw = document.getElementById("mcu-pin").value;
  return {
    kind: "led",
    board: document.getElementById("mcu-board").value,
    w, h,
    pin: pinRaw === "" ? null : parseInt(pinRaw, 10),
    serpentine: document.getElementById("mcu-serpentine").checked,
    ssid: document.getElementById("mcu-ssid").value || null,
    password: document.getElementById("mcu-pass").value || null,
    brightness: clamp(intVal("mcu-brightness", 64), 0, 255),
  };
}

async function doLedFirmware() {
  msg("generating firmware…");
  try {
    const d = await api("/api/mcu/firmware", ledBody());
    fileLinks(ledFilesEl, d, "/api/mcu/file");
    ledNotesEl.textContent = d.notes || "";
    msg(`firmware: ${(d.files || []).join(", ")}`, "ok");
  } catch (e) {
    msg(`firmware failed: ${e.message}`, "error");
  }
}

async function doSendFrame() {
  const host = document.getElementById("mcu-host").value.trim();
  if (!host) { msg("give the board IP (e.g. 192.168.4.1)", "error"); return; }
  const [w, h] = gridWH();
  msg("sending frame…");
  try {
    const d = await api("/api/mcu/push", {
      host, w, h,
      serpentine: document.getElementById("mcu-serpentine").checked,
      pixels: frames[curFrame],
    });
    msg(`push: ${d.bytes} bytes → status ${d.status}`, d.ok ? "ok" : "error");
  } catch (e) {
    msg(`push failed: ${e.message}`, "error");
  }
}

// ---- MESH layer --------------------------------------------------------------

function drawMesh() {
  if (!meshCanvas) return;
  const ctx = meshCanvas.getContext("2d");
  const S = meshCanvas.width;
  ctx.clearRect(0, 0, S, S);
  const fg = themeColor("fg"), dim = themeColor("dim"), acc = themeColor("accent");
  ctx.font = "11px monospace";
  if (!lastMesh) {
    ctx.fillStyle = dim;
    ctx.fillText("collect a mesh to see the field", 12, S / 2);
    return;
  }
  const n = lastMesh.n || (lastMesh.rssi_dbm || []).length || 0;
  const R = lastMesh.rssi_dbm || [];
  if (!n) return;
  const cx = S / 2, cy = S / 2, rad = S / 2 - 34;
  const pos = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n - Math.PI / 2;
    pos.push([cx + rad * Math.cos(a), cy + rad * Math.sin(a)]);
  }
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const rssi = R[i] && R[i][j];
      if (rssi == null || rssi <= -128) continue;
      const t = clamp((rssi + 100) / 60, 0, 1);
      ctx.strokeStyle = fg;
      ctx.globalAlpha = 0.15 + 0.7 * t;
      ctx.lineWidth = 0.5 + 3.5 * t;
      ctx.beginPath();
      ctx.moveTo(pos[i][0], pos[i][1]);
      ctx.lineTo(pos[j][0], pos[j][1]);
      ctx.stroke();
      if (baseline && baseline[i] && baseline[i][j] != null && baseline[i][j] > -128) {
        const d = Math.abs(rssi - baseline[i][j]);
        if (d > 3) {                       // attenuation delta over baseline
          ctx.strokeStyle = acc;
          ctx.globalAlpha = 0.8;
          ctx.lineWidth = clamp(d / 6, 1, 6);
          ctx.beginPath();
          ctx.moveTo(pos[i][0], pos[i][1]);
          ctx.lineTo(pos[j][0], pos[j][1]);
          ctx.stroke();
        }
      }
    }
  ctx.globalAlpha = 1;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = i === (lastMesh.gateway ?? -1) ? acc : fg;
    ctx.fillRect(pos[i][0] - 5, pos[i][1] - 5, 10, 10);
    ctx.fillStyle = fg;
    ctx.fillText(String(i), pos[i][0] - 3, pos[i][1] - 9);
  }
}

async function doMeshFirmware() {
  msg("generating mesh firmware…");
  try {
    const d = await api("/api/mcu/firmware", {
      kind: "mesh",
      n_nodes: clamp(intVal("mcu-nodes", 4), 2, 12),
      gateway_id: clamp(intVal("mcu-gateway", 0), 0, 11),
    });
    fileLinks(document.getElementById("mcu-mesh-files"), d, "/api/mcu/file");
    msg(d.notes || "mesh firmware ready", "ok");
  } catch (e) {
    msg(`firmware failed: ${e.message}`, "error");
  }
}

async function doCollect() {
  const host = document.getElementById("mcu-mesh-host").value.trim();
  if (!host) { msg("give the gateway IP (e.g. 192.168.4.1)", "error"); return; }
  msg("collecting…");
  try {
    const d = await api("/api/mcu/mesh/collect", { host });
    lastMesh = d;
    const R = d.rssi_dbm || [];
    let links = 0;
    for (const row of R) for (const v of row) if (v != null && v > -128) links++;
    document.getElementById("mcu-mesh-stats").replaceChildren(
      el("tr", {}, el("td", { class: "k" }, "t"), el("td", { class: "v" }, d.t ?? "—")),
      el("tr", {}, el("td", { class: "k" }, "nodes"), el("td", { class: "v" }, d.n ?? R.length)),
      el("tr", {}, el("td", { class: "k" }, "links heard"), el("td", { class: "v" }, links)),
      el("tr", {}, el("td", { class: "k" }, "baseline"), el("td", { class: "v" }, baseline ? "set" : "none")),
    );
    drawMesh();
    msg(`mesh: ${d.n ?? "?"} nodes, ${links} directed links`, "ok");
  } catch (e) {
    msg(`collect failed: ${e.message}`, "error");
  }
}

function doCalibrate() {
  if (!lastMesh || !lastMesh.rssi_dbm) { msg("collect first, then calibrate", "error"); return; }
  baseline = lastMesh.rssi_dbm.map((row) => row.slice());
  drawMesh();
  msg("baseline stored — delta links now drawn in accent", "ok");
}

function doExportMesh() {
  if (!lastMesh) { msg("nothing collected yet", "error"); return; }
  const blob = new Blob([JSON.stringify(lastMesh, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "mesh_field.json" });
  a.click();
  URL.revokeObjectURL(a.href);
  msg("mesh_field.json downloaded (spatialxr handoff)", "ok");
}

// ---- EDGE layer --------------------------------------------------------------

const TARGET_NOTES = {
  circuitpython: "single .py — pure Python + math, no NumPy; ulab optional. Drop on the CIRCUITPY drive.",
  rust_no_std: "#![no_std] lib crate — caller-provided buffers, no alloc; fBm pulls libm only.",
  c_baremetal: ".c/.h pair — integer-only kernels, no heap; fBm uses floorf from math.h.",
};

async function doExport() {
  const engine = document.getElementById("mcu-engine").value;
  const target = document.getElementById("mcu-target").value;
  msg("exporting…");
  try {
    const d = await api("/api/mcu/export", { engine, target });
    fileLinks(document.getElementById("mcu-edge-files"), d, "/api/mcu/file");
    msg(`export: ${(d.files || []).join(", ")}`, "ok");
  } catch (e) {
    msg(`export failed: ${e.message}`, "error");
  }
}

// ---- init --------------------------------------------------------------------

function onThemechange() { drawLed(); drawMesh(); }

export function init(container) {
  layerBtns = {};
  frames = [];
  curFrame = 0;
  lastMesh = null;
  baseline = null;
  stopPlay();

  const layerRow = el(
    "div", { class: "btn-row layer-select", id: "mcu-layer-select" },
    ...Object.entries(LAYERS).map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name }, `[${label}]`);
      b.addEventListener("click", () => selectLayer(name));
      layerBtns[name] = b;
      return b;
    })
  );

  // ---- LED sidebar
  ledCanvas = el("canvas", { class: "sim-canvas", width: "384", height: "384" });
  const swatchRow = el("div", { class: "btn-row" },
    ...SWATCHES.map(([hex, name]) => {
      const b = el("button", {
        class: "btn btn-xs", title: name,
        style: `background:${hex};width:20px;height:20px;min-width:20px;padding:0;` +
          `border:1px solid ${hex === paint ? "var(--accent)" : "#6a6e74"}`,
      });
      b.addEventListener("click", () => {
        paint = hex;
        eraseMode = false;
        swatchRow.childNodes.forEach((s) => (s.style.border = "1px solid #6a6e74"));
        b.style.border = "1px solid var(--accent)";
      });
      return b;
    }));
  const eraseBtn = el("button", { class: "btn btn-xs" }, "[ERASE]");
  eraseBtn.addEventListener("click", () => {
    eraseMode = !eraseMode;
    eraseBtn.style.opacity = eraseMode ? "1" : "0.45";
  });
  eraseBtn.style.opacity = "0.45";

  ledNotesEl = el("div", { class: "dim" }, "");
  ledFilesEl = el("div", {}, el("div", { class: "dim" }, "no firmware yet"));

  const ledSide = el("div", { class: "panel" },
    el("h2", {}, "Matrix"),
    el("div", { class: "row" }, el("label", {}, "width"), el("input", { id: "mcu-w", type: "number", value: "16", min: "1", max: "64" })),
    el("div", { class: "row" }, el("label", {}, "height"), el("input", { id: "mcu-h", type: "number", value: "16", min: "1", max: "64" })),
    el("div", { class: "row" }, el("label", {}, "board"),
      el("select", { id: "mcu-board" },
        el("option", { value: "esp32" }, "ESP32"),
        el("option", { value: "teensy" }, "TEENSY"),
        el("option", { value: "circuitpython" }, "CIRCUITPY"))),
    el("div", { class: "row" }, el("label", {}, "data pin"), el("input", { id: "mcu-pin", type: "number", placeholder: "(default)", min: "0", max: "48" })),
    el("div", { class: "row" }, el("label", {}, "ssid"), el("input", { id: "mcu-ssid", type: "text", placeholder: "(AP mode)", style: "flex:1" })),
    el("div", { class: "row" }, el("label", {}, "password"), el("input", { id: "mcu-pass", type: "text", style: "flex:1" })),
    el("div", { class: "row" }, el("label", {}, "serpentine"),
      el("input", { id: "mcu-serpentine", type: "checkbox", checked: true })),
    el("div", { class: "row" }, el("label", {}, "brightness"), el("input", { id: "mcu-brightness", type: "number", value: "64", min: "0", max: "255" })),
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "mcu-fw" }, "Firmware")),
    ledNotesEl,
    ledFilesEl,
    el("h2", {}, "Device"),
    el("div", { class: "row" }, el("label", {}, "host"),
      el("input", { id: "mcu-host", type: "text", placeholder: "192.168.4.1", style: "flex:1" }),
      el("button", { class: "btn", id: "mcu-send" }, "Send Frame")),
  );

  const ledMain = el("div", { class: "panel" },
    el("h2", {}, "Paint"),
    el("div", { class: "panel-row" },
      el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "frame editor — drag paints, right-click erases"), ledCanvas)),
    swatchRow,
    el("div", { class: "btn-row" },
      eraseBtn,
      el("button", { class: "btn btn-xs", id: "mcu-add" }, "[ADD FRAME]"),
      el("button", { class: "btn btn-xs", id: "mcu-del" }, "[DEL]"),
      el("button", { class: "btn btn-xs", id: "mcu-play" }, "[PLAY]"),
      el("label", {}, "delay ms"),
      el("input", { id: "mcu-delay", type: "number", value: "200", min: "30", max: "5000", step: "10", style: "width:70px" })),
    el("div", { class: "btn-row", id: "mcu-frames" }),
  );

  // ---- MESH panels
  meshCanvas = el("canvas", { class: "sim-canvas", width: "384", height: "384" });
  const meshSide = el("div", { class: "panel" },
    el("h2", {}, "Mesh field"),
    el("div", { class: "row" }, el("label", {}, "nodes"), el("input", { id: "mcu-nodes", type: "number", value: "4", min: "2", max: "12" })),
    el("div", { class: "row" }, el("label", {}, "gateway id"), el("input", { id: "mcu-gateway", type: "number", value: "0", min: "0", max: "11" })),
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "mcu-mesh-fw" }, "Firmware")),
    el("div", { id: "mcu-mesh-files" }, el("div", { class: "dim" }, "no firmware yet")),
    el("h2", {}, "Gateway"),
    el("div", { class: "row" }, el("label", {}, "host"),
      el("input", { id: "mcu-mesh-host", type: "text", placeholder: "192.168.4.1", style: "flex:1" }),
      el("button", { class: "btn", id: "mcu-collect" }, "Collect")),
    el("div", { class: "btn-row" },
      el("button", { class: "btn btn-xs", id: "mcu-cal" }, "[CALIBRATE]"),
      el("button", { class: "btn btn-xs", id: "mcu-export-mesh" }, "[EXPORT JSON]")),
    el("table", { class: "stats" }, el("tbody", { id: "mcu-mesh-stats" },
      el("tr", {}, el("td", { class: "dim" }, "collect to fill")))),
    el("div", { class: "dim" },
      "ALPHA: RSSI tomography is coarse — dBm-quantized, multipath-",
      el("br"),
      "dominated, no CSI. Occupancy hint, not imaging."),
  );
  const meshMain = el("div", { class: "panel" },
    el("h2", {}, "Field"),
    el("div", { class: "panel-row" },
      el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "link strength ∝ (rssi+100)/60 — accent = delta vs baseline"), meshCanvas)),
  );

  // ---- EDGE panels
  const engineSel = el("select", { id: "mcu-engine" },
    el("option", { value: "hadamard_core" }, "HADAMARD CORE"),
    el("option", { value: "flux_map" }, "FLUX MAP"),
    el("option", { value: "terrain_fbm" }, "TERRAIN FBM"));
  const targetSel = el("select", { id: "mcu-target" },
    el("option", { value: "circuitpython" }, "CIRCUITPYTHON"),
    el("option", { value: "rust_no_std" }, "RUST NO_STD"),
    el("option", { value: "c_baremetal" }, "C BARE METAL"));
  const targetNote = el("div", { class: "dim" }, TARGET_NOTES.circuitpython);
  targetSel.addEventListener("change", () => {
    targetNote.textContent = TARGET_NOTES[targetSel.value];
  });
  const edgePanel = el("div", { class: "panel" },
    el("h2", {}, "Edge engine export"),
    el("div", { class: "row" }, el("label", {}, "engine"), engineSel),
    el("div", { class: "row" }, el("label", {}, "target"), targetSel),
    targetNote,
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "mcu-export" }, "Export")),
    el("div", { id: "mcu-edge-files" }, el("div", { class: "dim" }, "no export yet")),
    el("div", { class: "dim" },
      "Templates port hoa64.mcu's py_* kernels — bitset Sylvester +",
      " integer descent, flux tiles, integer-hash fBm."),
  );

  const head = el("div", { class: "panel" },
    el("h2", {}, "MICROCONTROLLER LAB"),
    layerRow,
    (msgEl = el("div", { class: "msg" })),
    el("div", { class: "dim" }, "WS2812 frames · ESP-NOW mesh field · no-NumPy edge kernels"));

  // LED/MESH keep the lab's sidebar+viewport split; EDGE lives in the
  // viewport column (its form is small enough to be the whole layer).
  const sideBlocks = { led: ledSide, mesh: meshSide };
  const mainBlocks = { led: ledMain, mesh: meshMain, edge: edgePanel };
  selectLayer = (name) => {
    layer = name;
    for (const [k, b] of Object.entries(layerBtns)) b.style.opacity = k === layer ? "1" : "0.45";
    for (const [k, p] of Object.entries(sideBlocks)) p.classList.toggle("hidden", k !== layer);
    for (const [k, p] of Object.entries(mainBlocks)) p.classList.toggle("hidden", k !== layer);
    if (name !== "led") stopPlay();
  };

  container.replaceChildren(el("div", { class: "lab ant-cap" },
    el("div", {}, head, ledSide, meshSide),
    el("div", {}, ledMain, meshMain, edgePanel)));

  selectLayer("led");

  // LED wiring
  ledCanvas.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    painting = true;
    if (ev.button === 2) strokeErase = true; // right-click erases this stroke
    paintAt(ev);
  });
  ledCanvas.addEventListener("mousemove", (ev) => { if (painting) paintAt(ev); });
  window.addEventListener("mouseup", onMouseUp);
  ledCanvas.addEventListener("contextmenu", (ev) => ev.preventDefault());
  document.getElementById("mcu-w").addEventListener("change", resetFrames);
  document.getElementById("mcu-h").addEventListener("change", resetFrames);
  document.getElementById("mcu-add").addEventListener("click", () => {
    frames.push(frames[curFrame].map((px) => px.slice()));
    curFrame = frames.length - 1;
    renderFrameList();
    drawLed();
  });
  document.getElementById("mcu-del").addEventListener("click", () => {
    if (frames.length <= 1) { msg("cannot delete the last frame", "error"); return; }
    frames.splice(curFrame, 1);
    curFrame = Math.min(curFrame, frames.length - 1);
    renderFrameList();
    drawLed();
  });
  document.getElementById("mcu-play").addEventListener("click", togglePlay);
  document.getElementById("mcu-fw").addEventListener("click", doLedFirmware);
  document.getElementById("mcu-send").addEventListener("click", doSendFrame);

  // MESH wiring
  document.getElementById("mcu-mesh-fw").addEventListener("click", doMeshFirmware);
  document.getElementById("mcu-collect").addEventListener("click", doCollect);
  document.getElementById("mcu-cal").addEventListener("click", doCalibrate);
  document.getElementById("mcu-export-mesh").addEventListener("click", doExportMesh);

  // EDGE wiring
  document.getElementById("mcu-export").addEventListener("click", doExport);

  resetFrames();
  drawMesh();
  window.addEventListener("themechange", onThemechange);
}

function onMouseUp() {
  painting = false;
  strokeErase = false;
}

export function deactivate() {
  stopPlay();
  window.removeEventListener("mouseup", onMouseUp);
  window.removeEventListener("themechange", onThemechange);
}
