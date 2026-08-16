// Materials Lab — physical homes for the H.8 flux-tile catalog.
//   CLOTH   ternary flux yarn: P=2W−1 → + / 0 / −  (not H=±1)
//   TOUCH   mutual-cap on unlike-P bonds; three electrode families
//   META    full flux sheet + H.8 atom lattice
// POST /api/materials/design  → preview + stats
// POST /api/materials/kicad   → .kicad_mod / .kicad_pcb

import { drawKicadPrims, KICAD_CU, KICAD_KEY } from "/js/kicad_layers.js";

const LAYERS = { cloth: "CLOTH", touchpad: "TOUCH", metamaterial: "META" };

let msgEl, layerBtns = {}, panels = {};
let layer = "cloth";
let last = null;
let lastKicadPreviewBoard = null, kicadPreviewMode = "footprint";
let previewCanvas;

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
  drawPreview();
}

function body() {
  const b = {
    kind: layer,
    order: parseInt(document.getElementById("mat-order").value, 10),
    start: document.getElementById("mat-start").value,
    pitch_mm: numVal("mat-pitch", 1.0),
    tile: parseInt(document.getElementById("mat-tile").value, 10),
  };
  const gap = parseFloat(document.getElementById("mat-gap").value);
  if (Number.isFinite(gap)) b.gap_frac = gap;
  return b;
}

function renderStats(d) {
  const host = document.getElementById("mat-stats");
  if (!host || !d) return;
  const s = d.stats || {}, t = d.tiles || {};
  const rows = [
    statRow("kind", s.kind || d.kind),
    statRow("order", d.order),
    statRow("stack", s.stack || "2-layer"),
  ];
  for (const [k, v] of Object.entries(s)) {
    if (["kind", "n", "stack", "field"].includes(k) || v == null || typeof v === "object") continue;
    rows.push(statRow(k, typeof v === "number" ? Number(v.toPrecision(4)) : v));
  }
  if (s.field) rows.push(statRow("field", s.field));
  if (t.n_tiles != null)
    rows.push(statRow("flux tiles", `${t.n_tiles} × ${t.tile}×${t.tile}`, t.kronecker_h8 ? "good" : ""));
  if (typeof t.h8_agree === "number")
    rows.push(statRow("H.8 agree", `${(t.h8_agree * 100).toFixed(1)}%`, t.kronecker_h8 ? "good" : ""));
  if (Array.isArray(t.counts))
    rows.push(statRow("tile counts", t.counts.join(" / ")));
  const bz = d.brillouin;
  if (bz && bz.fold_coherence != null) {
    rows.push(statRow("BZ fold", Number(bz.fold_coherence).toFixed(3),
      bz.fold_coherence > 0.8 ? "good" : ""));
    if (bz.Q != null) rows.push(statRow("BZ Q", Number(bz.Q).toFixed(0)));
    if (bz.CD != null) rows.push(statRow("weave CD", Number(bz.CD).toFixed(3)));
  }
  const sz = d.actual_size;
  if (sz && sz.L_cm != null) {
    rows.push(statRow("actual L", `${Number(sz.L_cm).toPrecision(3)} cm`));
    rows.push(statRow("actual pitch", `${Number(sz.pitch_mm).toPrecision(3)} mm`));
  }
  host.replaceChildren(...rows);
  renderKey(d.key);
}

function swatchColor(name, layer) {
  if (layer && KICAD_CU[layer]) return KICAD_CU[layer];
  if (name === "F.Cu" || name === "fg") return KICAD_CU["F.Cu"];
  if (name === "B.Cu" || name === "accent") return KICAD_CU["B.Cu"];
  if (name === "In1.Cu") return KICAD_CU["In1.Cu"];
  if (name === "In2.Cu") return KICAD_CU["In2.Cu"];
  return "transparent";
}

function renderKey(key) {
  const host = document.getElementById("mat-key");
  if (!host) return;
  if (!key) {
    host.replaceChildren(el("div", { class: "dim" }, "generate to fill the map key"));
    return;
  }
  const kids = [el("div", { class: "dim" }, key.stack || "2-layer")];
  const block = (title, rows) => {
    kids.push(el("div", { class: "sim-label", style: "margin-top:8px" }, title));
    for (const r of rows || []) {
      const sw = el("span", {
        style: `display:inline-block;width:12px;height:12px;margin-right:8px;vertical-align:middle;background:${swatchColor(r.swatch, r.layer)};border:1px solid #6a6e74`,
      });
      kids.push(el("div", { class: "row" }, sw,
        el("span", {}, `${r.name}  ${r.means}`)));
    }
  };
  block("COPPER — KiCad layers (not themed)", KICAD_KEY.filter((r) => r.layer));
  block("COPPER — 2-layer stack", key.copper);
  kids.push(el("div", { class: "row" },
    el("span", { style: "display:inline-block;width:12px;height:12px;margin-right:8px;border:1px solid #6a6e74" }),
    el("span", {}, "0 / gap  W=½ edge — unfilled dielectric")));
  if (key.feeds && key.feeds.length) block("MARKS — electrode feeds", key.feeds);
  host.replaceChildren(...kids);
}

function drawPreview() {
  if (!previewCanvas) return;
  const prev = (kicadPreviewMode === "board" && lastKicadPreviewBoard)
    ? lastKicadPreviewBoard : (last && last.preview);
  drawKicadPrims(previewCanvas, prev, { empty: "generate a layout" });
}

function syncKicadPreviewBtns() {
  const row = document.getElementById("mat-prev-toggle");
  if (row) row.classList.toggle("hidden", !lastKicadPreviewBoard);
  const fpBtn = document.getElementById("mat-prev-fp");
  if (fpBtn) fpBtn.style.opacity = kicadPreviewMode === "footprint" ? "1" : "0.45";
  const boardBtn = document.getElementById("mat-prev-board");
  if (boardBtn) boardBtn.style.opacity = kicadPreviewMode === "board" ? "1" : "0.45";
}

async function doDesign() {
  msg("laying out…");
  try {
    const d = await api("/api/materials/design", body());
    last = d;
    lastKicadPreviewBoard = null;
    kicadPreviewMode = "footprint";
    syncKicadPreviewBtns();
    renderStats(d);
    drawPreview();
    const s = d.stats || {};
    msg(`${s.kind} n=${d.order}` + (s.n_caps != null ? ` · ${s.n_caps} caps` : "")
      + (s.n_atoms != null ? ` · ${s.n_atoms} atoms` : ""), "ok");
  } catch (e) {
    msg(`design failed: ${e.message}`, "error");
  }
}

async function doActualSize() {
  const order = parseInt(document.getElementById("mat-order").value, 10);
  msg("Press 1980 actual size…");
  try {
    const d = await api(`/api/materials/actual-size?order=${order}`);
    const elp = document.getElementById("mat-pitch");
    if (elp && d.pitch_mm != null) elp.value = Number(d.pitch_mm).toPrecision(4);
    msg(
      `actual size: L=${Number(d.L_cm).toPrecision(3)} cm · T=${Number(d.T_K).toFixed(0)} K · pitch ${Number(d.pitch_mm).toPrecision(3)} mm`,
      "ok"
    );
    await doDesign();
  } catch (e) {
    msg(`actual size failed: ${e.message}`, "error");
  }
}

async function doKicad() {
  const btn = document.getElementById("mat-kicad");
  btn.disabled = true;
  try {
    const d = await api("/api/materials/kicad", body());
    last = { ...(last || {}), preview: d.preview, stats: d.stats, tiles: d.tiles, key: d.key };
    lastKicadPreviewBoard = d.preview_board || null;
    kicadPreviewMode = lastKicadPreviewBoard ? "board" : "footprint";
    renderStats(last);
    drawPreview();
    syncKicadPreviewBtns();
    const host = document.getElementById("mat-files");
    host.replaceChildren(
      ...(d.files || []).map((name) =>
        el("div", { class: "row" },
          el("a", { class: "btn btn-xs", href: `/api/materials/kicad/${d.token}/${name}`, download: name }, name))
      ),
      el("div", { class: "dim" }, "one-shot links — click to download")
    );
    msg(`kicad: ${(d.files || []).join(", ")}`, "ok");
  } catch (e) {
    msg(`kicad failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

function onThemechange() { drawPreview(); }

export function init(container) {
  layerBtns = {};
  const layerRow = el(
    "div", { class: "btn-row layer-select", id: "mat-layer-select" },
    ...Object.entries(LAYERS).map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name }, `[${label}]`);
      b.addEventListener("click", () => selectLayer(name));
      layerBtns[name] = b;
      return b;
    })
  );
  previewCanvas = el("canvas", { class: "sim-canvas kicad-canvas", width: "384", height: "384" });

  const head = el("div", { class: "panel" },
    el("h2", {}, "MATERIALS LAB"),
    layerRow,
    (msgEl = el("div", { class: "msg" })),
    el("div", { class: "dim" }, "H.8 flux tiles as cloth, touchpad electrodes, or a spin-ice cell")
  );
  const controls = el("div", { class: "panel" },
    el("h2", {}, "Source"),
    el("div", { class: "row" }, el("label", {}, "order"), el("input", { id: "mat-order", type: "number", value: "16", min: "4", step: "4" })),
    el("div", { class: "row" }, el("label", {}, "start"),
      el("select", { id: "mat-start" },
        el("option", { value: "sylvester" }, "SYLVESTER"),
        el("option", { value: "library" }, "LIBRARY"))),
    el("div", { class: "row" }, el("label", {}, "pitch mm"), el("input", { id: "mat-pitch", type: "number", value: "1.0", step: "0.1", min: "0.2" })),
    el("div", { class: "row" }, el("label", {}, "gap frac"), el("input", { id: "mat-gap", type: "number", step: "0.01", min: "0.05", max: "0.5", placeholder: "auto" })),
    el("div", { class: "row" }, el("label", {}, "flux tile"),
      el("select", { id: "mat-tile" },
        el("option", { value: "4" }, "4×4"),
        el("option", { value: "8", selected: "selected" }, "8×8"),
        el("option", { value: "16" }, "16×16"))),
    el("div", { class: "btn-row" },
      el("button", { class: "btn", id: "mat-run" }, "Generate"),
      el("button", { class: "btn", id: "mat-kicad" }, "Export KiCad"),
      el("button", { class: "btn", id: "mat-actual" }, "Actual size"))
  );
  const view = el("div", { class: "panel" },
    el("h2", {}, "Layout"),
    el("div", { class: "btn-row hidden", id: "mat-prev-toggle" },
      el("button", { class: "btn", id: "mat-prev-fp" }, "[FOOTPRINT]"),
      el("button", { class: "btn", id: "mat-prev-board" }, "[BOARD]")),
    el("div", { class: "panel-row" },
      el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "2-layer copper — red F.Cu / blue B.Cu (no B.Cu pour)"), previewCanvas)),
    el("h2", {}, "Map key"),
    el("div", { id: "mat-key" }, el("div", { class: "dim" }, "generate to fill"))
  );
  const stats = el("div", { class: "panel" },
    el("h2", {}, "Stats"),
    el("table", { class: "stats" }, el("tbody", { id: "mat-stats" },
      el("tr", {}, el("td", { class: "dim" }, "generate to fill"))))
  );
  const files = el("div", { class: "panel" },
    el("h2", {}, "Files"),
    el("div", { id: "mat-files" }, el("div", { class: "dim" }, "no export yet"))
  );

  container.replaceChildren(el("div", { class: "lab ant-cap" },
    el("div", {}, head, controls, stats, files),
    el("div", {}, view)));
  selectLayer("cloth");
  document.getElementById("mat-run").addEventListener("click", doDesign);
  document.getElementById("mat-kicad").addEventListener("click", doKicad);
  document.getElementById("mat-actual").addEventListener("click", doActualSize);
  document.getElementById("mat-prev-fp").addEventListener("click", () => {
    kicadPreviewMode = "footprint";
    syncKicadPreviewBtns();
    drawPreview();
  });
  document.getElementById("mat-prev-board").addEventListener("click", () => {
    if (!lastKicadPreviewBoard) return;
    kicadPreviewMode = "board";
    syncKicadPreviewBtns();
    drawPreview();
  });
  syncKicadPreviewBtns();
  drawPreview();
  window.addEventListener("themechange", onThemechange);
}

export function deactivate() {
  window.removeEventListener("themechange", onThemechange);
}
