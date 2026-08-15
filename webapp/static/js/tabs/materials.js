// Materials Lab — physical homes for the H.8 flux-tile catalog.
//   CLOTH   ternary flux yarn: P=2W−1 → + / 0 / −  (not H=±1)
//   TOUCH   mutual-cap on unlike-P bonds; three electrode families
//   META    full flux sheet + H.8 atom lattice
// POST /api/materials/design  → preview + stats
// POST /api/materials/kicad   → .kicad_mod / .kicad_pcb

import { themeColor } from "/js/theme.js";

const LAYERS = { cloth: "CLOTH", touchpad: "TOUCH", metamaterial: "META" };

let msgEl, layerBtns = {}, panels = {};
let layer = "cloth";
let last = null;
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
  return {
    kind: layer,
    order: parseInt(document.getElementById("mat-order").value, 10),
    start: document.getElementById("mat-start").value,
    pitch_mm: numVal("mat-pitch", 1.0),
  };
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
  host.replaceChildren(...rows);
  renderKey(d.key);
}

function swatchColor(name) {
  if (name === "fg") return themeColor("fg");
  if (name === "accent") return themeColor("accent") || themeColor("dim");
  return themeColor("dim");
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
        style: `display:inline-block;width:12px;height:12px;margin-right:8px;vertical-align:middle;background:${swatchColor(r.swatch || (r.layer === "B.Cu" ? "accent" : "fg"))};border:1px solid ${themeColor("dim")}`,
      });
      kids.push(el("div", { class: "row" }, sw,
        el("span", {}, `${r.name}  ${r.means}`)));
    }
  };
  block("FILL — flux tile P = 2W−1", key.fill);
  block("COPPER — 2-layer stack", key.copper);
  if (key.feeds && key.feeds.length) block("MARKS — electrode feeds", key.feeds);
  host.replaceChildren(...kids);
}

function drawPreview() {
  if (!previewCanvas) return;
  const ctx = previewCanvas.getContext("2d");
  const W = previewCanvas.width, H = previewCanvas.height;
  ctx.fillStyle = themeColor("bg");
  ctx.fillRect(0, 0, W, H);
  const prev = last && last.preview;
  if (!prev || !prev.prims || !prev.prims.length) {
    ctx.fillStyle = themeColor("dim");
    ctx.font = "11px monospace";
    ctx.fillText("generate a layout", 8, H / 2);
    return;
  }
  const b = prev.bbox;
  const span = Math.max(b.xmax - b.xmin, b.ymax - b.ymin, 1e-3);
  const sc = (Math.min(W, H) - 36) / span;
  const x0 = (W - (b.xmax - b.xmin) * sc) / 2 - b.xmin * sc;
  const y0 = (H - (b.ymax - b.ymin) * sc) / 2 - b.ymin * sc;
  const X = (x) => x0 + x * sc;
  const Y = (y) => H - (y0 + y * sc);
  const col = (p) => {
    // ternary flux: + = fg, 0 = dim (olive in the capture), − = accent
    if (p.polarity === 1) return themeColor("fg");
    if (p.polarity === -1) return themeColor("accent") || themeColor("dim");
    if (p.polarity === 0) return themeColor("dim");
    if (p.layer && p.layer.includes("B.Cu")) return themeColor("accent") || themeColor("dim");
    if (p.layer && p.layer.includes("Silk")) return themeColor("dim");
    return themeColor("fg");
  };
  for (const p of prev.prims) {
    ctx.globalAlpha = p.polarity === 0 ? 0.7 : 0.92;
    ctx.fillStyle = col(p);
    ctx.strokeStyle = col(p);
    if (p.kind === "rect" && p.a && p.b) {
      ctx.fillRect(X(Math.min(p.a[0], p.b[0])), Y(Math.max(p.a[1], p.b[1])),
        Math.abs(p.b[0] - p.a[0]) * sc, Math.abs(p.b[1] - p.a[1]) * sc);
    } else if (p.kind === "pad" && p.c && p.role === "feed") {
      const r = Math.max(2.5, Math.min(p.size ? p.size[0] : 0.3, p.size ? p.size[1] : 0.3) * sc * 0.5);
      ctx.beginPath();
      ctx.arc(X(p.c[0]), Y(p.c[1]), r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = themeColor("bg");
      ctx.lineWidth = 1;
      ctx.stroke();
      if (p.name) {
        ctx.fillStyle = themeColor("fg");
        ctx.font = "9px monospace";
        ctx.fillText(p.name, X(p.c[0]) + r + 2, Y(p.c[1]) + 3);
      }
    } else if (p.kind === "pad" && p.c) {
      const sx = (p.size ? p.size[0] : 1) * sc, sy = (p.size ? p.size[1] : 1) * sc;
      ctx.fillRect(X(p.c[0]) - sx / 2, Y(p.c[1]) - sy / 2, sx, sy);
    } else if (p.kind === "circle" && p.c) {
      ctx.beginPath(); ctx.arc(X(p.c[0]), Y(p.c[1]), (p.r || 0) * sc, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  // map key drawn on the canvas (always visible with the tile)
  const key = (last && last.key && last.key.fill) || [
    { polarity: 1, swatch: "fg", name: "+" },
    { polarity: 0, swatch: "dim", name: "0" },
    { polarity: -1, swatch: "accent", name: "−" },
  ];
  const kx = 8, ky = H - 44;
  ctx.fillStyle = themeColor("bg");
  ctx.globalAlpha = 0.82;
  ctx.fillRect(kx - 4, ky - 12, 200, 40);
  ctx.globalAlpha = 1;
  ctx.font = "10px monospace";
  let x = kx;
  for (const row of key) {
    ctx.fillStyle = swatchColor(row.swatch);
    ctx.fillRect(x, ky, 10, 10);
    ctx.strokeStyle = themeColor("dim");
    ctx.strokeRect(x, ky, 10, 10);
    ctx.fillStyle = themeColor("fg");
    const label = row.polarity === 1 ? "+ W=1" : row.polarity === 0 ? "0 W=½" : "− W=0";
    ctx.fillText(label, x + 14, ky + 9);
    x += 66;
  }
  ctx.fillStyle = themeColor("dim");
  ctx.fillText("2-layer copper: F.Cu = H=+1 face · B.Cu = H=−1 reverse · 0 = gap",
    kx, ky + 24);
}

async function doDesign() {
  msg("laying out…");
  try {
    const d = await api("/api/materials/design", body());
    last = d;
    renderStats(d);
    drawPreview();
    const s = d.stats || {};
    msg(`${s.kind} n=${d.order}` + (s.n_caps != null ? ` · ${s.n_caps} caps` : "")
      + (s.n_atoms != null ? ` · ${s.n_atoms} atoms` : ""), "ok");
  } catch (e) {
    msg(`design failed: ${e.message}`, "error");
  }
}

async function doKicad() {
  const btn = document.getElementById("mat-kicad");
  btn.disabled = true;
  try {
    const d = await api("/api/materials/kicad", body());
    last = { ...(last || {}), preview: d.preview, stats: d.stats, tiles: d.tiles, key: d.key };
    renderStats(last);
    drawPreview();
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
    el("div", { class: "btn-row" },
      el("button", { class: "btn", id: "mat-run" }, "Generate"),
      el("button", { class: "btn", id: "mat-kicad" }, "Export KiCad"))
  );
  const view = el("div", { class: "panel" },
    el("h2", {}, "Layout"),
    el("div", { class: "panel-row" },
      el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, "flux tile (fill) on 2-layer copper"), previewCanvas)),
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
  drawPreview();
  window.addEventListener("themechange", onThemechange);
}

export function deactivate() {
  window.removeEventListener("themechange", onThemechange);
}
