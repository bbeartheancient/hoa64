// Library — construction-DAG map of all Hadamard orders (Phase 5b).
// GET /api/dag?max=N → built/claimed/gap tiers, method glyphs, Kronecker
// depths (from game_of_hadamard.classify_orders). Hand-rolled canvas
// renderer: 16 cells per row, one cell per valid order (1, 2, 4k);
// built = solid fg block, claimed = dim outline, gap = faint bracket.
// Selecting a cell highlights its Kronecker relatives (×2/×4/×8
// descendants, /2 ancestor chain through built orders) and offers deep
// links into Matrix Lab / Search Studio via "hoa64:open-tab".
// Bottom strip: /api/detbounds scatter — achieved log₁₀ det vs the
// Hadamard bound ½n log₁₀ n.

import { themeColor } from "/js/theme.js";

const COLS = 16;
const CELL = 52;
const LABEL_H = 15;
const PAD = 8;
const GAP_CHIPS = 20;

let dagData = null; // last /api/dag response
let detData = null; // last /api/detbounds response
let orders = []; // valid orders [1, 2, 4, 8, ...] ≤ max
let builtSet = new Set();
let claimedSet = new Set();
let selected = null;
let relatives = new Set(); // Kronecker descendants/ancestors of `selected`
let dagCanvas, detCanvas, msgEl, countsEl, selInfoEl, gapsEl;
let openBtn, searchBtn;

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

async function apiGet(path) {
  const r = await fetch(path);
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

// ---- construction map -------------------------------------------------------

function cellRect(idx) {
  return {
    x: PAD + (idx % COLS) * CELL,
    y: PAD + Math.floor(idx / COLS) * (CELL + LABEL_H),
  };
}

function stateOf(n) {
  if (builtSet.has(n)) return "built";
  if (claimedSet.has(n)) return "claimed";
  return "gap";
}

function drawDag() {
  if (!dagData || !dagCanvas) return;
  const rows = Math.ceil(orders.length / COLS);
  dagCanvas.width = COLS * CELL + 2 * PAD;
  dagCanvas.height = rows * (CELL + LABEL_H) + 2 * PAD;
  const ctx = dagCanvas.getContext("2d");
  const fg = themeColor("fg");
  const bg = themeColor("bg");
  const dim = themeColor("dim");
  const faint = themeColor("faint");
  const accent = themeColor("accent");

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, dagCanvas.width, dagCanvas.height);

  for (let idx = 0; idx < orders.length; idx++) {
    const n = orders[idx];
    const { x, y } = cellRect(idx);
    const state = builtSet.has(n) ? "built" : claimedSet.has(n) ? "claimed" : "gap";
    const s = CELL - 6;

    // Kronecker relatives of the selection get an accent wash first
    if (relatives.has(n)) {
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = accent;
      ctx.fillRect(x - 1, y - 1, s + 2, s + 2);
      ctx.globalAlpha = 1.0;
    }

    if (state === "built") {
      ctx.fillStyle = fg;
      ctx.fillRect(x, y, s, s);
    } else if (state === "claimed") {
      ctx.strokeStyle = dim;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, s - 2, s - 2);
    } else {
      // gap: faint corner bracket
      const b = Math.floor(s * 0.35);
      ctx.strokeStyle = faint;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + (s - b) / 2, y + (s - b) / 2, b, b);
    }

    if (n === selected) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 3, y - 3, s + 6, s + 6);
    }

    // order number (top-left) + method glyphs (under the cell)
    ctx.fillStyle = state === "gap" ? faint : dim;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(String(n), x + 2, y + 9);
    const glyphs = (dagData.labels[String(n)] || []).join("");
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = state === "built" ? fg : dim;
    ctx.fillText(glyphs || (state === "gap" ? "·" : ""), x + s / 2, y + s + 12);
  }
}

function onDagClick(e) {
  if (!dagData) return;
  const rect = dagCanvas.getBoundingClientRect();
  const px = ((e.clientX - rect.left) / rect.width) * dagCanvas.width;
  const py = ((e.clientY - rect.top) / rect.height) * dagCanvas.height;
  const col = Math.floor((px - PAD) / CELL);
  const row = Math.floor((py - PAD) / (CELL + LABEL_H));
  if (col < 0 || col >= COLS || row < 0) return;
  const idx = row * COLS + col;
  if (idx >= 0 && idx < orders.length) selectOrder(orders[idx]);
}

function selectOrder(n) {
  selected = n;
  // Kronecker relatives: ×2/×4/×8 descendants (within the loaded range),
  // and the /2 ancestor chain while the parent is built
  relatives = new Set();
  const maxN = dagData.max;
  for (const k of [2, 4, 8]) {
    if (n * k <= maxN && (n * k <= 2 || (n * k) % 4 === 0)) relatives.add(n * k);
  }
  let a = n;
  while (a % 2 === 0 && builtSet.has(a / 2)) {
    a /= 2;
    relatives.add(a);
  }
  relatives.delete(n);

  const state = stateOf(n);
  const glyphs = (dagData.labels[String(n)] || []).join(" ") || "—";
  const depth = dagData.depths[String(n)];
  selInfoEl.textContent =
    `H(${n}) — ${state} · methods ${glyphs} · depth ${depth ?? "?"}` +
    (relatives.size ? ` · ${relatives.size} Kronecker relatives` : "");
  openBtn.disabled = state !== "built";
  searchBtn.disabled = state !== "gap";
  drawDag();
}

// ---- gap chips --------------------------------------------------------------

function showGaps() {
  const gaps = dagData.gaps.slice(0, GAP_CHIPS);
  if (!gaps.length) {
    gapsEl.replaceChildren(el("div", { class: "status-line" }, `no gaps ≤ ${dagData.max}`));
    return;
  }
  gapsEl.replaceChildren(
    ...gaps.map((n) => {
      const chip = el("button", { class: "btn btn-xs" }, String(n));
      chip.addEventListener("click", () => selectOrder(n));
      return chip;
    })
  );
}

// ---- det-bound chart ----------------------------------------------------------

function drawDet() {
  if (!detData || !detCanvas) return;
  const W = 720;
  const H = 210;
  detCanvas.width = W;
  detCanvas.height = H;
  const ctx = detCanvas.getContext("2d");
  const fg = themeColor("fg");
  const bg = themeColor("bg");
  const dim = themeColor("dim");
  const faint = themeColor("faint");
  const entries = detData.entries;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  if (!entries.length) return;

  const yMax = Math.max(...entries.map((e) => e.det_bound_log10)) * 1.06;
  const xMax = entries[entries.length - 1].order;
  const L = 44;
  const B = 20;
  const px = (n) => L + ((n / xMax) * (W - L - 8));
  const py = (v) => H - B - (v / yMax) * (H - B - 8);

  // axes + caption
  ctx.strokeStyle = faint;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(L, 4);
  ctx.lineTo(L, H - B);
  ctx.lineTo(W - 8, H - B);
  ctx.stroke();
  ctx.fillStyle = dim;
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`log10 det — bound ½n·log10(n)`, L + 4, 14);
  ctx.textAlign = "right";
  ctx.fillText(yMax.toFixed(0), L - 4, 14);
  ctx.fillText(String(xMax), W - 10, H - 6);

  // bound line (all entries) then achieved points (det computed ≤ 256)
  ctx.strokeStyle = dim;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const [i, e] of entries.entries()) {
    const X = px(e.order);
    const Y = py(e.det_bound_log10);
    if (i === 0) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  }
  ctx.stroke();
  ctx.fillStyle = fg;
  let drawn = 0;
  for (const e of entries) {
    if (e.det_log10 === null || e.det_log10 === undefined) continue;
    ctx.fillRect(px(e.order) - 2, py(e.det_log10) - 2, 4, 4);
    drawn++;
  }
  ctx.fillStyle = dim;
  ctx.textAlign = "left";
  ctx.fillText(`■ achieved (${drawn} orders ≤ 256)   — bound`, L + 4, H - 6);
}

// ---- data ---------------------------------------------------------------------

async function doLoad() {
  const m = parseInt(document.getElementById("lib-max").value, 10);
  msg("classifying orders… (first call scans the toolchain)");
  try {
    dagData = await apiGet(`/api/dag?max=${m}`);
    builtSet = new Set(dagData.built);
    claimedSet = new Set(dagData.claimed);
    orders = [1, 2];
    for (let n = 4; n <= m; n += 4) orders.push(n);
    const firstGap = dagData.gaps.length ? dagData.gaps[0] : "—";
    countsEl.textContent =
      `built ${dagData.built.length} · claimed ${dagData.claimed.length} · ` +
      `gaps ${dagData.gaps.length} · first gap ${firstGap}`;
    selected = null;
    relatives = new Set();
    selInfoEl.textContent = "click a cell";
    openBtn.disabled = true;
    searchBtn.disabled = true;
    drawDag();
    showGaps();
    msg(`classified orders ≤ ${m}`, "ok");
  } catch (e) {
    msg(`dag failed: ${e.message}`, "error");
  }
  try {
    detData = await apiGet(`/api/detbounds?max=${m}`);
    drawDet();
  } catch (e) {
    msg(`detbounds failed: ${e.message}`, "error");
  }
}

function onTheme() {
  drawDag();
  drawDet();
}

// ---- tab lifecycle --------------------------------------------------------------

export function init(container) {
  dagData = detData = null;
  selected = null;
  relatives = new Set();

  const controlsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Construction map"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "max"),
      el(
        "select",
        { id: "lib-max" },
        ...[128, 256, 400, 800].map((v) =>
          el("option", { value: String(v), ...(v === 400 ? { selected: "" } : {}) }, String(v))
        )
      ),
      el("button", { class: "btn", id: "lib-load" }, "Load")
    ),
    (countsEl = el("div", { class: "status-line" }, "—")),
    (msgEl = el("div", { class: "msg" }))
  );

  const selPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Selected order"),
    (selInfoEl = el("div", { class: "status-line" }, "click a cell")),
    el(
      "div",
      { class: "btn-row" },
      (openBtn = el("button", { class: "btn", disabled: true }, "Open in Matrix Lab")),
      (searchBtn = el("button", { class: "btn", disabled: true }, "Launch search"))
    )
  );

  const gapsPanel = el("div", { class: "panel" }, el("h2", {}, "Gaps"), (gapsEl = el("div", { class: "btn-row" })));

  dagCanvas = el("canvas", { style: "width:100%;image-rendering:pixelated;border:1px solid var(--panel-line)" });
  detCanvas = el("canvas", { style: "width:100%;border:1px solid var(--panel-line)" });
  const mapPanel = el("div", { class: "panel" }, el("h2", {}, "Orders (16 per row)"), dagCanvas);
  const detPanel = el("div", { class: "panel" }, el("h2", {}, "Determinant vs Hadamard bound"), detCanvas);

  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, controlsPanel, selPanel, gapsPanel),
      el("div", {}, detPanel, mapPanel)
    )
  );

  dagCanvas.addEventListener("click", onDagClick);
  document.getElementById("lib-load").addEventListener("click", doLoad);
  openBtn.addEventListener("click", () => {
    if (selected)
      window.dispatchEvent(
        new CustomEvent("hoa64:open-tab", { detail: { tab: "matrix_lab", order: selected } })
      );
  });
  searchBtn.addEventListener("click", () => {
    if (selected)
      window.dispatchEvent(
        new CustomEvent("hoa64:open-tab", { detail: { tab: "search_studio", order: selected } })
      );
  });
  window.addEventListener("themechange", onTheme);
  doLoad();
}

export function deactivate() {
  window.removeEventListener("themechange", onTheme);
  dagData = detData = null;
}
