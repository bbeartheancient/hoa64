// Micromag Sim — live micromagnetic annealing with energy-field view.
// POST /api/sim/micromag, then stream /ws/job/{id}. Frames:
//   {step,T,E,best_E,accepts,E_exch,E_dem,E_anis,elapsed_s}  (every 500 steps)
//   (+ E_goal / goal_agree when a library goal target is active)
//   {matrix_png_b64}  throttled ±1 preview
//   {field_png_b64, grad_png_b64, flux_png_b64, z_png_b64}  energy / |dF| / walls / |Z|
//   {"type":"end"}  terminal
// Live retune: {"op":"set",cooling,lam_ex,lam_ani,lam_goal,lam_tile,lam_z} → job.params["live"].
// Unified visualizers (Item 4): ONE matrix canvas + ONE field canvas with a
// [MATRIX][ENERGY][GRAD][FLUX][GERZON] layer selector. Each WS PNG is cached
// pristine (server-green) per layer; the visible field canvas composites the
// selected layer and is retinted (theme LUT lives only there). The FLUX
// layer keeps its animated FLUX_FRAG flow underlay — paused and hidden when
// another layer is selected; the electric ELECTRIC_FRAG background stays
// behind the field canvas throughout.

import { connect } from "/js/ws.js";
import { makeStripChart } from "/js/viz/stripchart.js";
import { retintCanvas, fillPlusOne, themeColor } from "/js/theme.js";
import { ELECTRIC_FRAG, FLUX_FRAG, makeShaderCanvas, hexToRgb01 } from "/js/viz/shaders.js";
import { SIM_ALGORITHMS } from "/js/algorithms.js";

let msgEl, statusEl, statsEl, cancelBtn, exportBtn;
let matrixCanvas, fieldCanvas, fieldLabel;
let waveChart; // unified waveform strip chart (E/E_DEM/E_EXCH/E_ANIS/T)
let ws = null;
let currentJob = null;
let fxCanvas, fxShader, fluxShader, fluxShaderCanvas, fxRaf = null; // bg layers
let firstE = null, latestE = null, lastFluxTiles = null, fluxInfoEl;
let lastGerzon = null, gerzonInfoEl;
let layer = "energy"; // active field layer: matrix | energy | grad | flux | gerzon
const layerCache = {}; // name → {canvas, ctx, has} pristine PNG per layer
let layerBtns = {};

const LAYER_LABELS = { matrix: "MATRIX", energy: "ENERGY", grad: "GRAD", flux: "FLUX", gerzon: "GERZON" };
const LAYER_TITLES = {
  matrix: "matrix (best)",
  energy: "site energy",
  grad: "|dF| gradient",
  flux: "flux (walls)",
  gerzon: "Gerzon |Z| walls",
};

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

function drawPng(canvas, b64, after = null) {
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (after) after(); // capture pristine pixels before retinting
    retintCanvas(canvas);
  };
  img.src = `data:image/png;base64,${b64}`;
}

// cache an incoming layer PNG pristine (no retint — the theme LUT is
// applied only when compositing onto the visible field canvas)
function cacheDraw(name, b64) {
  const c = layerCache[name];
  if (!c) return;
  const img = new Image();
  img.onload = () => {
    c.ctx.imageSmoothingEnabled = false;
    c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height);
    c.ctx.drawImage(img, 0, 0, c.canvas.width, c.canvas.height);
    c.has = true;
    if (name === "energy") updateFlowTexture();
    if (name === layer) showLayer();
  };
  img.src = `data:image/png;base64,${b64}`;
}

function showLayer() {
  if (!fieldCanvas) return;
  const ctx = fieldCanvas.getContext("2d");
  const c = layerCache[layer];
  ctx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height);
  if (c && c.has) ctx.drawImage(c.canvas, 0, 0, fieldCanvas.width, fieldCanvas.height);
  retintCanvas(fieldCanvas);
  if (fieldLabel) fieldLabel.textContent = LAYER_TITLES[layer];
  // flux flow underlay only while the FLUX layer is up
  const fluxOn = layer === "flux";
  if (fluxShaderCanvas) fluxShaderCanvas.style.display = fluxOn ? "" : "none";
  fieldCanvas.style.opacity = fluxOn ? "0.8" : "0.88";
}

function selectLayer(name) {
  layer = name;
  for (const [k, b] of Object.entries(layerBtns)) {
    b.style.opacity = k === layer ? "1" : "0.45";
  }
  showLayer();
}

function renderFluxTiles(ft) {
  if (!fluxInfoEl) return;
  if (!ft) {
    fluxInfoEl.replaceChildren(statRow("flux tiles", "start a run, or [READ TILES]"));
    return;
  }
  const rows = [];
  if (ft.n_tiles != null)
    rows.push(statRow("unique tiles", `${ft.n_tiles} × ${ft.tile}×${ft.tile}  /  ${ft.n_blocks} blocks`,
      ft.kronecker_h8 ? "good" : ""));
  if (typeof ft.h8_agree === "number")
    rows.push(statRow("H.8 agree", `${(ft.h8_agree * 100).toFixed(1)}%`,
      ft.kronecker_h8 ? "good" : ""));
  if (Array.isArray(ft.counts))
    rows.push(statRow("tile counts", ft.counts.join(" / ")));
  if (ft.scales)
    rows.push(statRow("scales", Object.entries(ft.scales).map(([s, k]) => `${k}×${s}`).join("  ")));
  if (ft.nested) rows.push(statRow("nested", "top-left = H.n/2 flux", "good"));
  if (ft.mean_w != null) rows.push(statRow("mean W", Number(ft.mean_w).toFixed(3)));
  if (ft.n_tiles == null)
    rows.push(statRow("flux tiles", `n=${ft.n} not divisible by ${ft.tile}`));
  fluxInfoEl.replaceChildren(...(rows.length ? rows : [statRow("flux tiles", "—")]));
}

function renderGerzon(gz) {
  if (!gerzonInfoEl) return;
  if (!gz) {
    gerzonInfoEl.replaceChildren(statRow("gerzon", "start a run, or [READ GERZON]"));
    return;
  }
  const a = gz.aligned || {};
  const o = gz.overlap || {};
  const rows = [];
  if (a.n_cells != null)
    rows.push(statRow("aligned H₂", `${a.n_h2}/${a.n_cells}`,
      a.n_h2 === a.n_cells ? "good" : ""));
  if (a.n_wall != null) rows.push(statRow("aligned walls", a.n_wall));
  if (a.n_cohesive != null) rows.push(statRow("aligned cohesive", a.n_cohesive));
  if (o.n_wall != null)
    rows.push(statRow("overlap walls", `${o.n_wall}/${o.n_cells}`));
  if (gz.E_z != null)
    rows.push(statRow("E_z", Number(gz.E_z).toPrecision(4),
      Number(gz.E_z) === 0 ? "good" : ""));
  gerzonInfoEl.replaceChildren(...(rows.length ? rows : [statRow("gerzon", "—")]));
}

function resetRun() {
  waveChart.clear();
  statsEl.replaceChildren();
  statusEl.textContent = "connecting…";
  exportBtn.style.display = "none";
  firstE = null;
  latestE = null;
  lastFluxTiles = null;
  lastGerzon = null;
  renderFluxTiles(null);
  renderGerzon(null);
  for (const c of Object.values(layerCache)) c.has = false; // stale layers drop
  showLayer();
  // both viewports start as an all-+1 field (themed bright end), never
  // blank or stale from the previous run; real frames overwrite them via
  // drawPng / cacheDraw → showLayer
  fillPlusOne(matrixCanvas);
  fillPlusOne(fieldCanvas);
}

// electric-field background: RAF loop feeding uEnergy from the WS stream
function fxTick(t) {
  if (!fxCanvas || !fxCanvas.isConnected) {
    fxRaf = null;
    return;
  }
  const e =
    firstE !== null && latestE !== null && firstE > 0
      ? Math.min(1, latestE / firstE)
      : 0.15;
  fxShader.setUniform("uEnergy", e);
  fxShader.render(t / 1000);
  if (fluxShader && layer === "flux") {
    // flux flow underlay renders only while the FLUX layer is selected
    fluxShader.setUniform("uEnergy", e);
    fluxShader.render(t / 1000);
  }
  fxRaf = requestAnimationFrame(fxTick);
}

function fxTheme() {
  if (!fxShader) return;
  fxShader.setUniform("uFg", hexToRgb01(themeColor("fg")));
  fxShader.setUniform("uBg", hexToRgb01(themeColor("bg")));
  if (fluxShader) {
    fluxShader.setUniform("uFg", hexToRgb01(themeColor("fg")));
    fluxShader.setUniform("uBg", hexToRgb01(themeColor("bg")));
  }
}

function initFx(fieldCell) {
  fxCanvas = el("canvas", { class: "sim-fx", width: "128", height: "128" });
  fxShader = makeShaderCanvas(fxCanvas, ELECTRIC_FRAG, {
    uTime: 0,
    uEnergy: 0.15,
    uRes: [128, 128],
    uFg: hexToRgb01(themeColor("fg")),
    uBg: hexToRgb01(themeColor("bg")),
  });
  if (!fxShader) {
    fxCanvas = null; // no WebGL — silently skip the background layers
    return;
  }
  // flux flow-trace layer directly behind the field PNG (hidden unless the
  // FLUX layer is selected); electric layer at the very back. Both are
  // .sim-fx (z-index:-1), so DOM order decides: fx first, flux on top of it.
  fluxShaderCanvas = el("canvas", { class: "sim-fx", width: "128", height: "128", style: "display:none" });
  fluxShader = makeShaderCanvas(fluxShaderCanvas, FLUX_FRAG, {
    uTime: 0,
    uEnergy: 0.15,
    uRes: [128, 128],
    uFg: hexToRgb01(themeColor("fg")),
    uBg: hexToRgb01(themeColor("bg")),
  });
  fieldCell.prepend(fluxShaderCanvas);
  fieldCell.prepend(fxCanvas);
  fxTheme();
  window.addEventListener("themechange", fxTheme);
  if (!fxRaf) fxRaf = requestAnimationFrame(fxTick);
}

// derive the flow-direction texture from the cached site-energy frame's
// luminance gradient (runs only on a new energy frame — never per frame)
function updateFlowTexture() {
  const c = layerCache.energy;
  if (!fluxShader || !c || !c.has) return;
  const w = c.canvas.width;
  const h = c.canvas.height;
  const src = c.ctx.getImageData(0, 0, w, h).data;
  const N = 128;
  const step = w / N;
  const lum = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const k = (Math.floor(y * step) * w + Math.floor(x * step)) * 4;
      lum[y * N + x] = (src[k] * 77 + src[k + 1] * 150 + src[k + 2] * 29) / 65025;
    }
  }
  const gx = new Float32Array(N * N);
  const gy = new Float32Array(N * N);
  let maxg = 1e-9;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const xm = (x - 1 + N) % N;
      const xp = (x + 1) % N;
      const ym = (y - 1 + N) % N;
      const yp = (y + 1) % N;
      const ddx = (lum[y * N + xp] - lum[y * N + xm]) / 2;
      const ddy = (lum[yp * N + x] - lum[ym * N + x]) / 2;
      gx[y * N + x] = ddx;
      gy[y * N + x] = ddy;
      maxg = Math.max(maxg, Math.hypot(ddx, ddy));
    }
  }
  const data = new Uint8Array(N * N * 3);
  for (let i = 0; i < N * N; i++) {
    const mag = Math.hypot(gx[i], gy[i]);
    const ux = mag > 1e-12 ? gx[i] / mag : 0;
    const uy = mag > 1e-12 ? gy[i] / mag : 0;
    data[i * 3] = Math.round(255 * (0.5 + 0.5 * ux));       // dir.x → R
    data[i * 3 + 1] = Math.round(255 * (0.5 + 0.5 * uy));   // dir.y → G
    data[i * 3 + 2] = Math.round(255 * (mag / maxg));       // |velocity| → B
  }
  fluxShader.setTexture("uField", N, N, data);
}

function handleFrame(d) {
  if (d.type === "snapshot") {
    statusEl.textContent = `job ${d.status} — replaying ${d.history.length} frames`;
    for (const m of d.history) handleFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.matrix_png_b64) {
      drawPng(matrixCanvas, d.matrix_png_b64);
      cacheDraw("matrix", d.matrix_png_b64);
    }
    if (d.field_png_b64) cacheDraw("energy", d.field_png_b64);
    if (d.grad_png_b64) cacheDraw("grad", d.grad_png_b64);
    if (d.flux_png_b64) cacheDraw("flux", d.flux_png_b64);
    if (d.z_png_b64) cacheDraw("gerzon", d.z_png_b64);
    if (d.gerzon) {
      lastGerzon = d.gerzon;
      renderGerzon(lastGerzon);
    }
    if (d.flux_tiles) {
      lastFluxTiles = d.flux_tiles;
      renderFluxTiles(lastFluxTiles);
    }
    if (d.E !== undefined || d.T !== undefined) {
      // one sample across all waveform series; the chart keeps only the
      // keys present (E_exch/E_anis ride along whenever the frame has them)
      waveChart.push({
        E: d.E,
        E_dem: d.E_dem,
        E_exch: d.E_exch,
        E_anis: d.E_anis,
        E_goal: d.E_goal,
        E_tile: d.E_tile,
        E_z: d.E_z,
        T: d.T,
      });
    }
    if (typeof d.E === "number") {
      if (firstE === null) firstE = d.E;
      latestE = d.E;
    }
    if (d.step !== undefined) {
      const bits = [`step ${d.step}`];
      bits.push(`E ${Number(d.E).toPrecision(4)}`);
      bits.push(`best ${Number(d.best_E).toPrecision(4)}`);
      if (d.accepts !== undefined) bits.push(`accepts ${d.accepts}`);
      if (typeof d.goal_agree === "number")
        bits.push(`agree ${(d.goal_agree * 100).toFixed(1)}%`);
      if (d.elapsed_s !== undefined) bits.push(`${d.elapsed_s.toFixed(1)}s`);
      if (lastFluxTiles && lastFluxTiles.n_tiles != null) {
        bits.push(`tiles ${lastFluxTiles.n_tiles}×${lastFluxTiles.tile}`);
        if (lastFluxTiles.scales) {
          const extra = Object.entries(lastFluxTiles.scales)
            .filter(([s]) => s !== String(lastFluxTiles.tile))
            .map(([s, k]) => `${k}×${s}`)
            .slice(0, 3);
          if (extra.length) bits.push(extra.join(" "));
        }
        if (typeof lastFluxTiles.h8_agree === "number")
          bits.push(`H8 ${(lastFluxTiles.h8_agree * 100).toFixed(0)}%`);
      }
      if (typeof d.n_h2 === "number")
        bits.push(`H₂ ${d.n_h2}${typeof d.n_wall === "number" ? ` wall ${d.n_wall}` : ""}`);
      statusEl.textContent = bits.join(" · ");
    }
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
      msg(`annealed to a Hadamard matrix of order ${r.order}`, "ok");
      if (r.png_b64) {
        drawPng(matrixCanvas, r.png_b64);
        cacheDraw("matrix", r.png_b64);
      }
      const s = r.stats || {};
      const rows = [
        statRow("order", r.order),
        statRow("is_hadamard", s.is_hadamard, s.is_hadamard ? "good" : "bad"),
        statRow("max_off", s.max_off),
        statRow("f", s.f),
      ];
      if (s.det_log10 !== undefined && s.det_log10 !== null)
        rows.push(statRow("det_log10", s.det_log10.toFixed(2)));
      const ft = r.flux_tiles || lastFluxTiles;
      if (ft) renderFluxTiles(ft);
      if (ft && ft.n_tiles != null) {
        rows.push(statRow("flux tiles", `${ft.n_tiles} unique ${ft.tile}×${ft.tile} / ${ft.n_blocks} blocks`));
        if (typeof ft.h8_agree === "number")
          rows.push(statRow("H.8 tile agree", `${(ft.h8_agree * 100).toFixed(1)}%`,
            ft.kronecker_h8 ? "good" : ""));
      }
      const gz = r.gerzon || lastGerzon;
      if (gz) {
        lastGerzon = gz;
        renderGerzon(gz);
        const a = gz.aligned || {};
        if (a.n_cells != null)
          rows.push(statRow("gerzon H₂", `${a.n_h2}/${a.n_cells}`,
            a.n_h2 === a.n_cells ? "good" : ""));
        if (a.n_wall != null) rows.push(statRow("gerzon walls", a.n_wall));
        if (gz.E_z != null)
          rows.push(statRow("E_z", Number(gz.E_z).toPrecision(4),
            Number(gz.E_z) === 0 ? "good" : ""));
      }
      statsEl.replaceChildren(...rows);
      exportBtn.style.display = "";
      exportBtn.disabled = false;
    } else {
      msg(`no Hadamard found (best_E=${r.best_E ?? "?"})`, "error");
      statsEl.replaceChildren(statRow("ok", false, "bad"), statRow("best_E", r.best_E ?? "?"));
      if (lastFluxTiles) renderFluxTiles(lastFluxTiles);
      if (lastGerzon) renderGerzon(lastGerzon);
    }
  } catch (e) {
    msg(`result fetch failed: ${e.message}`, "error");
  }
}

function numVal(id, fallback) {
  const v = parseFloat(document.getElementById(id).value);
  return Number.isFinite(v) ? v : fallback;
}

async function doReadTiles() {
  const order = parseInt(document.getElementById("sim-order").value, 10);
  let start = document.getElementById("sim-start").value;
  if (start === "random") start = "sylvester";
  msg("reading flux tiles…");
  try {
    const d = await api(`/api/sim/flux-tiles?order=${order}&start=${start}`);
    lastFluxTiles = d.flux_tiles;
    renderFluxTiles(lastFluxTiles);
    if (d.flux_png_b64) {
      cacheDraw("flux", d.flux_png_b64);
      selectLayer("flux");
    }
    const t = d.flux_tiles || {};
    msg(
      t.n_tiles != null
        ? `flux tiles: ${t.n_tiles} unique ${t.tile}×${t.tile} · H8 ${((t.h8_agree || 0) * 100).toFixed(0)}%`
        : `flux tiles: n=${t.n} (no ${t.tile}-block tiling)`,
      "ok"
    );
  } catch (e) {
    msg(`flux tiles failed: ${e.message}`, "error");
  }
}

async function doReadGerzon() {
  const order = parseInt(document.getElementById("sim-order").value, 10);
  let start = document.getElementById("sim-start").value;
  if (start === "random") start = "sylvester";
  msg("reading Gerzon Z-wall…");
  try {
    const d = await api(`/api/sim/gerzon?order=${order}&start=${start}`);
    lastGerzon = d;
    renderGerzon(lastGerzon);
    if (d.z_png_b64) {
      cacheDraw("gerzon", d.z_png_b64);
      selectLayer("gerzon");
    }
    const a = d.aligned || {};
    msg(
      a.n_cells != null
        ? `Gerzon: H₂ ${a.n_h2}/${a.n_cells} · wall ${a.n_wall} · E_z ${Number(d.E_z).toPrecision(3)}`
        : "Gerzon: no cells",
      "ok"
    );
  } catch (e) {
    msg(`Gerzon read failed: ${e.message}`, "error");
  }
}

async function doStart() {
  const seedRaw = document.getElementById("sim-seed").value;
  const body = {
    order: parseInt(document.getElementById("sim-order").value, 10),
    T_start: numVal("sim-tstart", 10.0),
    T_end: numVal("sim-tend", 0.01),
    cooling: numVal("sim-cooling", 0.999),
    lam_ex: numVal("sim-lam-ex", 0),
    lam_ani: numVal("sim-lam-ani", 0),
    n_swap: parseInt(document.getElementById("sim-nswap").value, 10),
    budget_s: numVal("sim-budget", 30),
    field_every_steps: parseInt(document.getElementById("sim-field-every").value, 10),
    start: document.getElementById("sim-start").value,
    algorithm: document.getElementById("sim-algo").value,
  };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  const goalSel = document.getElementById("sim-goal-order").value;
  if (goalSel !== "") {
    const g = parseInt(goalSel, 10);
    if (g === body.order && body.start === "library") {
      msg("goal + library start at the same order is degenerate (the start already is the goal)", "error");
      return;
    }
    body.goal_order = g;
    body.lam_goal = numVal("sim-lam-goal", 0.5);
  }
  body.lam_tile = numVal("sim-lam-tile", 0);
  body.lam_z = numVal("sim-lam-z", 0);

  if (ws) ws.close();
  resetRun();
  msg("starting…");
  try {
    const { job_id } = await api("/api/sim/micromag", body);
    currentJob = job_id;
    cancelBtn.disabled = false;
    msg(`job ${job_id} running`, "ok");
    ws = connect(`/ws/job/${job_id}`, { message: handleFrame });
  } catch (e) {
    msg(`start failed: ${e.message}`, "error");
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
    lam_goal: parseFloat(document.getElementById("tune-lam-goal").value),
    lam_tile: parseFloat(document.getElementById("tune-lam-tile").value),
    lam_z: parseFloat(document.getElementById("tune-lam-z").value),
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

function labeledCanvas(label, size) {
  const canvas = el("canvas", { class: "sim-canvas", width: String(size), height: String(size) });
  return { canvas, node: el("div", { class: "sim-cell" }, el("div", { class: "sim-label" }, label), canvas) };
}

export function init(container) {
  const controls = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Micromagnetic annealing"),
    el("div", { class: "row" }, el("label", {}, "order"), el("input", { id: "sim-order", type: "number", value: "64", min: "4", step: "4" })),
    el(
      "div",
      { class: "row" },
      el("label", {}, "algorithm"),
      el("select", { id: "sim-algo" },
        ...SIM_ALGORITHMS.map((a) => el("option", { value: a }, a)))
    ),
    el(
      "div",
      { class: "row" },
      el("label", {}, "start"),
      el(
        "select",
        { id: "sim-start" },
        el("option", { value: "random" }, "random"),
        el("option", { value: "sylvester" }, "sylvester"),
        el("option", { value: "library" }, "library")
      )
    ),
    el("div", { class: "row" }, el("label", {}, "T start"), el("input", { id: "sim-tstart", type: "number", value: "10", step: "1" })),
    el("div", { class: "row" }, el("label", {}, "T end"), el("input", { id: "sim-tend", type: "number", value: "0.01", step: "0.01" })),
    el("div", { class: "row" }, el("label", {}, "cooling"), el("input", { id: "sim-cooling", type: "number", value: "0.999", step: "0.0001" })),
    el("div", { class: "row" }, el("label", {}, "lam_ex"), el("input", { id: "sim-lam-ex", type: "number", value: "0", step: "0.1" })),
    el("div", { class: "row" }, el("label", {}, "lam_ani"), el("input", { id: "sim-lam-ani", type: "number", value: "0", step: "0.1" })),
    el("div", { class: "row" }, el("label", {}, "n_swap"), el("input", { id: "sim-nswap", type: "number", value: "3", min: "1" })),
    el("div", { class: "row" }, el("label", {}, "budget s"), el("input", { id: "sim-budget", type: "number", value: "300", min: "1" })),
    el("div", { class: "row" }, el("label", {}, "field every"), el("input", { id: "sim-field-every", type: "number", value: "2500", min: "500", step: "500" })),
    el("div", { class: "row" }, el("label", {}, "seed"), el("input", { id: "sim-seed", type: "number", placeholder: "(random)" })),
    el(
      "div",
      { class: "row" },
      el("label", {}, "goal order"),
      // library orders from /api/orders; a goal ABOVE the start order
      // Kronecker-lifts a library start (H(n) ⊗ H(goal/n)) and anneals
      // at the goal's order
      el("select", { id: "sim-goal-order" }, el("option", { value: "" }, "none")),
      el("span", { class: "dim" }, "[EVOLVE TO LIBRARY GOAL]")
    ),
    el("div", { class: "row" }, el("label", {}, "lam_goal"), el("input", { id: "sim-lam-goal", type: "number", value: "0.5", step: "0.1", min: "0" })),
    el("div", { class: "row" }, el("label", {}, "lam_tile"), el("input", { id: "sim-lam-tile", type: "number", value: "0", step: "0.1", min: "0" })),
    el("div", { class: "row" }, el("label", {}, "lam_z"), el("input", { id: "sim-lam-z", type: "number", value: "0", step: "0.1", min: "0" })),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "sim-launch" }, "Start simulation"),
      el("button", { class: "btn", id: "sim-flux-read" }, "Read flux tiles"),
      el("button", { class: "btn", id: "sim-gerzon-read" }, "Read Gerzon"),
      (cancelBtn = el("button", { class: "btn", disabled: true }, "Cancel")),
      (exportBtn = el("button", { class: "btn", style: "display:none" }, "Export to library"))
    ),
    (msgEl = el("div", { class: "msg" }))
  );

  const tunePanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Live retune"),
    slider("tune-cooling", "cooling", "0.99", "0.99999", "0.00001", "0.999"),
    slider("tune-lam-ex", "lam_ex", "0", "1", "0.01", "0"),
    slider("tune-lam-ani", "lam_ani", "0", "1", "0.01", "0"),
    slider("tune-lam-goal", "lam_goal", "0", "5", "0.05", "0.5"),
    slider("tune-lam-tile", "lam_tile", "0", "5", "0.05", "0"),
    slider("tune-lam-z", "lam_z", "0", "5", "0.05", "0")
  );

  const statusPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Status"),
    (statusEl = el("div", { class: "status-line" }, "idle")),
    el("table", { class: "stats" }, (statsEl = el("tbody")))
  );

  // unified waveforms (Item 2): ONE strip chart for E / E_dem / E_exch /
  // E_anis / E_goal / E_tile / E_z / T with a multi-select bracket toggle
  // per series (default E + T on). E_EXCH/E_ANIS stay selectable even at
  // λ=0 — the series carries whatever the frames report; E_GOAL / E_Z
  // are flat 0 unless a library goal or Gerzon prior is active. Series
  // colors come from the active themeRamp (colors=null).
  const WAVE_SERIES = ["E", "E_dem", "E_exch", "E_anis", "E_goal", "E_tile", "E_z", "T"];
  const WAVE_DEFAULT_ON = new Set(["E", "T"]);
  const waveCanvas = el("canvas", { class: "chart", width: "520", height: "170" });
  waveChart = makeStripChart(waveCanvas, null);
  for (const n of WAVE_SERIES) waveChart.setVisible(n, WAVE_DEFAULT_ON.has(n));
  const waveBtns = {};
  const waveRow = el(
    "div",
    { class: "btn-row layer-select", id: "sim-wave-select" },
    ...WAVE_SERIES.map((name) => {
      const b = el("button", { class: "btn", "data-series": name }, `[${name.toUpperCase()}]`);
      b.addEventListener("click", () => {
        const on = !waveChart.isVisible(name);
        waveChart.setVisible(name, on);
        b.style.opacity = on ? "1" : "0.45";
      });
      b.style.opacity = WAVE_DEFAULT_ON.has(name) ? "1" : "0.45";
      waveBtns[name] = b;
      return b;
    })
  );

  const chartsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Waveforms"),
    waveRow,
    waveCanvas
  );

  // unified visualizers (Item 4): matrix canvas + ONE field canvas whose
  // layer is chosen by the [MATRIX][ENERGY][GRAD][FLUX][GERZON] selector;
  // each WS PNG is cached pristine per layer and composited on select
  for (const name of Object.keys(LAYER_LABELS)) {
    const c = el("canvas", { width: "256", height: "256" });
    layerCache[name] = { canvas: c, ctx: c.getContext("2d"), has: false };
  }
  layer = "energy";
  layerBtns = {};
  const layerRow = el(
    "div",
    { class: "btn-row layer-select", id: "sim-layer-select" },
    ...Object.entries(LAYER_LABELS).map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name }, `[${label}]`);
      b.addEventListener("click", () => selectLayer(name));
      layerBtns[name] = b;
      return b;
    })
  );

  const matrix = labeledCanvas("matrix (best)", 256);
  matrixCanvas = matrix.canvas;
  fieldCanvas = el("canvas", { class: "sim-canvas", width: "256", height: "256" });
  fieldCanvas.style.opacity = "0.88"; // let the electric layer bleed through
  const fieldCell = el(
    "div",
    { class: "sim-cell" },
    (fieldLabel = el("div", { class: "sim-label" }, LAYER_TITLES[layer])),
    fieldCanvas
  );

  const viewsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Live fields"),
    layerRow,
    el("div", { class: "panel-row" }, matrix.node, fieldCell),
    el("h2", {}, "Flux tiles (H.8 catalog)"),
    el("table", { class: "stats" }, (fluxInfoEl = el("tbody", {},
      el("tr", {}, el("td", { class: "dim" }, "idle — [READ FLUX TILES] or start a run"))))),
    el("h2", {}, "Gerzon AB (Z-wall)"),
    el("table", { class: "stats" }, (gerzonInfoEl = el("tbody", {},
      el("tr", {}, el("td", { class: "dim" }, "idle — [READ GERZON] or start a run")))))
  );
  initFx(fieldCell);
  for (const [k, b] of Object.entries(layerBtns)) b.style.opacity = k === layer ? "1" : "0.45";

  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, controls, tunePanel, statusPanel),
      el("div", {}, viewsPanel, chartsPanel)
    )
  );

  document.getElementById("sim-launch").addEventListener("click", doStart);
  document.getElementById("sim-flux-read").addEventListener("click", doReadTiles);
  document.getElementById("sim-gerzon-read").addEventListener("click", doReadGerzon);
  cancelBtn.addEventListener("click", doCancel);
  exportBtn.addEventListener("click", doExport);

  // fill the goal-order dropdown with the library orders
  fetch("/api/orders?max=4000")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || !d.known) return;
      const sel = document.getElementById("sim-goal-order");
      if (!sel) return;
      for (const n of d.known) sel.append(el("option", { value: String(n) }, String(n)));
    })
    .catch(() => {});
}

export function deactivate() {
  // stop the electric-field RAF loop and the job stream; the job itself
  // keeps running server-side
  if (fxRaf) cancelAnimationFrame(fxRaf);
  fxRaf = null;
  window.removeEventListener("themechange", fxTheme);
  if (ws) ws.close();
  ws = null;
}
