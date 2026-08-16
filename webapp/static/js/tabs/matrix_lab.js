// Matrix Lab — construct / load / verify / export Hadamard matrices.
// Renders server-side PNGs (png_b64) onto a pixelated canvas; stats come
// from hoa64.hadamard.check via the API (det shown as % of the Hadamard
// bound det_bound_log10(n)).
//
// The ℍ³ TRANSMUTE morphs the matrix viewport itself: the 2D canvas
// crossfades into a three.js overlay (same .canvas-wrap) whose points
// interpolate from their flat 2D grid positions to the ℍ³ targets over
// ~1 s. ROWS renders the row simplex as a Poincaré-ball point cloud with
// orthogonal-circle geodesics (unit wireframe sphere = ball boundary);
// LATTICE renders the entry grid as a hyperboloid wireframe. The Transmute
// button toggles back to 2D (reverse morph). No lights, post pipeline per
// theme.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { retintCanvas, fillPlusOne, themeColor, currentTheme, themeRamp, getSetting, setSetting } from "/js/theme.js";
import { makePostPipeline, hexToRgb01 } from "/js/viz/shaders.js";
import { connect } from "/js/ws.js";
import { CONSTRUCT_METHODS, SEARCH_ENGINES } from "/js/algorithms.js";

// cgb → "dmg": the 4-shade DMG-style grid fits the CGB LCD look (Feature 6)
const THEME_POST = { mono: "crt", green: "crt", amber: "crt", plasma: "crt", dmg: "dmg", cgb: "dmg", vga: "off" };

const METHODS = CONSTRUCT_METHODS;

let canvas, ctx, msgEl, statsEl, csvLink, currentOrder = null;
let searchWs = null, searchJob = null, searchStatusEl;

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

function statRow(k, v, cls = "") {
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, String(v)));
}

function showStats(d) {
  const s = d.stats || {};
  const rows = [];
  rows.push(statRow("ok", d.ok, d.ok ? "good" : "bad"));
  if (!d.ok) {
    statsEl.replaceChildren(...rows);
    return;
  }
  rows.push(statRow("order", d.order ?? s.n ?? "?"));
  rows.push(statRow("is_hadamard", s.is_hadamard, s.is_hadamard ? "good" : "bad"));
  rows.push(statRow("max_off", s.max_off));
  rows.push(statRow("f", s.f));
  const bound = s.det_bound ?? s.det_bound_log10;
  if (s.det_log10 !== undefined && s.det_log10 !== null && bound != null) {
    const pct = ((s.det_log10 / bound) * 100).toFixed(1);
    rows.push(
      statRow(
        "det_log10",
        `${Number(s.det_log10).toFixed(2)} / ${Number(bound).toFixed(2)} (${pct}% of bound)`
      )
    );
  } else if (bound != null) {
    rows.push(statRow("det_bound_log10", Number(bound).toFixed(2)));
  }
  rows.push(
    statRow("h2_all_balanced", s.h2_all_balanced, s.h2_all_balanced ? "good" : "bad")
  );
  if (d.h2) rows.push(statRow("h2 pairs", `${d.h2.balanced}/${d.h2.pairs} balanced`));
  if (d.modular)
    rows.push(statRow(`mod ${d.modular.mod}`, d.modular.ok, d.modular.ok ? "good" : "bad"));
  statsEl.replaceChildren(...rows);
}

function drawPng(b64) {
  const img = new Image();
  img.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    retintCanvas(canvas); // re-tint server PNG into the active theme
  };
  img.src = `data:image/png;base64,${b64}`;
}

function showResult(d) {
  resetTo2D(); // a new matrix invalidates the ℍ³ view — drop back to 2D
  showStats(d);
  if (d.ok && d.png_b64) drawPng(d.png_b64);
  if (d.ok && d.order) {
    currentOrder = d.order;
    csvLink.href = `/api/library/${d.order}/csv`;
    csvLink.classList.remove("hidden");
    csvLink.style.display = "";
  }
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

async function doConstruct() {
  const order = parseInt(document.getElementById("order").value, 10);
  const method = document.getElementById("method").value;
  const seedRaw = document.getElementById("seed").value;
  const body = { order, method };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  fillPlusOne(canvas); // all-+1 until the constructed matrix's PNG arrives
  msg("constructing…");
  try {
    const d = await api("/api/construct", body);
    showResult(d);
    msg(d.ok ? `constructed order ${d.order} via ${d.method}` : d.error, d.ok ? "ok" : "error");
  } catch (e) {
    msg(`construct failed: ${e.message}`, "error");
  }
}

function handleSearchFrame(d) {
  if (d.type === "snapshot") {
    for (const m of d.history || []) handleSearchFrame(m);
    return;
  }
  if (d.type === "progress") {
    if (d.matrix_png_b64) {
      const img = new Image();
      img.onload = () => {
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        retintCanvas(canvas);
      };
      img.src = `data:image/png;base64,${d.matrix_png_b64}`;
    }
    const bits = [];
    if (d.iter !== undefined) bits.push(`iter ${d.iter}`);
    if (d.step !== undefined) bits.push(`step ${d.step}`);
    const best = d.best_E ?? d.best_f;
    if (best !== undefined) bits.push(`best ${Number(best).toPrecision(4)}`);
    if (searchStatusEl) searchStatusEl.textContent = bits.join(" · ") || "running";
    return;
  }
  if (d.type === "end") {
    if (searchStatusEl) searchStatusEl.textContent = `job ${d.status}`;
    finishSearch();
  }
}

async function finishSearch() {
  if (!searchJob) return;
  try {
    const d = await api(`/api/search/${searchJob}`);
    const r = d.result || {};
    if (r.ok && r.png_b64) {
      showResult(r);
      currentOrder = r.order;
      msg(`found Hadamard order ${r.order} via ${r.engine}`, "ok");
    } else {
      msg(`search finished — no Hadamard (best ${r.best_f ?? r.best_E ?? "?"})`, "");
    }
  } catch (e) {
    msg(`search result failed: ${e.message}`, "error");
  }
}

async function doSearch() {
  const engine = document.getElementById("ml-engine").value;
  const order = parseInt(document.getElementById("order").value, 10);
  const budget_s = parseFloat(document.getElementById("ml-budget").value) || 30;
  if (searchWs) searchWs.close();
  fillPlusOne(canvas);
  msg("searching…");
  if (searchStatusEl) searchStatusEl.textContent = "connecting…";
  try {
    const { job_id } = await api("/api/search", { engine, order, budget_s, mode: "ils" });
    searchJob = job_id;
    msg(`job ${job_id} running`, "ok");
    searchWs = connect(`/ws/job/${job_id}`, { message: handleSearchFrame });
  } catch (e) {
    msg(`search failed: ${e.message}`, "error");
  }
}

async function doLoadLibrary() {
  const order = parseInt(document.getElementById("order").value, 10);
  fillPlusOne(canvas); // all-+1 until the library matrix's PNG arrives
  msg("loading…");
  try {
    const d = await api(`/api/library/${order}`);
    showResult(d);
    msg(`loaded order ${d.order} from library`, "ok");
  } catch (e) {
    msg(`library load failed: ${e.message}`, "error");
  }
}

function parseCsvMatrix(text) {
  const rows = text
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(",").map((x) => parseInt(x.trim(), 10)));
  if (rows.some((r) => r.length !== rows[0].length || r.some((v) => !Number.isFinite(v))))
    throw new Error("CSV must be a rectangular grid of integers");
  return rows;
}

async function doVerifyFile(file) {
  fillPlusOne(canvas); // all-+1 until the verification lands
  msg(`verifying ${file.name}…`);
  try {
    const matrix = parseCsvMatrix(await file.text());
    const d = await api("/api/verify", { matrix });
    showStats(d);
    msg(
      d.ok
        ? `verify ${d.order}×${d.order}: is_hadamard=${d.stats.is_hadamard}`
        : d.error,
      d.ok && d.stats.is_hadamard ? "ok" : "error"
    );
  } catch (e) {
    msg(`verify failed: ${e.message}`, "error");
  }
}

function doDownloadPng() {
  const a = el("a", { href: canvas.toDataURL("image/png"), download: `hadamard_${currentOrder ?? "matrix"}.png` });
  a.click();
}

// ---- ℍ³ transmute — in-place 2D→3D morph ---------------------------------
// ONE viewport: the 2D matrix canvas and the (lazy) ℍ³ three.js canvas are
// stacked inside .canvas-wrap. Transmute morphs the view in place (~1 s
// RAF): every point starts at its 2D grid position on a flat plane and
// interpolates to its Poincaré/hyperboloid target (position-attribute
// lerp — perf-first; geodesics + ball sphere fade in at the end). LATTICE
// is the natural case (flat n×n grid → warped hyperboloid); ROWS animates
// the row points from a √n grid layout to the simplex embedding. The
// Transmute button toggles back, reversing the morph to the 2D canvas.

let spRenderer, spScene, spCamera, spControls, spPipeline, spWrap;
let spObjects = []; // scene objects to dispose on rebuild
let spData = null; // last /api/viz/hadamard-space response (theme recolor)
let spFxOn = getSetting("fx"); // global FX default (Feature 8); last toggle wins
let spMode = "rows";
let spActive = false; // 3D view currently shown in the viewport
let spMorph = null; // running morph RAF handle
let spMorphAttr = null; // position attribute being morphed
let spTargetPos = null; // Float32Array ℍ³ target positions
let spGridPos = null; // Float32Array flat 2D grid positions
let spFadeIns = []; // materials fading in after the morph
let spMsgEl, spStatsEl, spToggleBtn;

function spMsg(text, kind = "") {
  spMsgEl.textContent = text;
  spMsgEl.className = `msg ${kind}`;
}

function syncTransmuteBtn() {
  if (spToggleBtn) {
    spToggleBtn.textContent = spActive ? "← back to 2D" : "Transmute → ℍ³";
  }
}

function gridStarts(count) {
  // flat √n×√n grid layout in the z=0 plane (for n² counts this is exactly
  // the matrix's own (row, col) grid), scaled to the unit-ball view
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    out[i * 3] = ((i % cols) / Math.max(1, cols - 1) - 0.5) * 1.7;
    out[i * 3 + 1] = (0.5 - Math.floor(i / cols) / Math.max(1, rows - 1)) * 1.7;
    out[i * 3 + 2] = 0;
  }
  return out;
}

function startMorph(attr, from, to, onDone) {
  if (spMorph) cancelAnimationFrame(spMorph);
  const dur = 1000; // ~1 s morph, smoothstep easing
  const t0 = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - t0) / dur);
    const e = t * t * (3 - 2 * t);
    const a = attr.array;
    for (let i = 0; i < a.length; i++) a[i] = from[i] + (to[i] - from[i]) * e;
    attr.needsUpdate = true;
    renderSpace();
    if (t < 1) {
      spMorph = requestAnimationFrame(step);
    } else {
      spMorph = null;
      if (onDone) onDone();
    }
  };
  spMorph = requestAnimationFrame(step);
}

function initSpaceThree() {
  if (spRenderer) return;
  const w = 512; // match the 2D matrix canvas — same viewport
  spRenderer = new THREE.WebGLRenderer({ antialias: true });
  spRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5) * getSetting("renderScale"));
  spRenderer.setSize(w, w);
  spRenderer.setClearColor(new THREE.Color(themeColor("bg")));
  spWrap.appendChild(spRenderer.domElement);

  spScene = new THREE.Scene();
  spCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  spCamera.position.set(0.35, 0.3, 2.2);
  spControls = new OrbitControls(spCamera, spRenderer.domElement);
  spControls.addEventListener("change", renderSpace);

  spPipeline = makePostPipeline(THREE, spRenderer, spScene, spCamera, {
    mode: THEME_POST[currentTheme()] || "crt",
  });
  spPipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });
  window.addEventListener("themechange", applySpaceTheme);
  renderSpace();
}

function renderSpace() {
  if (!spRenderer || !spScene || !spCamera) return;
  if (spFxOn && spPipeline) spPipeline.render();
  else spRenderer.render(spScene, spCamera);
}

function clearSpaceObjects() {
  for (const o of spObjects) {
    spScene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
  spObjects = [];
  spFadeIns = [];
}

function disposeSpace() {
  window.removeEventListener("themechange", applySpaceTheme);
  if (spMorph) {
    cancelAnimationFrame(spMorph);
    spMorph = null;
  }
  spMorphAttr = spTargetPos = spGridPos = null;
  spActive = false;
  clearSpaceObjects();
  if (spControls) spControls.dispose();
  if (spPipeline) spPipeline.dispose();
  if (spRenderer) {
    spRenderer.dispose();
    spRenderer.domElement.remove();
  }
  spRenderer = spScene = spCamera = spControls = spPipeline = null;
  spData = null;
}

function applySpaceTheme() {
  if (!spRenderer) return;
  spRenderer.setClearColor(new THREE.Color(themeColor("bg")));
  if (spPipeline) {
    spPipeline.setMode(THEME_POST[currentTheme()] || "crt");
    spPipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });
  }
  if (spData) showSpace(spData, false); // rebuild colors with the new theme
  else renderSpace();
}

function finishFadeIns() {
  for (const f of spFadeIns) f.mat.opacity = f.opacity;
  spFadeIns = [];
  renderSpace();
}

function showSpace(d, morph = false) {
  spData = d;
  clearSpaceObjects();
  const fg = new THREE.Color(themeColor("fg"));
  const accent = new THREE.Color(themeColor("accent"));
  const dim = new THREE.Color(themeColor("dim"));
  const faint = new THREE.Color(themeColor("faint"));
  const fade = (mat, opacity) => {
    // morph: supporting geometry (sphere/geodesics) fades in afterwards
    mat.opacity = morph ? 0 : opacity;
    if (morph) spFadeIns.push({ mat, opacity });
  };

  if (d.mode === "rows") {
    // ball boundary: faint unit wireframe sphere
    const sphMat = new THREE.LineBasicMaterial({ color: faint, transparent: true });
    fade(sphMat, 0.35);
    const sph = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.SphereGeometry(1, 24, 12)),
      sphMat
    );
    spScene.add(sph);
    spObjects.push(sph);

    // simplex points, sign-colored
    const n = d.points.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos.set(d.points[i], i * 3);
      const c = d.colors[i] ? fg : accent;
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const cloud = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ size: 0.035, vertexColors: true, sizeAttenuation: true })
    );
    spScene.add(cloud);
    spObjects.push(cloud);
    spMorphAttr = geo.attributes.position;
    spTargetPos = pos.slice();
    spGridPos = gridStarts(n);

    // geodesics: polylines expanded into one LineSegments draw call
    if (d.geodesics.length) {
      const segs = [];
      for (const g of d.geodesics) {
        for (let i = 0; i + 1 < g.length; i++) segs.push(...g[i], ...g[i + 1]);
      }
      const lgeo = new THREE.BufferGeometry();
      lgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs), 3));
      const lineMat = new THREE.LineBasicMaterial({ color: dim, transparent: true });
      fade(lineMat, 0.55);
      const lines = new THREE.LineSegments(lgeo, lineMat);
      spScene.add(lines);
      spObjects.push(lines);
    }
    const s = d.stats;
    const stretch = s.mean_euclidean_chord > 0 ? s.mean_hyperbolic_dist / s.mean_euclidean_chord : 0;
    spStatsEl.textContent =
      `κ ${s.kappa.toFixed(2)} · ${s.n_points} pts · ${s.n_geodesics} geodesics · ` +
      `mean d_ℍ ${s.mean_hyperbolic_dist.toFixed(3)} vs chord ${s.mean_euclidean_chord.toFixed(3)} ` +
      `(×${stretch.toFixed(2)} CAT(0) stretch)`;
  } else {
    // lattice: indexed grid mesh, wireframe, vertex colors by ±1 value
    const V = d.verts;
    const n = V.length;
    const pos = new Float32Array(n * n * 3);
    const col = new Float32Array(n * n * 3);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const k = i * n + j;
        pos.set(V[i][j], k * 3);
        const c = d.colors[i][j] ? fg : accent;
        col[k * 3] = c.r;
        col[k * 3 + 1] = c.g;
        col[k * 3 + 2] = c.b;
      }
    }
    const idx = [];
    for (let i = 0; i + 1 < n; i++) {
      for (let j = 0; j + 1 < n; j++) {
        const a = i * n + j;
        const b = (i + 1) * n + j;
        const c2 = i * n + j + 1;
        const e = (i + 1) * n + j + 1;
        idx.push(a, b, c2, b, e, c2);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true })
    );
    spScene.add(mesh);
    spObjects.push(mesh);
    spMorphAttr = geo.attributes.position;
    spTargetPos = pos.slice();
    spGridPos = gridStarts(n * n);
    const s = d.stats;
    spStatsEl.textContent = `κ ${s.kappa.toFixed(2)} · ${n}² hyperboloid grid · z-scale ${s.z_scale.toFixed(2)}`;
  }

  if (morph && spMorphAttr) {
    // start from the flat 2D grid, lerp to the ℍ³ targets, then fade extras in
    spMorphAttr.array.set(spGridPos);
    spMorphAttr.needsUpdate = true;
    startMorph(spMorphAttr, spGridPos, spTargetPos, finishFadeIns);
  } else {
    finishFadeIns();
  }
  renderSpace();
}

async function fetchSpace() {
  const order = currentOrder ?? parseInt(document.getElementById("order").value, 10);
  if (!Number.isFinite(order)) {
    spMsg("no order — construct or load a matrix first", "error");
    return null;
  }
  const body = {
    order,
    mode: spMode,
    kappa: parseFloat(document.getElementById("sp-kappa").value),
    geodesics: document.getElementById("sp-geos").checked,
    max_points: 256,
  };
  spMsg(`transmuting H(${order}) → ℍ³…`);
  try {
    const d = await api("/api/viz/hadamard-space", body);
    spMsg(`transmuted H(${d.order}) — ${d.mode}`, "ok");
    return d;
  } catch (e) {
    spMsg(`transmute failed: ${e.message}`, "error");
    return null;
  }
}

async function doTransmute() {
  // toggle: 2D → morph to ℍ³ in place; 3D → morph back to the 2D canvas
  if (spActive) {
    morphBackTo2D();
    return;
  }
  const d = await fetchSpace();
  if (!d) return;
  initSpaceThree();
  spWrap.style.display = "flex"; // overlay becomes the morph surface
  canvas.style.opacity = "0"; // 2D canvas crossfades out (CSS transition)
  showSpace(d, true);
  spActive = true;
  syncTransmuteBtn();
}

async function doRefetch() {
  // mode/κ/geodesics changed while the 3D view is up — rebuild in place
  const d = await fetchSpace();
  if (!d) return;
  showSpace(d, false);
}

function morphBackTo2D() {
  spActive = false;
  syncTransmuteBtn();
  canvas.style.opacity = "1"; // 2D canvas fades back in as the grid returns
  if (!spMorphAttr || !spTargetPos) {
    spWrap.style.display = "none";
    return;
  }
  startMorph(spMorphAttr, spTargetPos, spGridPos, () => {
    spWrap.style.display = "none";
  });
}

function resetTo2D() {
  // new matrix constructed/loaded — drop back to the plain 2D view
  if (spMorph) {
    cancelAnimationFrame(spMorph);
    spMorph = null;
  }
  spActive = false;
  syncTransmuteBtn();
  if (spWrap) spWrap.style.display = "none";
  if (canvas) canvas.style.opacity = "1";
}

async function doOpenOrder(order) {
  // cross-tab entry point (Library tab): library first, toolchain fallback
  document.getElementById("order").value = order;
  fillPlusOne(canvas); // all-+1 until the loaded matrix's PNG arrives
  msg("loading…");
  try {
    const d = await api(`/api/library/${order}`);
    showResult(d);
    msg(`loaded order ${d.order} from library`, "ok");
  } catch {
    try {
      const d = await api("/api/construct", { order, method: "auto" });
      showResult(d);
      msg(d.ok ? `constructed order ${d.order} via auto` : d.error, d.ok ? "ok" : "error");
    } catch (e) {
      msg(`open failed: ${e.message}`, "error");
    }
  }
}

export function init(container) {
  disposeSpace(); // tab re-activation: release any previous GL context
  spMode = "rows";
  const methodSel = el(
    "select",
    { id: "method" },
    ...METHODS.map((m) => el("option", { value: m }, m))
  );

  const constructPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Construct"),
    el("div", { class: "row" }, el("label", {}, "order"), el("input", { id: "order", type: "number", value: "64", min: "1" })),
    el("div", { class: "row" }, el("label", {}, "method"), methodSel),
    el("div", { class: "row" }, el("label", {}, "seed"), el("input", { id: "seed", type: "number", placeholder: "(gcp)" })),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "btn-construct" }, "Construct"),
      el("button", { class: "btn", id: "btn-library" }, "Load from library")
    ),
    (msgEl = el("div", { class: "msg" }))
  );

  const searchPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Search"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "algorithm"),
      el("select", { id: "ml-engine" },
        ...SEARCH_ENGINES.map((e) => el("option", { value: e }, e)))
    ),
    el("div", { class: "row" }, el("label", {}, "budget s"),
      el("input", { id: "ml-budget", type: "number", value: "30", min: "1" })),
    el("div", { class: "btn-row" },
      el("button", { class: "btn", id: "btn-search" }, "Search")),
    (searchStatusEl = el("div", { class: "status-line" }, "idle"))
  );

  const verifyPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Verify"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "csv file"),
      el("input", { id: "verify-file", type: "file", accept: ".csv,text/csv" })
    )
  );

  const exportPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Export"),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "btn-png" }, "Download PNG"),
      (csvLink = el("a", { class: "btn", href: "#", style: "display:none" }, "Download CSV"))
    )
  );

  const statsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Stats"),
    el("table", { class: "stats" }, (statsEl = el("tbody")))
  );

  canvas = el("canvas", { id: "matrix-canvas", width: "512", height: "512" });
  ctx = canvas.getContext("2d");

  // ---- ℍ³ transmute controls (single-viewport morph; the 3D canvas is an
  // overlay inside .canvas-wrap, created lazily on first Transmute) ----
  const rowsBtn = el("button", { class: "btn" }, "[ ROWS ]");
  const latBtn = el("button", { class: "btn" }, "[ LATTICE ]");
  const syncMode = () => {
    rowsBtn.style.opacity = spMode === "rows" ? "1" : "0.45";
    latBtn.style.opacity = spMode === "lattice" ? "1" : "0.45";
  };
  rowsBtn.addEventListener("click", () => {
    spMode = "rows";
    syncMode();
    if (spActive) doRefetch();
  });
  latBtn.addEventListener("click", () => {
    spMode = "lattice";
    syncMode();
    if (spActive) doRefetch();
  });
  syncMode();
  const modeRow = el("div", { class: "btn-row" }, rowsBtn, latBtn);
  const kappaVal = el("span", { class: "slider-val" }, "1.00");
  const kappaIn = el("input", { id: "sp-kappa", type: "range", min: "0.25", max: "4", step: "0.05", value: "1" });
  kappaIn.addEventListener("input", () => (kappaVal.textContent = Number(kappaIn.value).toFixed(2)));
  kappaIn.addEventListener("change", () => {
    if (spActive) doRefetch();
  });
  const geosBox = el("input", { type: "checkbox", id: "sp-geos", checked: "" });
  geosBox.addEventListener("change", () => {
    if (spActive) doRefetch();
  });
  const fxBox = el("input", { type: "checkbox" });
  fxBox.checked = spFxOn;
  fxBox.addEventListener("change", () => {
    spFxOn = fxBox.checked;
    setSetting("fx", spFxOn);
    renderSpace();
  });
  spWrap = el("div", { class: "three-wrap sp-overlay", style: "display:none" });
  const spacePanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "ℍ³ Transmute"),
    modeRow,
    el("div", { class: "row slider-row" }, el("label", {}, "κ curv"), kappaIn, kappaVal),
    el(
      "div",
      { class: "row" },
      el("label", { class: "fx-toggle" }, geosBox, el("span", {}, "geodesics")),
      el("label", { class: "fx-toggle" }, fxBox, el("span", {}, "FX (post shader)"))
    ),
    el(
      "div",
      { class: "btn-row" },
      (spToggleBtn = el("button", { class: "btn", id: "sp-transmute" }, "Transmute → ℍ³"))
    ),
    (spMsgEl = el("div", { class: "msg" })),
    (spStatsEl = el("div", { class: "status-line" }, "—"))
  );

  const lab = el(
    "div",
    { class: "lab" },
    el("div", {}, constructPanel, searchPanel, verifyPanel, exportPanel, statsPanel),
    el("div", {}, el("div", { class: "canvas-wrap" }, canvas, spWrap), spacePanel)
  );
  container.replaceChildren(lab);
  fillPlusOne(canvas); // tab setup: all-+1 field, not a blank viewport

  document.getElementById("btn-construct").addEventListener("click", doConstruct);
  document.getElementById("btn-library").addEventListener("click", doLoadLibrary);
  document.getElementById("btn-search").addEventListener("click", doSearch);
  document.getElementById("sp-transmute").addEventListener("click", doTransmute);
  document.getElementById("btn-png").addEventListener("click", doDownloadPng);
  document.getElementById("verify-file").addEventListener("change", (e) => {
    if (e.target.files[0]) doVerifyFile(e.target.files[0]);
  });
}

// cross-tab deep link from the Library tab (dispatched by main.js after
// this tab's init has rebuilt the DOM)
window.addEventListener("hoa64:payload", (e) => {
  const d = e.detail || {};
  if (d.tab !== "matrix_lab" || !msgEl || !Number.isFinite(d.order)) return;
  doOpenOrder(d.order);
});

export function deactivate() {
  // called by main.js on tab switch — release the ℍ³ view's GL context
  disposeSpace();
  if (searchWs) searchWs.close();
  searchWs = null;
}
