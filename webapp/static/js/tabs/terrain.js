// Terrain — Hadamard-layered Perlin fBm generator (Phase 5a).
// POST /api/gen/terrain → heightmap JSON (≤128²) + per-octave layer PNGs +
// layers_f32 (signed amp-weighted octave contribs on the same grid).
// 3D view: displaced PlaneGeometry rotated into the three.js navigation
// convention (ground = XZ, up = +Y, heightmap row 0 at −Z front, columns
// along +X — Item 6), vertex colors by height on the active themeRamp,
// faint wireframe overlay, no lights; half-res post pipeline per theme
// (crt for phosphor themes, dmg/cgb for the LCDs, off for vga). The layer
// view sits LEFT of the 3D viewport in a .panel-row.ter-views (equal-width
// 1:1 panels, both canvases 384² with CSS-owned display size — Item 1); its
// [HEIGHTFIELD][OCT n] selector plus [MUTE]/[SOLO]/[FULL] live in the
// sidebar. Mute drops the selected octave's contribution, solo keeps only
// it, both recombine client-side from layers_f32 and renormalize to [0,1]
// (mesh updated in place via the position attribute, no realloc).
// Renders on demand (no RAF) — controls "change" events.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { retintCanvas, themeColor, currentTheme, themeRamp, getSetting, setSetting, themeRampSample } from "/js/theme.js";
import { makePostPipeline } from "/js/viz/shaders.js";

// cgb → "dmg": the 4-shade DMG-style grid fits the CGB LCD look (Feature 6)
const THEME_POST = { mono: "crt", green: "crt", amber: "crt", plasma: "crt", dmg: "dmg", cgb: "dmg", vga: "off" };
const ORDERS = [4, 8, 16, 32, 64, 128, 256]; // small constructible orders
const PLANE = 2.0; // world size of the displaced plane
const AMP = 0.55; // z-displacement amplitude for height ∈ [0, 1]

let msgEl, statusEl;
let renderer, scene3d, camera, controls, pipeline;
let terrainMesh = null, wireMesh = null;
let currentHeight = null; // last heightmap JSON — recolored on themechange
let fxOn = getSetting("fx"); // global FX default (Feature 8); last toggle wins

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
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
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

// ---- three.js scene -----------------------------------------------------

function initThree(container) {
  // 384² backing store (matches the layer canvas); updateStyle=false — CSS
  // (.ter-views canvas: width 100%, aspect-ratio 1) owns the display size so
  // both viewports stay exactly equal (Item 1)
  const w = 384;
  const h = 384;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5) * getSetting("renderScale"));
  renderer.setSize(w, h, false);
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  container.appendChild(renderer.domElement);

  scene3d = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(0.9, 0.85, 1.6);
  camera.lookAt(0, 0, 0);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", renderThree);
  scene3d.add(new THREE.AxesHelper(1.4));

  pipeline = makePostPipeline(THREE, renderer, scene3d, camera, {
    mode: THEME_POST[currentTheme()] || "crt",
  });
  pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });

  window.addEventListener("themechange", applyThreeTheme);
  renderThree();
}

function applyThreeTheme() {
  if (!renderer) return;
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  if (wireMesh) wireMesh.material.color.set(themeColor("dim"));
  if (pipeline) {
    pipeline.setMode(THEME_POST[currentTheme()] || "crt");
    pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });
  }
  if (currentHeight) colorize(currentHeight);
  else renderThree();
}

function renderThree() {
  if (!renderer || !scene3d || !camera) return;
  if (fxOn && pipeline) pipeline.render();
  else renderer.render(scene3d, camera);
}

function disposeTerrain() {
  for (const m of [terrainMesh, wireMesh]) {
    if (!m) continue;
    scene3d.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  terrainMesh = wireMesh = null;
}

function buildTerrain(hm) {
  disposeTerrain();
  const rows = hm.length;
  const cols = hm[0].length;
  // cap 128×128 segments (the server grid is already ≤128²)
  const sx = Math.min(cols - 1, 128);
  const sy = Math.min(rows - 1, 128);
  const geo = new THREE.PlaneGeometry(PLANE, PLANE, sx, sy);
  const pos = geo.attributes.position;
  // Axis convention (Item 6): three.js navigation — ground plane XZ, up +Y,
  // matching HOA Studio's three(x,y,z) = (−hoa.y, hoa.z, −hoa.x).
  // PlaneGeometry vertices are row-major, row 0 at local y = +PLANE/2; after
  // geo.rotateX(−π/2) that row lands at world z = −PLANE/2 (the −Z front
  // edge) and the height displacement becomes +Y. Mapping hm[iy] to vertex
  // row iy therefore puts heightmap row 0 at −Z (front) and columns
  // increasing along +X (right).
  for (let iy = 0; iy <= sy; iy++) {
    for (let ix = 0; ix <= sx; ix++) {
      pos.setZ(iy * (sx + 1) + ix, hm[iy][ix] * AMP);
    }
  }
  geo.rotateX(-Math.PI / 2);
  const colors = new Float32Array(pos.count * 3);
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  terrainMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true })); // no lights
  scene3d.add(terrainMesh);
  wireMesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(themeColor("dim")),
      wireframe: true,
      transparent: true,
      opacity: 0.3,
    })
  );
  scene3d.add(wireMesh);
  colorize(hm);
}

function colorize(hm) {
  // vertex colors sample the active themeRamp by height; re-runs on themechange
  currentHeight = hm;
  const geo = terrainMesh.geometry;
  const pos = geo.attributes.position;
  const col = geo.attributes.color;
  const sx = geo.parameters.widthSegments;
  for (let iy = 0; iy <= geo.parameters.heightSegments; iy++) {
    for (let ix = 0; ix <= sx; ix++) {
      const t = Math.pow(hm[iy][ix], 0.7); // gamma lift for dark lows; hm row 0 = −Z front
      const i = iy * (sx + 1) + ix;
      const c = themeRampSample(t);
      col.setXYZ(i, c[0], c[1], c[2]);
    }
  }
  col.needsUpdate = true;
  renderThree();
}

// ---- octave mute/solo recombination (Item 3) -------------------------------
// currentContribs holds the signed amp-weighted per-octave fields
// (layers_f32, same grid as the heightmap). mute = drop the octave from the
// Σ, solo = keep only it; the result is renormalized to [0, 1] and the mesh
// is rebuilt IN PLACE (position attribute Y = displacement, post-rotateX).

let currentContribs = null; // layers_f32 from the last generate
let muted = new Set(); // octave indices dropped from the sum
let solo = null; // octave index, or null

function recombHeight() {
  if (!currentContribs || !currentContribs.length) return null;
  const rows = currentContribs[0].length;
  const cols = currentContribs[0][0].length;
  const out = Array.from({ length: rows }, () => new Float64Array(cols));
  let any = false;
  currentContribs.forEach((layer, i) => {
    const inc = solo !== null ? i === solo : !muted.has(i);
    if (!inc) return;
    any = true;
    for (let r = 0; r < rows; r++) {
      const lr = layer[r];
      const or = out[r];
      for (let c = 0; c < cols; c++) or[c] += lr[c];
    }
  });
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of out) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const norm = any && hi > lo ? (v) => (v - lo) / (hi - lo) : () => 0;
  return out.map((row) => Array.from(row, norm));
}

function updateDisplacement(hm) {
  // in-place rebuild: after geo.rotateX(−π/2) the displacement lives in +Y
  if (!terrainMesh) return;
  const geo = terrainMesh.geometry;
  const pos = geo.attributes.position;
  const sx = geo.parameters.widthSegments;
  for (let iy = 0; iy <= geo.parameters.heightSegments; iy++) {
    for (let ix = 0; ix <= sx; ix++) {
      pos.setY(iy * (sx + 1) + ix, hm[iy][ix] * AMP);
    }
  }
  pos.needsUpdate = true;
}

function applyRecomb() {
  const hm = recombHeight();
  if (!hm) return;
  updateDisplacement(hm);
  colorize(hm); // also sets currentHeight → the heightfield layer follows
  showTerLayer();
}

let octActionBtns = {};

function syncOctActionBtns() {
  const i = typeof terLayer === "number" ? terLayer : null;
  const on = (kind) =>
    i === null ? false : kind === "mute" ? muted.has(i) : kind === "solo" ? solo === i : false;
  for (const [kind, b] of Object.entries(octActionBtns)) {
    b.style.opacity = on(kind) ? "1" : "0.45";
  }
}

function octaveAction(kind) {
  if (kind === "full") {
    muted.clear();
    solo = null;
  } else {
    if (typeof terLayer !== "number") return; // acts on the SELECTED octave
    const i = terLayer;
    if (kind === "mute") {
      solo = null;
      if (muted.has(i)) muted.delete(i);
      else muted.add(i);
    } else if (kind === "solo") {
      solo = solo === i ? null : i;
    }
  }
  syncOctActionBtns();
  applyRecomb();
}

// ---- layer view (Item 5) ---------------------------------------------------
// ONE 2D canvas + a [HEIGHTFIELD][OCT 1]…[OCT n] selector. HEIGHTFIELD is
// rendered from the heightmap JSON (grayscale luminance → theme LUT via
// retintCanvas, same pipeline as the server octave PNGs); the 3D mesh always
// shows the final heightfield regardless of the selected layer.

let layerCanvas, terLayerLabel, layerSelEl;
let terLayer = "height"; // "height" | octave index (number)
let octCache = []; // per-octave PNG b64 from the last generate
let terLayerBtns = {};

function showTerLayer() {
  if (!layerCanvas) return;
  const ctx = layerCanvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, layerCanvas.width, layerCanvas.height);
  if (terLayerLabel) {
    terLayerLabel.textContent =
      terLayer === "height" ? "heightfield" : `octave ${terLayer + 1}`;
  }
  if (terLayer === "height") {
    if (!currentHeight) return;
    const hm = currentHeight;
    const rows = hm.length;
    const cols = hm[0].length;
    const tmp = el("canvas", { width: String(cols), height: String(rows) });
    const tctx = tmp.getContext("2d");
    const img = tctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.max(0, Math.min(255, Math.round(hm[r][c] * 255)));
        const k = (r * cols + c) * 4;
        img.data[k] = v;
        img.data[k + 1] = v;
        img.data[k + 2] = v;
        img.data[k + 3] = 255;
      }
    }
    tctx.putImageData(img, 0, 0);
    ctx.drawImage(tmp, 0, 0, layerCanvas.width, layerCanvas.height);
    retintCanvas(layerCanvas);
    return;
  }
  const b64 = octCache[terLayer];
  if (!b64) return;
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, layerCanvas.width, layerCanvas.height);
    retintCanvas(layerCanvas);
  };
  img.src = `data:image/png;base64,${b64}`;
}

function selectTerLayer(name) {
  terLayer = name;
  for (const [k, b] of Object.entries(terLayerBtns)) {
    b.style.opacity = String(k) === String(terLayer) ? "1" : "0.45";
  }
  syncOctActionBtns();
  showTerLayer();
}

function buildLayerButtons(nOct) {
  terLayerBtns = {};
  const mk = (name, label) => {
    const b = el("button", { class: "btn", "data-layer": String(name) }, `[${label}]`);
    b.addEventListener("click", () => selectTerLayer(name));
    terLayerBtns[name] = b;
    return b;
  };
  layerSelEl.replaceChildren(
    mk("height", "HEIGHTFIELD"),
    ...Array.from({ length: nOct }, (_, i) => mk(i, `OCT ${i + 1}`))
  );
}

function showLayers(layers) {
  octCache = layers;
  buildLayerButtons(layers.length);
  selectTerLayer("height"); // default view: the final heightfield
}

// ---- generate -------------------------------------------------------------

function numVal(id, fallback) {
  const v = parseFloat(document.getElementById(id).value);
  return Number.isFinite(v) ? v : fallback;
}

async function doGenerate() {
  const seedRaw = document.getElementById("ter-seed").value;
  const body = {
    size: parseInt(document.getElementById("ter-size").value, 10),
    order: parseInt(document.getElementById("ter-order").value, 10),
    octaves: parseInt(document.getElementById("ter-octaves").value, 10),
    persistence: numVal("ter-persistence", 0.5),
    lacunarity: numVal("ter-lacunarity", 2.0),
    hadamard_mix: numVal("ter-mix", 0.5),
  };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  msg("generating terrain…");
  try {
    const d = await api("/api/gen/terrain", body);
    currentContribs = d.layers_f32 || null;
    muted = new Set(); // fresh terrain: no mute/solo state
    solo = null;
    buildTerrain(d.heightmap);
    showLayers(d.layers);
    syncOctActionBtns();
    const s = d.stats;
    statusEl.textContent =
      `${d.heightmap.length}×${d.heightmap[0].length} grid · min ${s.min.toFixed(3)} ` +
      `max ${s.max.toFixed(3)} mean ${s.mean.toFixed(3)} · mix ${body.hadamard_mix.toFixed(2)}`;
    msg("terrain generated", "ok");
  } catch (e) {
    msg(`generate failed: ${e.message}`, "error");
  }
}

// ---- tab lifecycle ----------------------------------------------------------

export function init(container) {
  // layer-select + octave action buttons live in the SIDEBAR (Item 1) — the
  // viewports hold only their canvases, so nothing drives size drift
  layerSelEl = el("div", { class: "btn-row layer-select", id: "ter-layer-select" });
  // [MUTE]/[SOLO] act on the SELECTED octave layer; [FULL] resets both
  octActionBtns = {};
  const octRow = el(
    "div",
    { class: "btn-row" },
    ...[["mute", "MUTE"], ["solo", "SOLO"], ["full", "FULL"]].map(([kind, label]) => {
      const b = el("button", { class: "btn", "data-oct": kind, style: "opacity:0.45" }, `[${label}]`);
      b.addEventListener("click", () => octaveAction(kind));
      octActionBtns[kind] = b;
      return b;
    })
  );

  const controlsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Hadamard-layered fBm"),
    el("div", { class: "row" }, el("label", {}, "size"), el("input", { id: "ter-size", type: "number", value: "256", min: "8", max: "512", step: "8" })),
    el(
      "div",
      { class: "row" },
      el("label", {}, "order"),
      el("select", { id: "ter-order" }, ...ORDERS.map((o) => el("option", { value: String(o), ...(o === 64 ? { selected: "" } : {}) }, String(o))))
    ),
    el("div", { class: "row" }, el("label", {}, "octaves"), el("input", { id: "ter-octaves", type: "number", value: "6", min: "1", max: "8" })),
    el("div", { class: "row" }, el("label", {}, "persistence"), el("input", { id: "ter-persistence", type: "number", value: "0.5", min: "0", max: "1", step: "0.05" })),
    el("div", { class: "row" }, el("label", {}, "lacunarity"), el("input", { id: "ter-lacunarity", type: "number", value: "2.0", min: "1", max: "4", step: "0.1" })),
    (() => {
      const val = el("span", { class: "slider-val", id: "ter-mix-val" }, "0.50");
      const input = el("input", { id: "ter-mix", type: "range", min: "0", max: "1", step: "0.01", value: "0.5" });
      input.addEventListener("input", () => (val.textContent = Number(input.value).toFixed(2)));
      return el("div", { class: "row slider-row" }, el("label", {}, "P↔H mix"), input, val);
    })(),
    el("div", { class: "row" }, el("label", {}, "seed"), el("input", { id: "ter-seed", type: "number", placeholder: "(random)" })),
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ter-generate" }, "Generate")),
    (msgEl = el("div", { class: "msg" })),
    (statusEl = el("div", { class: "status-line" }, "idle")),
    el("h2", {}, "Layers"),
    layerSelEl,
    octRow
  );

  // both viewports get the same 384² backing store; .ter-views CSS stretches
  // them to equal-width 1:1 panels (renderer uses setSize(…, false) so CSS
  // owns the display size — no inline-style size drift)
  layerCanvas = el("canvas", { class: "sim-canvas", width: "384", height: "384" });
  const layersPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Layer view"),
    el(
      "div",
      { class: "sim-cell" },
      (terLayerLabel = el("div", { class: "sim-label" }, "heightfield")),
      layerCanvas
    )
  );

  const threeWrap = el("div", { class: "panel three-wrap" }, el("h2", {}, "Heightfield (drag to orbit)"));

  // Item 1: Layer view LEFT (next to the sidebar), 3D heightfield RIGHT;
  // .ter-views makes both panels equal-width 1:1 viewports
  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, controlsPanel),
      el("div", {}, el("div", { class: "panel-row ter-views" }, layersPanel, threeWrap))
    )
  );

  initThree(threeWrap);

  const fxBox = el("input", { type: "checkbox", id: "ter-fx" });
  fxBox.checked = fxOn;
  fxBox.addEventListener("change", () => {
    fxOn = fxBox.checked;
    setSetting("fx", fxOn);
    renderThree();
  });
  threeWrap.appendChild(el("label", { class: "fx-toggle" }, fxBox, el("span", {}, "FX (post shader)")));

  document.getElementById("ter-generate").addEventListener("click", doGenerate);
  doGenerate(); // show something on first open
}

export function deactivate() {
  window.removeEventListener("themechange", applyThreeTheme);
  if (controls) controls.dispose();
  disposeTerrain();
  if (pipeline) pipeline.dispose();
  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
  }
  renderer = scene3d = camera = controls = pipeline = null;
  currentHeight = null;
  currentContribs = null;
  muted = new Set();
  solo = null;
}
