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
let surveyData = null; // last site survey result
let linkGroup = null; // TX/RX masts + LOS + Fresnel (survey 3-D)
let satTex = null;
let linkPanel, linkStatsEl, generateBtn;
let viewMode = "generate"; // "generate" | "survey"
let layersH2, threeTitle, octRowEl, viewH2;

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
  const w = 256;
  const h = 256;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5) * getSetting("renderScale"));
  renderer.setSize(w, h, false); // CSS owns display size via .sim-canvas
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  renderer.domElement.classList.add("sim-canvas");
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
  if (surveyData) {
    showTerLayer();
    if (linkGroup) {
      linkGroup.traverse((o) => {
        if (o.material && o.material.color && o.userData.theme) {
          o.material.color.set(themeColor(o.userData.theme));
        }
      });
    }
    renderThree();
  } else if (currentHeight) colorize(currentHeight);
  else renderThree();
}

function renderThree() {
  if (!renderer || !scene3d || !camera) return;
  // satellite texture is real colour — the CRT/DMG post pass would
  // quantize it into the theme ramp and look like a blank heightfield
  const skipPost = !!(surveyData && satTex);
  if (fxOn && pipeline && !skipPost) pipeline.render();
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

function disposeLink() {
  if (satTex) {
    satTex.dispose();
    satTex = null;
  }
  if (!linkGroup || !scene3d) return;
  scene3d.remove(linkGroup);
  linkGroup.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  linkGroup = null;
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
  const active = i !== null;
  const on = (kind) =>
    !active ? false : kind === "mute" ? muted.has(i) : kind === "solo" ? solo === i : false;
  for (const [kind, b] of Object.entries(octActionBtns)) {
    b.style.opacity = active ? (on(kind) ? "1" : "0.45") : "0.35";
  }
}

function octaveAction(kind) {
  if (kind === "full") {
    muted.clear();
    solo = null;
  } else {
    if (typeof terLayer !== "number") return;
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
      terLayer === "height" ? "heightfield" : terLayer === "profile" ? "path profile" : `octave ${Number(terLayer) + 1}`;
  }
  if (terLayer === "profile") {
    drawPathProfile(ctx);
    return;
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
  const b64 = octCache[Number(terLayer)];
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
  const btns = [mk("height", "HEIGHTFIELD")];
  if (surveyData) btns.push(mk("profile", "PROFILE"));
  for (let i = 0; i < nOct; i++) btns.push(mk(i, `OCT ${i + 1}`));
  layerSelEl.replaceChildren(...btns);
}

function drawPathProfile(ctx) {
  // Annotated TX→RX slice: ground (elev + 4/3-earth bulge), LOS chord,
  // 0.6·r₁ Fresnel corridor, obstruction fill, worst-point callout.
  // Colours stay on the theme tokens (no LUT) so the legend stays readable.
  if (!surveyData) return;
  const s = surveyData.survey;
  const W = layerCanvas.width, H = layerCanvas.height;
  const dist = s.dist_m || [];
  const n = dist.length;
  if (n < 2) return;
  const ground = (s.elev_m || []).map((e, i) => e + (s.bulge_m ? s.bulge_m[i] : 0));
  const los = s.los_line_m || ground;
  const r1 = s.fresnel_r1_m || ground.map(() => 0);
  const D = dist[n - 1] || 1;
  const txH = (s.tx && s.tx.h_m) || 0;
  const rxH = (s.rx && s.rx.h_m) || 0;
  const fg = themeColor("fg"), dim = themeColor("dim");
  const acc = themeColor("accent"), bg = themeColor("bg");
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    for (const v of [ground[i], los[i], los[i] + r1[i], los[i] - r1[i],
                    ground[0] + txH, ground[n - 1] + rxH]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!(hi > lo)) hi = lo + 10;
  const pad = 0.12 * (hi - lo);
  lo -= pad; hi += pad;
  const L = 44, R = 10, T = 36, B = 28; // plot margins
  const xOf = (d) => L + (d / D) * (W - L - R);
  const yOf = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.font = "10px monospace";
  ctx.fillStyle = dim;
  ctx.fillText("PATH PROFILE  ground · LOS · Fresnel 0.6 r₁", 6, 12);
  // legend chips
  const chips = [
    [dim, "GROUND"],
    [fg, "LOS"],
    [acc, "BLOCK"],
  ];
  let lx = 6;
  ctx.font = "9px monospace";
  for (const [col, lab] of chips) {
    ctx.fillStyle = col;
    ctx.fillRect(lx, 18, 8, 8);
    ctx.fillStyle = dim;
    ctx.fillText(lab, lx + 11, 26);
    lx += 11 + ctx.measureText(lab).width + 10;
  }
  // grid
  ctx.strokeStyle = dim;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  const nY = 4;
  for (let k = 0; k <= nY; k++) {
    const v = lo + (k / nY) * (hi - lo);
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = dim;
    ctx.fillText(`${v.toFixed(0)}`, 4, y + 3);
    ctx.globalAlpha = 0.35;
  }
  const kmStep = D >= 5000 ? 1000 : D >= 1500 ? 500 : 200;
  for (let d = 0; d <= D + 1; d += kmStep) {
    const x = xOf(Math.min(d, D));
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // obstruction fill (ground above LOS)
  ctx.fillStyle = acc;
  ctx.globalAlpha = 0.35;
  for (let i = 0; i < n - 1; i++) {
    if (ground[i] <= los[i] && ground[i + 1] <= los[i + 1]) continue;
    ctx.beginPath();
    ctx.moveTo(xOf(dist[i]), yOf(Math.max(ground[i], los[i])));
    ctx.lineTo(xOf(dist[i + 1]), yOf(Math.max(ground[i + 1], los[i + 1])));
    ctx.lineTo(xOf(dist[i + 1]), yOf(los[i + 1]));
    ctx.lineTo(xOf(dist[i]), yOf(los[i]));
    ctx.closePath();
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  // ground
  ctx.strokeStyle = dim;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(xOf(dist[0]), yOf(ground[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(dist[i]), yOf(ground[i]));
  ctx.stroke();
  // Fresnel 0.6 r1 corridor
  ctx.strokeStyle = dim;
  ctx.globalAlpha = 0.7;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(xOf(dist[0]), yOf(los[0] + 0.6 * r1[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(dist[i]), yOf(los[i] + 0.6 * r1[i]));
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(xOf(dist[i]), yOf(los[i] - 0.6 * r1[i]));
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  // LOS
  ctx.strokeStyle = fg;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(xOf(dist[0]), yOf(los[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(dist[i]), yOf(los[i]));
  ctx.stroke();
  // masts
  ctx.strokeStyle = acc;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xOf(dist[0]), yOf(ground[0]));
  ctx.lineTo(xOf(dist[0]), yOf(ground[0] + txH));
  ctx.moveTo(xOf(dist[n - 1]), yOf(ground[n - 1]));
  ctx.lineTo(xOf(dist[n - 1]), yOf(ground[n - 1] + rxH));
  ctx.stroke();
  ctx.fillStyle = acc;
  ctx.font = "9px monospace";
  ctx.fillText("TX", xOf(dist[0]) + 3, yOf(ground[0] + txH) - 3);
  ctx.fillText("RX", xOf(dist[n - 1]) - 16, yOf(ground[n - 1] + rxH) - 3);
  // worst-point callout
  const wd = s.worst_point_dist_m || D / 2;
  let wi = 0, best = Infinity;
  for (let i = 0; i < n; i++) {
    const err = Math.abs(dist[i] - wd);
    if (err < best) { best = err; wi = i; }
  }
  const clr = los[wi] - ground[wi];
  const frac = r1[wi] > 0 ? clr / r1[wi] : 0;
  ctx.strokeStyle = acc;
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(xOf(wd), T);
  ctx.lineTo(xOf(wd), H - B);
  ctx.stroke();
  ctx.setLineDash([]);
  const tag = clr >= 0
    ? `clear ${clr.toFixed(1)} m  (${frac.toFixed(2)} r₁)`
    : `BLOCKED  ${(-clr).toFixed(1)} m into LOS`;
  ctx.fillStyle = acc;
  ctx.font = "9px monospace";
  const tw = ctx.measureText(tag).width;
  const tx0 = Math.max(L, Math.min(W - R - tw, xOf(wd) - tw / 2));
  ctx.fillText(tag, tx0, T + 11);
  // slope of the ground over the whole hop
  const rise = ground[n - 1] - ground[0];
  const slopePct = (rise / D) * 100;
  ctx.fillStyle = dim;
  ctx.font = "9px monospace";
  ctx.fillText(`ASL ${lo.toFixed(0)}–${hi.toFixed(0)} m`, 4, H - 14);
  ctx.fillText(
    `${(D / 1000).toFixed(2)} km   slope ${rise >= 0 ? "+" : ""}${rise.toFixed(1)} m (${slopePct.toFixed(2)} %)`,
    4, H - 4
  );
  ctx.fillText(s.verdict || "", W - R - ctx.measureText(s.verdict || "").width, H - 4);
  if (terLayerLabel) {
    terLayerLabel.textContent =
      `${(D / 1000).toFixed(2)} km · ${s.verdict || "path"} · slope ${slopePct.toFixed(2)}%`;
  }
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

function rawNum(id) {
  const n = document.getElementById(id);
  if (!n) return null;
  const s = String(n.value).trim();
  if (s === "") return null;
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : null;
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
    disposeLink();
    surveyData = null;
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

// ---- survey ----------------------------------------------------------------

function statRow(k, v, cls = "") {
  return el("tr", {}, el("td", { class: "k" }, k), el("td", { class: `v ${cls}` }, String(v)));
}

function pathMApprox(tx, rx) {
  const latm = 111320;
  const lonm = 111320 * Math.cos((tx.lat * Math.PI) / 180);
  return Math.hypot((rx.lat - tx.lat) * latm, (rx.lon - tx.lon) * lonm);
}

function themedLine(points, themeName, dashed = false) {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: themeColor(themeName) });
  if (dashed) {
    mat.dashed = true;
  }
  const line = new THREE.Line(geo, mat);
  line.userData.theme = themeName;
  return line;
}

function lonLatToXZ(lon, lat, map) {
  // metric, unstretched: 1 world unit = (span_m / PLANE) metres
  const span = map.span_m || 2000;
  const latc = 0.5 * (map.lat_lo + map.lat_hi);
  const lonc = 0.5 * (map.lon_lo + map.lon_hi);
  const mPerLat = 111320;
  const mPerLon = 111320 * Math.max(0.2, Math.cos((latc * Math.PI) / 180));
  const x = ((lon - lonc) * mPerLon / span) * PLANE;
  const z = (-(lat - latc) * mPerLat / span) * PLANE; // north = −Z
  return { x, z, span };
}

function _bindSat(img) {
  if (satTex) satTex.dispose();
  satTex = new THREE.Texture(img);
  satTex.colorSpace = THREE.SRGBColorSpace;
  satTex.flipY = false; // row 0 = north = first PlaneGeometry row
  satTex.needsUpdate = true;
  if (terrainMesh) {
    terrainMesh.material.dispose();
    terrainMesh.material = new THREE.MeshBasicMaterial({ map: satTex });
  }
  if (wireMesh) wireMesh.visible = false;
  renderThree();
}

async function applySatTexture(map) {
  if (map.imagery_png_b64) {
    const img = new Image();
    img.onload = () => _bindSat(img);
    img.src = `data:image/png;base64,${map.imagery_png_b64}`;
    return;
  }
  // server may lack Pillow (Esri tiles are JPEG). Mosaic in the browser.
  const z = 16;
  const n = 1 << z;
  const toPix = (lat, lon) => {
    const px = (lon + 180) / 360 * n * 256;
    const phi = (lat * Math.PI) / 180;
    const py = (1 - Math.asinh(Math.tan(phi)) / Math.PI) / 2 * n * 256;
    return { px, py };
  };
  const nw = toPix(map.lat_hi, map.lon_lo);
  const se = toPix(map.lat_lo, map.lon_hi);
  const x0 = Math.floor(Math.min(nw.px, se.px) / 256);
  const x1 = Math.floor(Math.max(nw.px, se.px) / 256);
  const y0 = Math.floor(Math.min(nw.py, se.py) / 256);
  const y1 = Math.floor(Math.max(nw.py, se.py) / 256);
  if ((x1 - x0 + 1) * (y1 - y0 + 1) > 16) return;
  const load = (x, y) => new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
  });
  try {
    const cnv = el("canvas", { width: "256", height: "256" });
    const ctx = cnv.getContext("2d");
    const tiles = [];
    for (let ty = y0; ty <= y1; ty++) {
      for (let tx = x0; tx <= x1; tx++) tiles.push(load(tx, ty).then((im) => ({ im, tx, ty })));
    }
    const got = await Promise.all(tiles);
    const ox = x0 * 256, oy = y0 * 256;
    const W = (x1 - x0 + 1) * 256, H = (y1 - y0 + 1) * 256;
    const full = el("canvas", { width: String(W), height: String(H) });
    const fctx = full.getContext("2d");
    for (const t of got) fctx.drawImage(t.im, t.tx * 256 - ox, t.ty * 256 - oy);
    const sx = Math.min(nw.px, se.px) - ox;
    const sy = Math.min(nw.py, se.py) - oy;
    const sw = Math.abs(se.px - nw.px) || 1;
    const sh = Math.abs(se.py - nw.py) || 1;
    ctx.drawImage(full, sx, sy, sw, sh, 0, 0, 256, 256);
    _bindSat(cnv);
  } catch {
    /* keep vertex-colour fallback */
  }
}

function sampleHm(lon, lat, map) {
  // bilinear sample of the DEM so masts/LOS sit on the mesh, not inside it
  const hm = map.heightmap;
  if (!hm || !hm.length) return map.elev_lo_m || 0;
  const rows = hm.length, cols = hm[0].length;
  const du = map.lon_hi - map.lon_lo || 1;
  const dv = map.lat_hi - map.lat_lo || 1;
  const u = (lon - map.lon_lo) / du;
  const v = (map.lat_hi - lat) / dv; // row 0 = north
  const x = Math.max(0, Math.min(cols - 1, u * (cols - 1)));
  const y = Math.max(0, Math.min(rows - 1, v * (rows - 1)));
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(cols - 1, x0 + 1), y1 = Math.min(rows - 1, y0 + 1);
  const tx = x - x0, ty = y - y0;
  return (hm[y0][x0] * (1 - tx) * (1 - ty)
    + hm[y0][x1] * tx * (1 - ty)
    + hm[y1][x0] * (1 - tx) * ty
    + hm[y1][x1] * tx * ty);
}

function elevToY(e, ref, span) {
  // VE=1.5: a 20 m field ripple on a 1 km hop is 6 cm of the mesh.
  // The old min-max stretch made that same ripple fill 0.55 world units
  // (~hills).  Keep a little exaggeration so gentle relief is visible.
  const VE = 1.5;
  return ((e - ref) / span) * PLANE * VE;
}

function buildSurveyScene(s, map) {
  disposeTerrain();
  disposeLink();
  if (!map || !map.heightmap || !map.heightmap.length) return;
  const hm = map.heightmap; // metres, row 0 = north
  const rows = hm.length;
  const cols = hm[0].length;
  const span = map.span_m || 2000;
  const lo = map.elev_lo_m;
  const hi = map.elev_hi_m || lo + 1;
  const sx = Math.min(cols - 1, 128);
  const sy = Math.min(rows - 1, 128);
  const geo = new THREE.PlaneGeometry(PLANE, PLANE, sx, sy);
  const pos = geo.attributes.position;
  // same rotateX convention as buildTerrain: displace in +Z then flip to +Y
  for (let iy = 0; iy <= sy; iy++) {
    const r = Math.round((iy / sy) * (rows - 1));
    for (let ix = 0; ix <= sx; ix++) {
      const c = Math.round((ix / sx) * (cols - 1));
      pos.setZ(iy * (sx + 1) + ix, elevToY(hm[r][c], lo, span));
    }
  }
  geo.rotateX(-Math.PI / 2);
  const colors = new Float32Array(pos.count * 3);
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const colourSpan = Math.max(30, hi - lo);
  for (let iy = 0; iy <= sy; iy++) {
    const r = Math.round((iy / sy) * (rows - 1));
    for (let ix = 0; ix <= sx; ix++) {
      const c = Math.round((ix / sx) * (cols - 1));
      const t = Math.max(0, Math.min(1, (hm[r][c] - lo) / colourSpan));
      const col = themeRampSample(Math.pow(t, 0.7));
      const i = iy * (sx + 1) + ix;
      colors[i * 3] = col[0];
      colors[i * 3 + 1] = col[1];
      colors[i * 3 + 2] = col[2];
    }
  }
  terrainMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true }));
  scene3d.add(terrainMesh);
  // wireframe only as a fallback before the satellite lands (it hides
  // photographic texture and made the link look "inside" a cage)
  wireMesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(themeColor("dim")),
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    })
  );
  scene3d.add(wireMesh);
  currentHeight = null; // survey mesh is coloured in metres, not [0,1]
  applySatTexture(map);

  // ---- link overlay: masts, LOS, Fresnel rings, ground path --------------
  // Heights come off the DEM mesh (not the path array) so the ray starts
  // at the surface + mast, not buried in a sample mismatch.
  const EPS = 0.006; // lift lines a hair above the mesh
  linkGroup = new THREE.Group();
  const tx = s.tx, rx = s.rx;
  const pTx = lonLatToXZ(tx.lon, tx.lat, map);
  const pRx = lonLatToXZ(rx.lon, rx.lat, map);
  const gTx = sampleHm(tx.lon, tx.lat, map);
  const gRx = sampleHm(rx.lon, rx.lat, map);
  const yTx0 = elevToY(gTx, lo, span) + EPS;
  const yRx0 = elevToY(gRx, lo, span) + EPS;
  const yTx1 = elevToY(gTx + (tx.h_m || 0), lo, span) + EPS;
  const yRx1 = elevToY(gRx + (rx.h_m || 0), lo, span) + EPS;
  linkGroup.add(themedLine(
    [new THREE.Vector3(pTx.x, yTx0, pTx.z), new THREE.Vector3(pTx.x, yTx1, pTx.z)],
    "accent"
  ));
  linkGroup.add(themedLine(
    [new THREE.Vector3(pRx.x, yRx0, pRx.z), new THREE.Vector3(pRx.x, yRx1, pRx.z)],
    "accent"
  ));
  linkGroup.add(themedLine(
    [new THREE.Vector3(pTx.x, yTx1, pTx.z), new THREE.Vector3(pRx.x, yRx1, pRx.z)],
    "fg"
  ));
  // ground path (elev + bulge, mapped along the great-circle samples)
  if (s.dist_m && s.elev_m && s.dist_m.length > 1) {
    const D = s.dist_m[s.dist_m.length - 1] || 1;
    const groundPts = [];
    for (let i = 0; i < s.dist_m.length; i++) {
      const t = s.dist_m[i] / D;
      const lon = tx.lon + (rx.lon - tx.lon) * t;
      const lat = tx.lat + (rx.lat - tx.lat) * t;
      const p = lonLatToXZ(lon, lat, map);
      const g = sampleHm(lon, lat, map);
      groundPts.push(new THREE.Vector3(p.x, elevToY(g, lo, span) + EPS, p.z));
    }
    linkGroup.add(themedLine(groundPts, "dim"));
    // Fresnel rings at 25 / 50 / 75 % of the path, in the plane ⟂ path
    const dx = pRx.x - pTx.x, dz = pRx.z - pTx.z;
    const plen = Math.hypot(dx, dz) || 1;
    const ux = dx / plen, uz = dz / plen; // along-path
    // perpendicular in XZ: (-uz, ux)
    const px = -uz, pz = ux;
    const r1 = s.fresnel_r1_m || [];
    for (const frac of [0.25, 0.5, 0.75]) {
      const i = Math.round(frac * (r1.length - 1));
      if (!r1[i] || !s.los_line_m) continue;
      const t = frac;
      const cx = pTx.x + dx * t;
      const cz = pTx.z + dz * t;
      const cy = yTx1 + (yRx1 - yTx1) * t; // chord between mast tops, not buried in the DEM
      const rad = (r1[i] / span) * PLANE * 1.5; // same VE as height
      const ring = [];
      for (let k = 0; k <= 32; k++) {
        const a = (k / 32) * Math.PI * 2;
        // circle in the (perp-horizontal, vertical) plane
        ring.push(new THREE.Vector3(
          cx + Math.cos(a) * rad * px,
          cy + Math.sin(a) * rad,
          cz + Math.cos(a) * rad * pz
        ));
      }
      linkGroup.add(themedLine(ring, "dim"));
    }
  }
  // Coverage: a ground ring at the radio horizon ONLY if it fits in
  // this window.  A 15 m mast's horizon is ~16–32 km — drawing that as
  // a sphere on a 1 km patch swallowed the whole scene.  The Fresnel
  // rings along the hop ARE the transmission volume at this scale.
  const hr = s.radio_horizon_m || 0;
  const wr = hr > 0 ? (hr / span) * PLANE : 0;
  if (wr > 0.05 && wr < PLANE * 0.45) {
    const circ = [];
    for (let k = 0; k <= 64; k++) {
      const a = (k / 64) * Math.PI * 2;
      circ.push(new THREE.Vector3(pTx.x + wr * Math.cos(a), yTx0, pTx.z + wr * Math.sin(a)));
    }
    linkGroup.add(themedLine(circ, "accent"));
  }
  // optimal site: highest DEM cell (ridge / local high)
  const best = map.best_site;
  if (best && Number.isFinite(best.lat) && Number.isFinite(best.lon)) {
    const pb = lonLatToXZ(best.lon, best.lat, map);
    const yb = elevToY(best.elev_m, lo, span);
    const cg = new THREE.ConeGeometry(0.035, 0.09, 8);
    const cm = new THREE.MeshBasicMaterial({ color: themeColor("accent") });
    const cone = new THREE.Mesh(cg, cm);
    cone.position.set(pb.x, yb + 0.05, pb.z);
    cone.userData.theme = "accent";
    linkGroup.add(cone);
  }
  scene3d.add(linkGroup);
  if (camera) {
    camera.position.set(0.2, 0.7, 1.8);
    camera.lookAt(0, 0.05, 0);
    if (controls) { controls.target.set(0, 0.05, 0); controls.update(); }
  }
  renderThree();
}

async function doSurvey() {
  const txLat = rawNum("ter-s-txlat");
  const txLon = rawNum("ter-s-txlon");
  if (txLat == null || txLon == null) {
    msg("need TX latitude and longitude", "error");
    return;
  }
  const tx = { lat: txLat, lon: txLon, h_m: rawNum("ter-s-txh") ?? 15 };
  let rxLat = rawNum("ter-s-rxlat");
  let rxLon = rawNum("ter-s-rxlon");
  const rxH = rawNum("ter-s-rxh") ?? tx.h_m;
  let hopNote = "";
  // blank RX, same point, or leftover Swiss default vs a new TX:
  // treat as a 1 km east site hop so the path is never degenerate
  let rx;
  if (rxLat == null || rxLon == null) {
    const dlon = 1000 / (111320 * Math.max(0.2, Math.cos((tx.lat * Math.PI) / 180)));
    rx = { lat: tx.lat, lon: tx.lon + dlon, h_m: rxH };
    hopNote = "RX blank — 1 km east hop";
  } else {
    rx = { lat: rxLat, lon: rxLon, h_m: rxH };
    const hop = pathMApprox(tx, rx);
    if (hop < 50 || hop > 200000) {
      const dlon = 1000 / (111320 * Math.max(0.2, Math.cos((tx.lat * Math.PI) / 180)));
      rx = { lat: tx.lat, lon: tx.lon + dlon, h_m: rxH };
      hopNote = hop < 50 ? "TX/RX coincident — 1 km east hop" : "path > 200 km — local 1 km hop at TX";
    }
  }
  const body = {
    tx, rx,
    f_mhz: rawNum("ter-s-f") ?? 2450,
    p_tx_dbw: rawNum("ter-s-ptx") ?? 0,
    g_tx_dbi: rawNum("ter-s-gtx") ?? 2.15,
    g_rx_dbi: rawNum("ter-s-grx") ?? 2.15,
    medium: (document.getElementById("ter-s-medium") || {}).value || "air",
    n: 200, zoom: 12,
  };
  msg("surveying… (SRTM + imagery)");
  try {
    const s = await api("/api/antenna/survey", body);
    const m = await api("/api/antenna/survey/map", {
      tx: s.tx, rx: s.rx, zoom: 12, size: 64, heightmap: true, imagery: true,
    });
    surveyData = { survey: s, map: m };
    currentContribs = null;
    muted = new Set();
    solo = null;
    octCache = [];
    buildSurveyScene(s, m);
    terLayer = "profile";
    buildLayerButtons(0);
    selectTerLayer("profile");
    const relief = m.relief_m != null ? m.relief_m : (m.elev_hi_m - m.elev_lo_m);
    const hr = s.radio_horizon_m;
    const brg = s.bearing_deg;
    const siteBit = s.site_only
      ? `site · best hop ${brg != null ? brg.toFixed(0) + "°" : ""}`.trim()
      : hopNote;
    statusEl.textContent =
      `${(s.path_m / 1000).toFixed(2)} km · relief ${Number(relief).toFixed(1)} m` +
      (hr ? ` · horizon ${(hr / 1000).toFixed(1)} km` : "") +
      ` · f ${s.f_mhz} MHz` +
      (siteBit ? ` · ${siteBit}` : "");
    if (m.imagery_error && !m.imagery_png_b64) {
      msg(`survey: ${s.verdict}  (imagery: ${m.imagery_error})`,
          s.verdict === "obstructed" ? "error" : "ok");
    } else {
      msg(`survey: ${s.verdict}`, s.verdict === "obstructed" ? "error" : "ok");
    }
    const vcls = s.verdict === "LOS clear" ? "good" : s.verdict === "obstructed" ? "bad" : "";
    const rows = [
      statRow("verdict", s.verdict, vcls),
      statRow("path km", (s.path_m / 1000).toFixed(3)),
      statRow("relief m", Number(relief).toFixed(1)),
      statRow("clearance m", Number(s.clearance_m).toFixed(1)),
      statRow("Fresnel frac", Number(s.min_fresnel_clearance).toFixed(2) + " r₁"),
      statRow("radio horizon", hr ? `${(hr / 1000).toFixed(2)} km` : "—"),
      statRow("diffraction dB", Number(s.diffraction_loss_db).toFixed(1)),
      statRow("received", `${Number(s.received_dbw).toFixed(1)} dBW`),
    ];
    if (s.suggest_tx_h_m != null && s.suggest_tx_h_m > (s.tx.h_m || 0) + 0.5) {
      rows.push(statRow("raise TX to", `${Number(s.suggest_tx_h_m).toFixed(1)} m AGL`));
    }
    if (s.suggest_rx_h_m != null && s.suggest_rx_h_m > (s.rx.h_m || 0) + 0.5) {
      rows.push(statRow("raise RX to", `${Number(s.suggest_rx_h_m).toFixed(1)} m AGL`));
    }
    if (m.best_site) {
      const b = m.best_site;
      rows.push(statRow(
        "best site",
        `${b.lat.toFixed(5)}, ${b.lon.toFixed(5)}  (${b.elev_m.toFixed(0)} m` +
        (b.delta_m >= 0 ? ` +${b.delta_m.toFixed(0)}` : ` ${b.delta_m.toFixed(0)}`) + " m)"
      ));
    }
    linkStatsEl.replaceChildren(...rows);
    linkPanel.classList.remove("hidden");
  } catch (e) {
    surveyData = null;
    disposeTerrain();
    disposeLink();
    currentHeight = null;
    renderThree();
    msg(`survey failed: ${e.message}`, "error");
  }
}

// ---- tab lifecycle ----------------------------------------------------------

export function init(container) {
  layerSelEl = el("div", { class: "btn-row layer-select", id: "ter-layer-select" });
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

  // ---- [GENERATE] / [SURVEY] mode toggle ---------------------------------
  const genBody = el("div", { id: "ter-gen-body" },
    el("div", { class: "row" }, el("label", {}, "size"), el("input", { id: "ter-size", type: "number", value: "256", min: "8", max: "512", step: "8" })),
    el("div", { class: "row" }, el("label", {}, "order"), el("select", { id: "ter-order" }, ...ORDERS.map((o) => el("option", { value: String(o), ...(o === 64 ? { selected: "" } : {}) }, String(o))))),
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
    el("div", { class: "btn-row" }, (generateBtn = el("button", { class: "btn", id: "ter-generate" }, "Generate")))
  );

  const surveyBody = el("div", { id: "ter-survey-body", class: "hidden" },
    el("div", { class: "row" }, el("label", {}, "tx lat"), el("input", { id: "ter-s-txlat", type: "number", value: "46.6", step: "0.000001" })),
    el("div", { class: "row" }, el("label", {}, "tx lon"), el("input", { id: "ter-s-txlon", type: "number", value: "8.0", step: "0.000001" })),
    el("div", { class: "row" }, el("label", {}, "tx alt m"), el("input", { id: "ter-s-txh", type: "number", value: "15", min: "0", max: "500" })),
    el("div", { class: "row" }, el("label", {}, "rx lat"), el("input", { id: "ter-s-rxlat", type: "number", step: "0.000001", placeholder: "(same as TX)" })),
    el("div", { class: "row" }, el("label", {}, "rx lon"), el("input", { id: "ter-s-rxlon", type: "number", step: "0.000001", placeholder: "(same as TX)" })),
    el("div", { class: "row" }, el("label", {}, "rx alt m"), el("input", { id: "ter-s-rxh", type: "number", placeholder: "(= tx)", min: "0", max: "500" })),
    el("div", { class: "row" }, el("label", {}, "f MHz"), el("input", { id: "ter-s-f", type: "number", value: "2450", step: "1" })),
    el("div", { class: "row" }, el("label", {}, "tx dBW"), el("input", { id: "ter-s-ptx", type: "number", value: "0", step: "1" })),
    el("div", { class: "row" }, el("label", {}, "g tx dBi"), el("input", { id: "ter-s-gtx", type: "number", value: "2.15", step: "0.1" })),
    el("div", { class: "row" }, el("label", {}, "g rx dBi"), el("input", { id: "ter-s-grx", type: "number", value: "2.15", step: "0.1" })),
    el("div", { class: "row" }, el("label", {}, "medium"), el("select", { id: "ter-s-medium" }, el("option", { value: "air" }, "AIR"), el("option", { value: "water" }, "WATER"), el("option", { value: "water_sea" }, "WATER-SEA"))),
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "ter-s-run" }, "Run Survey"))
  );

  // same layer-select convention as Antenna / Micromag: [GENERATE][SURVEY]
  const modeBtns = {};
  const setMode = (name) => {
    const isGen = name === "generate";
    viewMode = name;
    for (const [k, b] of Object.entries(modeBtns)) {
      b.style.opacity = k === name ? "1" : "0.45";
    }
    genBody.classList.toggle("hidden", !isGen);
    surveyBody.classList.toggle("hidden", isGen);
    if (octRowEl) octRowEl.classList.toggle("hidden", !isGen);
    if (layerSelEl) layerSelEl.classList.toggle("hidden", !isGen);
    if (layersH2) layersH2.textContent = isGen ? "Layers" : "Path";
    if (viewH2) viewH2.textContent = isGen ? "Layer view" : "TX/RX elevation";
    if (threeTitle) threeTitle.textContent = isGen ? "Heightfield (drag to orbit)" : "Signal link (drag to orbit)";
    if (!isGen && surveyData) {
      terLayer = "profile";
      showTerLayer();
    }
  };
  const modeRow = el(
    "div",
    { class: "btn-row layer-select", id: "ter-mode-select" },
    ...[["generate", "GENERATE"], ["survey", "SURVEY"]].map(([name, label]) => {
      const b = el("button", { class: "btn", "data-layer": name, id: `ter-mode-${name}` }, `[${label}]`);
      b.addEventListener("click", () => setMode(name));
      modeBtns[name] = b;
      return b;
    })
  );
  setMode("generate");

  const controlsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Terrain"),
    modeRow,
    genBody,
    surveyBody,
    (msgEl = el("div", { class: "msg" })),
    (statusEl = el("div", { class: "status-line" }, "idle")),
    (layersH2 = el("h2", {}, "Layers")),
    layerSelEl,
    octRow
  );
  octRowEl = octRow;

  linkPanel = el(
    "div",
    { class: "panel hidden", id: "ter-link-panel" },
    el("h2", {}, "Link budget"),
    el("table", { class: "stats" }, (linkStatsEl = el("tbody")))
  );

  layerCanvas = el("canvas", { class: "sim-canvas", width: "384", height: "384" });
  const layersPanel = el(
    "div",
    { class: "panel" },
    (viewH2 = el("h2", {}, "Layer view")),
    el("div", { class: "sim-cell" }, (terLayerLabel = el("div", { class: "sim-label" }, "heightfield")), layerCanvas)
  );

  const threeWrap = el("div", { class: "panel three-wrap" }, (threeTitle = el("h2", {}, "Heightfield (drag to orbit)")));

  container.replaceChildren(
    el("div", { class: "lab" },
      el("div", {}, controlsPanel, linkPanel),
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

  document.getElementById("ter-s-run").addEventListener("click", doSurvey);
  generateBtn.addEventListener("click", () => {
    surveyData = null;
    disposeLink();
    linkPanel.classList.add("hidden");
    buildLayerButtons(0);
    doGenerate();
  });
  doGenerate();
}

export function deactivate() {
  window.removeEventListener("themechange", applyThreeTheme);
  if (controls) controls.dispose();
  disposeTerrain();
  disposeLink();
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
