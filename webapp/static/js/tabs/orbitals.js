// Orbitals — hydrogenic real-orbital |ψ|² sampler (Phase 5a).
// POST /api/gen/orbital → point cloud + weights. Unified viewport (Item 6b/
// Item 4): ONE main viewport (.orb-viewport: position:relative +
// overflow:hidden) with a [3D][XZ][BOTH] layer selector — 3D is a
// THREE.Points cloud (per-point color by weight on the active themeRamp,
// no lights; half-res post pipeline per theme), XZ is a CLIENT-SIDE density
// splat of the same point cloud (additive 1px hits into a 256² grid, log1p
// stretch, rendered via heatmap.js — always matches the displayed cloud;
// the server's proj_png_b64 is no longer requested), BOTH overlays the
// transparent-cleared cloud (renderer alpha:true, clear alpha 0, scene
// background dropped, post pipeline bypassed) on the dimmed splat.
// The QUANTUM_FRAG standing-wave layer (Item 2) is DRIVEN BY the cloud: a
// 64-bin radial |ψ|² profile (log-stretched) uploads to the shader's
// uDensity texture after each Simulate, modulating shell brightness and
// warping the interference phase (∝ density(r)); flat pattern fallback when
// no cloud is loaded. The shader canvas lives inside .orb-viewport
// (display:none) and its pixels feed the 3D scene as scene.background via
// CanvasTexture — clipped to the viewport, behind the cloud, post-processed
// with it (slow RAF ~12 fps).
// Axis convention (Item 6): three(x,y,z) = (−p.y, p.z, −p.x) — the HOA
// Studio mapping, physics z → +Y up. Side panel: quantum-number readout
// only.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { themeColor, currentTheme, themeRamp, getSetting, setSetting, themeRampSample } from "/js/theme.js";
import { makePostPipeline, makeShaderCanvas, hexToRgb01, QUANTUM_FRAG } from "/js/viz/shaders.js";
import { makeHeatmap } from "/js/viz/heatmap.js";

// cgb → "dmg": the 4-shade DMG-style grid fits the CGB LCD look (Feature 6)
const THEME_POST = { mono: "crt", green: "crt", amber: "crt", plasma: "crt", dmg: "dmg", cgb: "dmg", vga: "off" };
const L_LETTER = "spdfghi"; // l = 0..6

let msgEl, statusEl, readoutEl, projCanvas, projHeat, viewportEl, nSel, lSel, mSel;
let renderer, scene3d, camera, controls, pipeline;
let cloud = null; // THREE.Points
let currentWeights = null; // recolored on themechange
let fxOn = getSetting("fx"); // global FX default (Feature 8); last toggle wins
let qCanvas, qShader, qRaf = null, qLast = 0, qBgTex = null;
let orbLayer = "3d"; // active viewport layer: "3d" | "xz" | "both"
let orbLayerBtns = {};

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

// ---- cascading quantum-number selectors -----------------------------------

function rebuildL() {
  const n = parseInt(nSel.value, 10);
  const prev = parseInt(lSel.value || "0", 10);
  lSel.replaceChildren(
    ...Array.from({ length: n }, (_, l) =>
      el("option", { value: String(l), ...(l === Math.min(prev, n - 1) ? { selected: "" } : {}) }, `${l} (${L_LETTER[l]})`)
    )
  );
  rebuildM();
}

function rebuildM() {
  const l = parseInt(lSel.value, 10);
  const prev = parseInt(mSel.value || "0", 10);
  const keep = Math.abs(prev) <= l ? prev : 0;
  mSel.replaceChildren(
    ...Array.from({ length: 2 * l + 1 }, (_, i) => {
      const m = i - l;
      return el("option", { value: String(m), ...(m === keep ? { selected: "" } : {}) }, String(m));
    })
  );
  updateReadout();
}

function updateReadout() {
  if (!readoutEl || !nSel) return;
  const n = parseInt(nSel.value, 10);
  const l = parseInt(lSel.value, 10);
  const m = parseInt(mSel.value, 10);
  readoutEl.textContent = `n=${n} l=${l} m=${m} // ${n}${L_LETTER[l]}`;
}

// ---- three.js scene -----------------------------------------------------

function initThree(container) {
  const w = 560;
  const h = 560;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5) * getSetting("renderScale"));
  renderer.setSize(w, h);
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  container.appendChild(renderer.domElement);

  scene3d = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(0.6, 0.5, 2.4);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", renderThree);
  scene3d.add(new THREE.AxesHelper(1.3));

  pipeline = makePostPipeline(THREE, renderer, scene3d, camera, {
    mode: THEME_POST[currentTheme()] || "crt",
  });
  pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });

  window.addEventListener("themechange", applyThreeTheme);
  renderThree();
}

function applyThreeTheme() {
  if (!renderer) return;
  // clear is transparent whenever something behind the canvas must show
  // through: the quantum scene background (3d) or the DOM splat (both)
  const seeThrough = orbLayer === "both" || (orbLayer === "3d" && qBgTex);
  renderer.setClearColor(new THREE.Color(themeColor("bg")), seeThrough ? 0 : 1);
  if (pipeline) {
    pipeline.setMode(THEME_POST[currentTheme()] || "crt");
    pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: themeRamp() });
  }
  if (currentWeights) colorizeCloud();
  else renderThree();
}

function renderThree() {
  if (!renderer || !scene3d || !camera) return;
  // BOTH mode needs the transparent clear so the XZ underlay shows through;
  // the post pipeline composites onto an opaque background, so bypass it
  if (fxOn && pipeline && orbLayer !== "both") pipeline.render();
  else renderer.render(scene3d, camera);
}

function disposeCloud() {
  if (!cloud) return;
  scene3d.remove(cloud);
  cloud.geometry.dispose();
  cloud.material.dispose();
  cloud = null;
  currentWeights = null;
}

function buildCloud(points, weights, extent) {
  disposeCloud();
  const n = points.length;
  const pos = new Float32Array(n * 3);
  // HOA → three.js axis convention: three(x,y,z) = (−p.y, p.z, −p.x) —
  // the physics z-axis (m-quantization axis) ends up as +Y (screen up).
  for (let i = 0; i < n; i++) {
    pos[i * 3] = -points[i][1] / extent; // normalize to the unit-ish ball
    pos[i * 3 + 1] = points[i][2] / extent; // physics z → screen up
    pos[i * 3 + 2] = -points[i][0] / extent;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  cloud = new THREE.Points(
    geo,
    new THREE.PointsMaterial({ size: 0.018, vertexColors: true, sizeAttenuation: true }) // no lights
  );
  scene3d.add(cloud);
  currentWeights = weights;
  colorizeCloud();
}

function colorizeCloud() {
  // per-point color samples the active themeRamp by |ψ|² weight; re-runs on themechange
  const col = cloud.geometry.attributes.color;
  for (let i = 0; i < currentWeights.length; i++) {
    const t = Math.pow(currentWeights[i], 0.6); // gamma lift — tails stay visible
    const c = themeRampSample(t);
    col.setXYZ(i, c[0], c[1], c[2]);
  }
  col.needsUpdate = true;
  renderThree();
}

// ---- viewport layer select ([3D][XZ][BOTH]) --------------------------------

function selectOrbLayer(name) {
  orbLayer = name;
  for (const [k, b] of Object.entries(orbLayerBtns)) b.style.opacity = k === name ? "1" : "0.45";
  if (!renderer || !projCanvas) return;
  const dom = renderer.domElement;
  if (name === "3d") {
    dom.style.display = "";
    dom.style.position = "";
    dom.style.zIndex = "";
    if (scene3d && qBgTex) scene3d.background = qBgTex; // quantum flux behind the cloud
    renderer.setClearColor(new THREE.Color(themeColor("bg")), qBgTex ? 0 : 1);
    projCanvas.style.cssText = "display:none";
    renderThree(); // pipeline froze while hidden — repaint on return
  } else if (name === "xz") {
    dom.style.display = "none";
    projCanvas.style.cssText = ""; // in-flow, opaque
  } else {
    // both: dimmed splat underlay behind the transparent-cleared cloud (the
    // scene background is dropped so the DOM splat shows through)
    dom.style.display = "";
    dom.style.position = "relative";
    dom.style.zIndex = "1";
    if (scene3d) scene3d.background = null;
    renderer.setClearColor(new THREE.Color(themeColor("bg")), 0);
    projCanvas.style.cssText =
      "position:absolute;inset:0;margin:auto;width:min(100%,384px);aspect-ratio:1;opacity:0.55;z-index:0;";
    renderThree();
  }
}

// ---- quantum background layer (slow RAF ~12 fps) --------------------------
// The QUANTUM_FRAG canvas is a child of .orb-viewport (position:relative +
// overflow:hidden — it cannot bleed outside) but display:none: its pixels
// feed the three.js scene as `scene.background` via a CanvasTexture, so the
// standing-wave layer is clipped to the viewport, painted BEHIND the point
// cloud, and goes through the same post pipeline as the cloud (a DOM
// underlay would be hidden by the opaque post composite with FX on).
// After each Simulate the shader's uDensity texture receives the cloud's
// radial |ψ|² profile — the flux is DRIVEN BY the orbital (Item 2).

function qTick(t) {
  if (!qCanvas || !qShader) {
    qRaf = null;
    return;
  }
  if (t - qLast >= 83) {
    qLast = t;
    qShader.render(t / 1000);
    if (qBgTex) qBgTex.needsUpdate = true;
    renderThree(); // repaint at shader rate so the background animates
  }
  qRaf = requestAnimationFrame(qTick);
}

function qTheme() {
  if (!qShader) return;
  qShader.setUniform("uFg", hexToRgb01(themeColor("fg")));
  qShader.setUniform("uBg", hexToRgb01(themeColor("bg")));
}

function initQuantum(panel) {
  qCanvas = el("canvas", {
    class: "quantum-bg",
    width: "256",
    height: "256",
    style: "position:absolute;inset:0;width:100%;height:100%;display:none;z-index:-1;pointer-events:none;",
  });
  panel.prepend(qCanvas); // stays inside .orb-viewport (overflow:hidden)
  qShader = makeShaderCanvas(qCanvas, QUANTUM_FRAG, {
    uTime: 0,
    uEnergy: 0.3,
    uRes: [256, 256],
    uFg: hexToRgb01(themeColor("fg")),
    uBg: hexToRgb01(themeColor("bg")),
  });
  if (!qShader) {
    qCanvas.remove(); // no WebGL — skip the layer
    qCanvas = null;
    return;
  }
  // flat mid-density profile until a cloud is loaded (uDensityOn = 0 → the
  // shader shows the plain standing-wave pattern)
  qShader.setTexture("uDensity", 64, 1, new Uint8Array(64 * 3).fill(64));
  qShader.setUniform("uDensityOn", 0);
  qBgTex = new THREE.CanvasTexture(qCanvas);
  if (scene3d && orbLayer === "3d") scene3d.background = qBgTex;
  qTheme();
  window.addEventListener("themechange", qTheme);
  if (!qRaf) qRaf = requestAnimationFrame(qTick);
}

// ---- client-side XZ density splat (Item 4) --------------------------------
// Project the sampled points onto the physics XZ plane: additive 1px hits
// into a 256² grid (row = z, col = x, both over ±extent), log1p stretch for
// the peaked density. Rendered by heatmap.js through the active themeRamp —
// always matches the displayed point cloud (no server round-trip).

function splatXZ(points, extent, n = 256) {
  const g = Array.from({ length: n }, () => new Float64Array(n));
  const s = (n - 1) / (2 * extent);
  for (const p of points) {
    const ix = Math.round((p[0] + extent) * s);
    const iz = Math.round((p[2] + extent) * s);
    if (ix >= 0 && ix < n && iz >= 0 && iz < n) g[iz][ix] += 1;
  }
  for (const row of g) for (let i = 0; i < n; i++) row[i] = Math.log1p(row[i]);
  return g;
}

// 64-bin radial |ψ|² profile of the cloud (Item 2): histogram of r/extent,
// log1p-stretched, max-normalized, packed as a 64×1 RGB byte texture for
// QUANTUM_FRAG's uDensity — the shader warps and brightens its shells with
// it, so the flux backdrop is driven by the displayed orbital.
function radialProfile(points, extent, n = 64) {
  const bins = new Float64Array(n);
  for (const p of points) {
    const r = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / extent;
    if (r <= 1) bins[Math.min(n - 1, Math.floor(r * n))] += 1;
  }
  let max = 0;
  for (let i = 0; i < n; i++) {
    bins[i] = Math.log1p(bins[i]);
    if (bins[i] > max) max = bins[i];
  }
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const v = max > 0 ? Math.round((255 * bins[i]) / max) : 0;
    out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = v;
  }
  return out;
}

// ---- simulate -------------------------------------------------------------

async function doSimulate() {
  const seedRaw = document.getElementById("orb-seed").value;
  const body = {
    n: parseInt(nSel.value, 10),
    l: parseInt(lSel.value, 10),
    m: parseInt(mSel.value, 10),
    samples: Math.min(100000, parseInt(document.getElementById("orb-samples").value, 10) || 20000),
  };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  msg(`sampling |ψ|² for ${body.n}${L_LETTER[body.l]}…`);
  try {
    const d = await api("/api/gen/orbital", body);
    buildCloud(d.points, d.weights, d.extent);
    projHeat.render(splatXZ(d.points, d.extent));
    statusEl.textContent =
      `${d.points.length} pts · extent ±${d.extent.toFixed(1)} bohr · weights = |ψ|² (norm.)`;
    if (qShader) {
      const mean = d.weights.reduce((a, b) => a + b, 0) / d.weights.length;
      qShader.setUniform("uEnergy", 0.2 + 0.8 * mean);
      // couple the flux layer to the cloud: upload the radial density profile
      qShader.setTexture("uDensity", 64, 1, radialProfile(d.points, d.extent));
      qShader.setUniform("uDensityOn", 1);
    }
    updateReadout();
    msg("orbital sampled", "ok");
  } catch (e) {
    msg(`simulate failed: ${e.message}`, "error");
  }
}

// ---- tab lifecycle ----------------------------------------------------------

export function init(container) {
  nSel = el(
    "select",
    { id: "orb-n" },
    ...Array.from({ length: 7 }, (_, i) => el("option", { value: String(i + 1), ...(i + 1 === 2 ? { selected: "" } : {}) }, String(i + 1)))
  );
  lSel = el("select", { id: "orb-l" });
  mSel = el("select", { id: "orb-m" });
  nSel.addEventListener("change", () => {
    rebuildL();
    updateReadout();
  });
  lSel.addEventListener("change", rebuildM);
  mSel.addEventListener("change", updateReadout);

  const controlsPanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Hydrogenic orbital"),
    el("div", { class: "row" }, el("label", {}, "n"), nSel),
    el("div", { class: "row" }, el("label", {}, "l"), lSel),
    el("div", { class: "row" }, el("label", {}, "m"), mSel),
    el("div", { class: "row" }, el("label", {}, "samples"), el("input", { id: "orb-samples", type: "number", value: "20000", min: "100", max: "100000", step: "1000" })),
    el("div", { class: "row" }, el("label", {}, "seed"), el("input", { id: "orb-seed", type: "number", placeholder: "(random)" })),
    el("div", { class: "btn-row" }, el("button", { class: "btn", id: "orb-simulate" }, "Simulate")),
    (msgEl = el("div", { class: "msg" })),
    (statusEl = el("div", { class: "status-line" }, "idle"))
  );

  projCanvas = el("canvas", { class: "sim-canvas", width: "384", height: "384", style: "display:none" });
  projHeat = makeHeatmap(projCanvas); // client-side XZ splat renders here
  // sidebar: quantum-number readout only (the QUANTUM_FRAG layer moved into
  // the viewport — Item 4)
  const sidePanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Quantum numbers"),
    (readoutEl = el("div", { class: "status-line", style: "font-size:14px" }, "n=2 l=1 m=0 // 2p"))
  );

  // unified viewport: [3D][XZ][BOTH] selector swaps/overlays the three.js
  // canvas and the client-side XZ splat inside the same .orb-viewport area
  orbLayer = "3d";
  orbLayerBtns = {};
  const layerRow = el("div", { class: "btn-row layer-select", id: "orb-layer-select" });
  for (const [name, label] of [["3d", "3D"], ["xz", "XZ"], ["both", "BOTH"]]) {
    const b = el("button", { class: "btn", "data-layer": name, style: name === "3d" ? "" : "opacity:0.45" }, label);
    b.addEventListener("click", () => selectOrbLayer(name));
    orbLayerBtns[name] = b;
    layerRow.appendChild(b);
  }
  // overflow:hidden clips the quantum layer to the viewport (Item 2)
  viewportEl = el("div", { class: "orb-viewport", style: "position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;" });
  const threeWrap = el("div", { class: "panel three-wrap" }, layerRow, viewportEl);

  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, controlsPanel, sidePanel),
      el("div", {}, threeWrap)
    )
  );

  rebuildL(); // also rebuilds m + readout
  initThree(viewportEl);
  viewportEl.appendChild(projCanvas);
  initQuantum(viewportEl); // prepends the z-index:-1 shader layer

  const fxBox = el("input", { type: "checkbox", id: "orb-fx" });
  fxBox.checked = fxOn;
  fxBox.addEventListener("change", () => {
    fxOn = fxBox.checked;
    setSetting("fx", fxOn);
    renderThree();
  });
  threeWrap.appendChild(el("label", { class: "fx-toggle" }, fxBox, el("span", {}, "FX (post shader)")));

  document.getElementById("orb-simulate").addEventListener("click", doSimulate);
}

export function deactivate() {
  if (qRaf) cancelAnimationFrame(qRaf);
  qRaf = null;
  window.removeEventListener("themechange", applyThreeTheme);
  window.removeEventListener("themechange", qTheme);
  if (controls) controls.dispose();
  disposeCloud();
  if (scene3d) scene3d.background = null;
  if (qBgTex) qBgTex.dispose();
  if (pipeline) pipeline.dispose();
  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
  }
  renderer = scene3d = camera = controls = pipeline = qShader = qCanvas = qBgTex = null;
}
