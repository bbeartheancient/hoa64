// HOA Studio — speaker-array designer + 3D sphere + scene encode/rotate/
// analyze with WAV export. Three.js is vendored (import map in index.html).
//
// Axis mapping (documented convention): HOA is right-handed with +X front,
// +Y left, +Z up (Ambix). Three.js is right-handed Y-up with the default
// camera looking down −Z. We map
//     three(x, y, z) = (−hoa.y, hoa.z, −hoa.x)
// a proper rotation (det +1): front → −z (into the screen), left → −x
// (screen left), up → +y. The inverse is hoa(x,y,z) = (−three.z, −three.x,
// three.y), used to convert sphere-vertex directions back to az/el for
// power-map sampling.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeHeatmap } from "/js/viz/heatmap.js";
import { themeColor, currentTheme, themeRamp, getSetting, setSetting } from "/js/theme.js";
import { makePostPipeline, hexToRgb01 } from "/js/viz/shaders.js";

const PRESETS = ["ring4", "ring8", "dome8", "icosa", "dodeca", "grid"];
const MAX_SOURCES = 4;

// theme → post-processing mode for the sphere view
// cgb → "dmg": the 4-shade DMG-style grid fits the CGB LCD look (Feature 6)
const THEME_POST = { mono: "crt", green: "crt", amber: "crt", plasma: "crt", dmg: "dmg", cgb: "dmg", vga: "off" };

let msgEl, statusEl, condEl, matrixHeat, positionsBody, speakers = [];
let doaEl, peakEl, energyEl, powerImg, wavLink;
let renderer, scene3d, camera, controls, sphereGroup, doaArrow, threeContainer;
let pipeline, fxOn = getSetting("fx"), baseMat, wireMat, markerMat, axesHelper; // fxOn: global FX default (Feature 8)
let currentPowerMap = null;

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

// ---- coordinate helpers -------------------------------------------------

function hoaUnit(azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  const ce = Math.cos(el);
  return [Math.cos(az) * ce, Math.sin(az) * ce, Math.sin(el)]; // hoa x,y,z
}

function hoaToThree(x, y, z) {
  return new THREE.Vector3(-y, z, -x);
}

function threeToAzEl(v) {
  // inverse of hoaToThree: hoa = (−v.z, −v.x, v.y)
  const hx = -v.z;
  const hy = -v.x;
  const hz = v.y;
  const az = (Math.atan2(hy, hx) * 180) / Math.PI;
  const el = (Math.asin(Math.max(-1, Math.min(1, hz))) * 180) / Math.PI;
  return [az, el];
}

// ---- three.js scene -----------------------------------------------------

function initThree(container) {
  const w = 560;
  const h = 560;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5) * getSetting("renderScale")); // perf: cap DPR
  renderer.setSize(w, h);
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  container.appendChild(renderer.domElement);

  scene3d = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(0.4, 0.35, 2.6);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", renderThree);

  sphereGroup = new THREE.Group();
  scene3d.add(sphereGroup);

  // power-map carrier: vertex-colored unit sphere (starts at theme dim)
  const geo = new THREE.SphereGeometry(1, 64, 32);
  const nVerts = geo.attributes.position.count;
  const colors = new Float32Array(nVerts * 3);
  colors.fill(0.04);
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  baseMat = new THREE.MeshBasicMaterial({ vertexColors: true }); // no lights
  sphereGroup.add(new THREE.Mesh(geo, baseMat));
  sphereGroup.userData.geo = geo;

  // faint wireframe + axes for orientation
  wireMat = new THREE.LineBasicMaterial({
    color: new THREE.Color(themeColor("dim")),
    transparent: true,
    opacity: 0.35,
  });
  sphereGroup.add(
    new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(1.002, 24, 12)), wireMat)
  );
  axesHelper = new THREE.AxesHelper(1.35);
  sphereGroup.add(axesHelper);

  // shared speaker-marker material (reused by showSpeakers)
  markerMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(themeColor("fg")) });

  // DOA arrow (hidden until a scene render)
  doaArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 0),
    1.35,
    new THREE.Color(themeColor("accent")),
    0.12,
    0.06
  );
  doaArrow.visible = false;
  scene3d.add(doaArrow);

  // half-res post pipeline (crt/dmg/off per theme)
  pipeline = makePostPipeline(THREE, renderer, scene3d, camera, {
    mode: THEME_POST[currentTheme()] || "crt",
  });
  pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: THEMES_ramp() });

  window.addEventListener("themechange", applyThreeTheme);
  renderThree();
}

function THEMES_ramp() {
  // active visualizer ramp (cgb palette packs + vga subthemes included) for
  // the post pass palette LUT
  return themeRamp();
}

function applyThreeTheme() {
  if (!renderer) return;
  renderer.setClearColor(new THREE.Color(themeColor("bg")));
  wireMat.color.set(themeColor("dim"));
  markerMat.color.set(themeColor("fg"));
  doaArrow.setColor(new THREE.Color(themeColor("accent")));
  if (pipeline) {
    pipeline.setMode(THEME_POST[currentTheme()] || "crt");
    pipeline.setTheme({ fg: themeColor("fg"), bg: themeColor("bg"), ramp: THEMES_ramp() });
  }
  if (currentPowerMap) showPowerMap(currentPowerMap);
  else renderThree();
}

function renderThree() {
  if (!renderer || !scene3d || !camera) return;
  if (fxOn && pipeline) pipeline.render();
  else renderer.render(scene3d, camera);
}

function clearMarkers() {
  if (!sphereGroup) return;
  const dead = sphereGroup.children.filter((c) => c.userData.marker);
  for (const m of dead) {
    sphereGroup.remove(m);
    m.geometry.dispose();
  }
}

function showSpeakers(positions) {
  clearMarkers();
  const geo = new THREE.SphereGeometry(0.028, 12, 8);
  for (const p of positions) {
    const [x, y, z] = hoaUnit(p.az, p.el);
    const m = new THREE.Mesh(geo, markerMat);
    m.position.copy(hoaToThree(x, y, z));
    m.userData.marker = true;
    sphereGroup.add(m);
  }
  renderThree();
}

function showDoa(az, elv) {
  const [x, y, z] = hoaUnit(az, elv);
  const dir = hoaToThree(x, y, z).normalize();
  doaArrow.setDirection(dir);
  doaArrow.visible = true;
  renderThree();
}

function showPowerMap(pm) {
  // sample the (n_el × n_azi) power map at every sphere vertex; vertex
  // colors lerp theme bg→fg so the sphere re-tints on themechange
  currentPowerMap = pm;
  const geo = sphereGroup.userData.geo;
  const pos = geo.attributes.position;
  const col = geo.attributes.color;
  const rows = pm.power.length; // n_el
  const cols = pm.power[0].length; // n_azi
  let max = 0;
  for (const row of pm.power) for (const v of row) max = Math.max(max, v);
  const inv = max > 0 ? 1 / max : 0;
  const bg = hexToRgb01(themeColor("bg"));
  const fg = hexToRgb01(themeColor("fg"));
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    const [az, elv] = threeToAzEl(v);
    let fi = ((elv + 90) / 180) * (rows - 1);
    let fj = (((az + 180) % 360) / 360) * cols;
    fi = Math.max(0, Math.min(rows - 1, Math.round(fi)));
    fj = ((Math.floor(fj) % cols) + cols) % cols;
    const t = Math.pow(pm.power[fi][fj] * inv, 0.5); // gamma-ish lift
    col.setXYZ(
      i,
      bg[0] + (fg[0] - bg[0]) * t,
      bg[1] + (fg[1] - bg[1]) * t,
      bg[2] + (fg[2] - bg[2]) * t
    );
  }
  col.needsUpdate = true;
  renderThree();
}

// ---- speaker designer ---------------------------------------------------

function renderPositionsTable() {
  positionsBody.replaceChildren(
    ...speakers.map((p, i) => {
      const az = el("input", { type: "number", value: String(Math.round(p.az * 10) / 10), step: "1" });
      const ev = el("input", { type: "number", value: String(Math.round(p.el * 10) / 10), step: "1" });
      az.addEventListener("change", () => (p.az = parseFloat(az.value) || 0));
      ev.addEventListener("change", () => (p.el = parseFloat(ev.value) || 0));
      const rm = el("button", { class: "btn btn-xs" }, "×");
      rm.addEventListener("click", () => {
        speakers.splice(i, 1);
        renderPositionsTable();
      });
      return el("tr", {}, el("td", {}, az), el("td", {}, ev), el("td", {}, rm));
    })
  );
}

async function doSpeakers() {
  const order = parseInt(document.getElementById("hoa-order").value, 10);
  msg("computing decode matrix…");
  try {
    const d = await api("/api/hoa/speakers", { positions: speakers, order });
    condEl.textContent = `cond(Y) = ${d.cond.toExponential(2)} · ${d.positions.length} speakers × ${d.n_channels} channels`;
    matrixHeat.render(d.decode_matrix);
    showSpeakers(d.positions);
    msg("decode matrix computed", "ok");
  } catch (e) {
    msg(`speakers failed: ${e.message}`, "error");
  }
}

async function doPreset() {
  const preset = document.getElementById("hoa-preset").value;
  const order = parseInt(document.getElementById("hoa-order").value, 10);
  msg(`loading preset ${preset}…`);
  try {
    const d = await api("/api/hoa/speakers", { preset, order });
    speakers = d.positions;
    renderPositionsTable();
    condEl.textContent = `cond(Y) = ${d.cond.toExponential(2)} · ${d.positions.length} speakers × ${d.n_channels} channels`;
    matrixHeat.render(d.decode_matrix);
    showSpeakers(d.positions);
    msg(`preset ${preset} loaded`, "ok");
  } catch (e) {
    msg(`preset failed: ${e.message}`, "error");
  }
}

// ---- scene --------------------------------------------------------------

const srcRows = [];

function sourceRow(i) {
  const kind = el(
    "select",
    { id: `src-kind-${i}` },
    el("option", { value: "tone" }, "tone"),
    el("option", { value: "noise" }, "noise")
  );
  const row = el(
    "div",
    { class: "row src-row" },
    el("label", {}, `src ${i + 1}`),
    el("input", { id: `src-az-${i}`, type: "number", placeholder: "az", value: i === 0 ? "30" : "" }),
    el("input", { id: `src-el-${i}`, type: "number", placeholder: "el", value: i === 0 ? "10" : "" }),
    el("input", { id: `src-freq-${i}`, type: "number", placeholder: "Hz", value: i === 0 ? "440" : "" }),
    el("input", { id: `src-gain-${i}`, type: "number", placeholder: "gain", value: i === 0 ? "1" : "", step: "0.1" }),
    kind
  );
  srcRows.push(row);
  return row;
}

function collectSources() {
  const out = [];
  for (let i = 0; i < MAX_SOURCES; i++) {
    const azRaw = document.getElementById(`src-az-${i}`).value;
    if (azRaw === "") continue;
    out.push({
      az: parseFloat(azRaw) || 0,
      el: parseFloat(document.getElementById(`src-el-${i}`).value) || 0,
      freq: parseFloat(document.getElementById(`src-freq-${i}`).value) || 440,
      gain: parseFloat(document.getElementById(`src-gain-${i}`).value) || 1,
      kind: document.getElementById(`src-kind-${i}`).value,
    });
  }
  return out;
}

function rotSlider(id, label) {
  const val = el("span", { class: "slider-val", id: `${id}-val` }, "0");
  const input = el("input", { id, type: "range", min: "-180", max: "180", step: "1", value: "0" });
  input.addEventListener("input", () => (val.textContent = input.value));
  input.addEventListener("change", () => doScene()); // re-render on release
  return el("div", { class: "row slider-row" }, el("label", {}, label), input, val);
}

async function doScene() {
  const sources = collectSources();
  if (!sources.length) {
    msg("no sources — fill at least one az", "error");
    return;
  }
  const body = {
    sources,
    rotate: {
      yaw: parseFloat(document.getElementById("rot-yaw").value),
      pitch: parseFloat(document.getElementById("rot-pitch").value),
      roll: parseFloat(document.getElementById("rot-roll").value),
    },
    order: parseInt(document.getElementById("hoa-order").value, 10),
    duration: 0.25,
    wav: true,
  };
  msg("rendering scene…");
  try {
    const d = await api("/api/hoa/scene", body);
    doaEl.textContent = `doa az ${d.doa.az.toFixed(1)}° el ${d.doa.el.toFixed(1)}°`;
    peakEl.textContent = `peak az ${d.peak.az.toFixed(1)}° el ${d.peak.el.toFixed(1)}° (${d.peak.val.toExponential(2)})`;
    energyEl.textContent = `energy ${d.energy.toExponential(3)}`;
    powerImg.src = `data:image/png;base64,${d.power_png_b64}`;
    powerImg.classList.remove("hidden");
    showPowerMap(d.power_map);
    showDoa(d.doa.az, d.doa.el);
    if (d.wav_token) {
      wavLink.href = `/api/hoa/wav/${d.wav_token}`;
      wavLink.classList.remove("hidden");
      wavLink.style.display = "";
    }
    msg("scene rendered", "ok");
  } catch (e) {
    msg(`scene failed: ${e.message}`, "error");
  }
}

// ---- tab lifecycle ------------------------------------------------------

export function init(container) {
  speakers = [
    { az: 0, el: 0 },
    { az: 90, el: 0 },
    { az: 180, el: 0 },
    { az: 270, el: 0 },
  ];

  const designer = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Speaker array"),
    el(
      "div",
      { class: "row" },
      el("label", {}, "preset"),
      el("select", { id: "hoa-preset" }, ...PRESETS.map((p) => el("option", { value: p }, p))),
      el("button", { class: "btn", id: "hoa-load-preset" }, "Load")
    ),
    el(
      "div",
      { class: "row" },
      el("label", {}, "order"),
      el(
        "select",
        { id: "hoa-order" },
        ...[1, 2, 3, 4, 5, 6, 7].map((o) =>
          el("option", { value: String(o), ...(o === 3 ? { selected: "" } : {}) }, `${o} (${(o + 1) ** 2}ch)`)
        )
      )
    ),
    el(
      "table",
      { class: "stats positions" },
      el("thead", {}, el("tr", {}, el("th", {}, "az°"), el("th", {}, "el°"), el("th", {}, ""))),
      (positionsBody = el("tbody"))
    ),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "hoa-add-spk" }, "+ speaker"),
      el("button", { class: "btn", id: "hoa-compute" }, "Compute decode matrix")
    ),
    (condEl = el("div", { class: "status-line" }, "")),
    (() => {
      const c = el("canvas", { class: "chart heat-canvas" });
      matrixHeat = makeHeatmap(c);
      return c;
    })()
  );

  const scenePanel = el(
    "div",
    { class: "panel" },
    el("h2", {}, "Scene"),
    ...[0, 1, 2, 3].map(sourceRow),
    rotSlider("rot-yaw", "yaw"),
    rotSlider("rot-pitch", "pitch"),
    rotSlider("rot-roll", "roll"),
    el(
      "div",
      { class: "btn-row" },
      el("button", { class: "btn", id: "hoa-render" }, "Render scene"),
      (wavLink = el("a", { class: "btn hidden", href: "#" }, "Download WAV"))
    ),
    (msgEl = el("div", { class: "msg" })),
    el(
      "div",
      { class: "status-line" },
      (doaEl = el("div", {}, "doa —")),
      (peakEl = el("div", {}, "peak —")),
      (energyEl = el("div", {}, "energy —"))
    ),
    (powerImg = el("img", { class: "power-img hidden", alt: "power map" }))
  );

  threeContainer = el("div", { class: "panel three-wrap" }, el("h2", {}, "Sphere (drag to orbit)"));

  // Scene panel sits NEXT TO the sphere visualizer (Bug 4: it overflowed
  // the 340px left column) — both flex inside a .panel-row, scene takes
  // the space its 4 source rows + rotation sliders actually need
  scenePanel.classList.add("scene-panel");
  container.replaceChildren(
    el(
      "div",
      { class: "lab" },
      el("div", {}, designer),
      el("div", {}, el("div", { class: "panel-row hoa-row" }, threeContainer, scenePanel))
    )
  );

  renderPositionsTable();
  initThree(threeContainer);
  showSpeakers(speakers);

  // FX toggle (post pipeline on/off, default from the global FX setting)
  const fxBox = el("input", { type: "checkbox", id: "hoa-fx" });
  fxBox.checked = fxOn;
  fxBox.addEventListener("change", () => {
    fxOn = fxBox.checked;
    setSetting("fx", fxOn);
    renderThree();
  });
  threeContainer.appendChild(
    el("label", { class: "fx-toggle" }, fxBox, el("span", {}, "FX (post shader)"))
  );

  document.getElementById("hoa-load-preset").addEventListener("click", doPreset);
  document.getElementById("hoa-compute").addEventListener("click", doSpeakers);
  document.getElementById("hoa-add-spk").addEventListener("click", () => {
    if (speakers.length < 128) {
      speakers.push({ az: 0, el: 0 });
      renderPositionsTable();
    }
  });
  document.getElementById("hoa-render").addEventListener("click", doScene);
}

export function deactivate() {
  // called by main.js on tab switch — release the WebGL context + post pass
  window.removeEventListener("themechange", applyThreeTheme);
  if (controls) controls.dispose();
  if (pipeline) pipeline.dispose();
  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
  }
  renderer = scene3d = camera = controls = sphereGroup = doaArrow = pipeline = null;
  baseMat = wireMat = markerMat = axesHelper = null;
  currentPowerMap = null;
}
