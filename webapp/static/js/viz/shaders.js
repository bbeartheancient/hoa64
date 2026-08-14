// GLSL shader layer (Phase 4.5) — original compact shaders, no copied code.
// CRT_FRAG / DMG_FRAG: original implementations inspired by the libretro CRT
// shader family and DMG LCD shaders (NOT derived from them — those are GPL).
// ELECTRIC_FRAG: original fbm-warped field-line shader, conceptually inspired
// by shadertoy "electricity flow" pieces. QUANTUM_FRAG: original standing-wave
// interference shader, inspired by davidar.io/post/quantum-glsl (reserved for
// the Orbitals tab, Phase 5).
//
// All fragment shaders are GLSL ES 1.00, precision declared, every uniform
// used is declared. makePostPipeline is an EffectComposer-free half-res post
// pass; makeShaderCanvas is a raw-WebGL single-quad runner (Micromag Sim
// electric background).
//
// LCD ghosting (Feature 5): both runners ping-pong frame textures and blend
// the current frame with the previous one (mix(cur, prev, uGhost)) whenever
// theme.js ghostAmount() > 0 — currently the DMG theme's Motion Blur setting.
// uGhost = 0 keeps the exact pre-Feature-5 code path.

import { ghostAmount } from "/js/theme.js";

export const NOISE_GLSL = `
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
float hash21(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash31(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float vnoise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = hash31(i);
  float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
  float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
  float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
  float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
  float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
  float n111 = hash31(i + vec3(1.0, 1.0, 1.0));
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z);
}
float fbm2(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int k = 0; k < 4; k++) {
    v += a * vnoise2(p);
    p = p * 2.03 + vec2(17.3, 9.1);
    a *= 0.5;
  }
  return v;
}
float fbm3(vec3 p) {
  float v = 0.0;
  float a = 0.5;
  for (int k = 0; k < 4; k++) {
    v += a * vnoise3(p);
    p = p * 2.03 + vec3(11.7, 5.3, 7.9);
    a *= 0.5;
  }
  return v;
}
`;

export const QUAD_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// CRT post: barrel distortion (k=0.08) + 3px scanlines + 3px aperture-grille
// mask + vignette, tinted bg→fg by luminance. Original implementation.
export const CRT_FRAG = `
precision highp float;
uniform sampler2D uTex;
uniform vec2 uRes;
uniform vec3 uFg;
uniform vec3 uBg;
uniform float uIntensity;
varying vec2 vUv;

vec2 distort(vec2 uv, float k) {
  vec2 c = uv - 0.5;
  return 0.5 + c * (1.0 + k * dot(c, c));
}

void main() {
  vec2 uv = distort(vUv, 0.08);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    gl_FragColor = vec4(uBg, 1.0);
    return;
  }
  vec3 c = texture2D(uTex, uv).rgb;
  float lum = dot(c, vec3(0.299, 0.587, 0.114));
  vec3 tint = mix(uBg, uFg, lum);

  // scanlines: 3px period, darken every third line
  float row = mod(floor(uv.y * uRes.y), 3.0);
  float scan = row < 1.0 ? 0.72 : 1.0;

  // aperture grille: 3px RGB mask
  float colm = mod(floor(uv.x * uRes.x), 3.0);
  vec3 mask = vec3(
    colm < 1.0 ? 1.0 : 0.45,
    (colm >= 1.0 && colm < 2.0) ? 1.0 : 0.45,
    colm >= 2.0 ? 1.0 : 0.45);

  // vignette
  float vig = 1.0 - 0.55 * smoothstep(0.35, 0.85, length(vUv - 0.5));

  vec3 fx = tint * scan * mask * vig;
  gl_FragColor = vec4(mix(tint, fx, uIntensity), 1.0);
}
`;

// DMG post: palette quantize + 1px sub-pixel grid + optional frame-blend
// ghosting (uGhost 0 disables). Original implementation.
// Palette via uniform arrays (Item 1 fix): uPal[8] + uPalLum[8] + uPalCount,
// installed by the pipeline's setPalette() from the ACTIVE theme ramp (4
// stops for DMG; CGB's 56-stop gradient is evenly downsampled to 8).
// Quantize = NEAREST stop by luminance: the scene is already themed (light
// LCD background, dark ink), so snapping preserves it — the old uniform
// lum<0.25/0.5/0.75 band remap crushed the light bg two shades down, and the
// intermediate 256×1 DataTexture LUT (uPalTex, RGBFormat) rendered BLACK on
// the WebGL2-only three.js pipeline (unsized RGB internal-format path is
// driver-finicky — unverifiable headlessly, replaced by this
// GLSL-ES-1.00-safe uniform array, loop-index indexing only).
// uPalEnabled=0 falls back to the 4-band uRamp0..3 path. uBivert (0/1)
// evaluates the match against 1.0−lum — normally kept 0 because DMG/CGB
// bivert is now a full theme reversal upstream in theme.js (the scene
// arrives already biverted; reversing here would double-invert).
export const DMG_FRAG = `
precision highp float;
uniform sampler2D uTex;
uniform sampler2D uPrev;
uniform vec2 uRes;
uniform vec3 uRamp0;
uniform vec3 uRamp1;
uniform vec3 uRamp2;
uniform vec3 uRamp3;
uniform vec3 uPal[8];
uniform float uPalLum[8];
uniform float uPalCount;
uniform float uGhost;
uniform float uPalEnabled;
uniform float uBivert;
varying vec2 vUv;

void main() {
  vec3 c = texture2D(uTex, vUv).rgb;
  vec3 p = texture2D(uPrev, vUv).rgb;
  c = mix(c, p, uGhost);
  float lum = dot(c, vec3(0.299, 0.587, 0.114));
  vec3 col;
  if (uPalEnabled > 0.5) {
    float x = clamp(lum, 0.0, 1.0);
    if (uBivert > 0.5) x = 1.0 - x;
    col = uPal[0];
    float bd = abs(uPalLum[0] - x);
    for (int i = 1; i < 8; i++) {
      if (float(i) < uPalCount) {
        float d = abs(uPalLum[i] - x);
        if (d < bd) {
          bd = d;
          col = uPal[i];
        }
      }
    }
  } else if (lum < 0.25) col = uRamp0;
  else if (lum < 0.5) col = uRamp1;
  else if (lum < 0.75) col = uRamp2;
  else col = uRamp3;
  // 1px sub-pixel grid: darken cell borders
  vec2 g = fract(vUv * uRes);
  float grid = (g.x < 0.25 || g.y < 0.25) ? 0.82 : 1.0;
  gl_FragColor = vec4(col * grid, 1.0);
}
`;

// Electric field / flow lines — fbm-warped line field, animated by uTime,
// amplitude follows uEnergy (fed from the sim's WS stream). Original.
export const ELECTRIC_FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uEnergy;
uniform vec3 uFg;
uniform vec3 uBg;
varying vec2 vUv;

${NOISE_GLSL}

void main() {
  vec2 p = vUv * vec2(uRes.x / uRes.y, 1.0) * 3.0;
  float t = uTime * 0.25;
  // double domain warp
  vec2 q = vec2(fbm2(p + t), fbm2(p + vec2(5.2, 1.3) - t));
  vec2 r = vec2(fbm2(p + 2.5 * q + vec2(1.7, 9.2)),
                fbm2(p + 2.5 * q + vec2(8.3, 2.8) + t * 0.6));
  float f = fbm2(p + 2.0 * r);
  // field lines: ridged bands through the warped field
  float lines = abs(sin((p.y + r.x * 2.0) * 12.0 + t * 4.0));
  float glow = pow(1.0 - lines, 6.0);
  float amp = 0.15 + 0.85 * clamp(uEnergy, 0.0, 1.0);
  float v = f * 0.4 + glow * amp;
  gl_FragColor = vec4(mix(uBg, uFg, clamp(v, 0.0, 1.0)), 1.0);
}
`;

// Polarity-flux flow field — flow-trace advection along a vector field
// supplied as a texture (uField: RG = flow direction in [0,1]→[−1,1],
// B = |velocity|). Original implementation inspired by FluX particle
// flow fields and fbm advection (no copied code). Brightness follows
// |velocity| × uEnergy.
export const FLUX_FRAG = `
precision highp float;
uniform sampler2D uField;
uniform vec2 uRes;
uniform float uTime;
uniform float uEnergy;
uniform vec3 uFg;
uniform vec3 uBg;
varying vec2 vUv;

${NOISE_GLSL}

void main() {
  vec2 p = vUv;
  vec2 drift = vec2(uTime * 0.05, 0.0);
  float trail = 0.0;
  float speed = 0.0;
  for (int k = 0; k < 6; k++) {
    vec3 f = texture2D(uField, p).rgb;
    vec2 v = f.rg * 2.0 - 1.0;
    speed = f.b;
    trail += vnoise2(p * 24.0 + drift * (1.0 + 4.0 * speed));
    p -= v * 0.015;
  }
  trail /= 6.0;
  float amp = trail * (0.25 + 0.75 * speed)
            * (0.2 + 0.8 * clamp(uEnergy, 0.0, 1.0));
  gl_FragColor = vec4(mix(uBg, uFg, clamp(amp, 0.0, 1.0)), 1.0);
}
`;

// Quantum standing-wave |ψ|² interference (Orbitals tab background).
// Sum of box eigenmodes with random phases; |ψ|² as intensity. Original,
// inspired by davidar.io/post/quantum-glsl.
// Cloud coupling (Item 2): uDensity is a 64×1 texture holding the sampled
// point cloud's radial |ψ|² profile (64-bin histogram of r/extent, log1p,
// max-normalized, uploaded by the Orbitals tab after each Simulate). While
// uDensityOn = 1 the interference pattern rides the actual orbital: the
// eigenmode domain is warped radially by the profile and the final
// intensity is modulated ∝ density(r) — bright shells where the cloud
// actually concentrates. uDensityOn = 0 (no cloud loaded) → flat pattern.
export const QUANTUM_FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uEnergy;
uniform vec3 uFg;
uniform vec3 uBg;
uniform sampler2D uDensity;
uniform float uDensityOn;
varying vec2 vUv;

${NOISE_GLSL}

void main() {
  // radial density profile of the displayed cloud (flat when uDensityOn=0)
  vec2 rel = vUv - 0.5;
  float rr = clamp(length(rel) * 2.0, 0.0, 1.0);
  float dens = texture2D(uDensity, vec2(rr, 0.5)).r;
  // phase warp: the eigenmode domain is pushed radially by density(r)
  vec2 p = vUv * 3.14159265;
  float rl = max(length(rel), 0.0001);
  p += (rel / rl) * dens * uDensityOn * 1.2;
  float re = 0.0;
  float im = 0.0;
  for (int n = 1; n <= 4; n++) {
    for (int m = 1; m <= 4; m++) {
      float fn = float(n);
      float fm = float(m);
      float w = sqrt(fn * fn + fm * fm);
      float ph = hash21(vec2(fn, fm)) * 6.2831853;
      float amp = 1.0 / (fn * fm);
      float e = sin(w * uTime * 0.8 + ph);
      float s = sin(fn * p.x) * sin(fm * p.y) * amp;
      re += s * e;
      im += s * cos(w * uTime * 0.8 + ph);
    }
  }
  float psi2 = re * re + im * im;
  float v = pow(clamp(psi2 * (0.5 + 2.0 * uEnergy), 0.0, 1.5), 0.6);
  // ring brightness ∝ density(r): shells form where the cloud concentrates
  float shell = mix(1.0, 0.15 + 2.2 * dens, uDensityOn);
  gl_FragColor = vec4(mix(uBg, uFg, clamp(v * shell, 0.0, 1.0)), 1.0);
}
`;

// Frame-blend pass for LCD ghosting: mix(current, previous, uGhost).
// uGhost = 0 doubles as a plain copy to screen. Original.
export const GHOST_FRAG = `
precision highp float;
uniform sampler2D uTex;
uniform sampler2D uPrev;
uniform float uGhost;
varying vec2 vUv;

void main() {
  vec3 c = texture2D(uTex, vUv).rgb;
  vec3 p = texture2D(uPrev, vUv).rgb;
  gl_FragColor = vec4(mix(c, p, uGhost), 1.0);
}
`;

// ---- three.js post pipeline (EffectComposer-free, half-res target) -------

const MODE_FRAG = { crt: CRT_FRAG, dmg: DMG_FRAG };

function hexToVec(THREE, hex) {
  const n = parseInt(hex.slice(1), 16);
  return new THREE.Vector3(
    ((n >> 16) & 255) / 255,
    ((n >> 8) & 255) / 255,
    (n & 255) / 255
  );
}

// Palette stops for DMG_FRAG's uPal[8] uniform array (Item 1 fix): the
// active theme ramp, downsampled to ≤8 stops (CGB's 56 → 8 evenly spaced),
// each with its 0–1 Rec.601 luminance. The shader snaps each pixel to the
// NEAREST stop by luminance — the scene is already themed, so snapping (not
// a linear lum→band remap) keeps the light LCD background light. Uniform
// arrays replace the finicky DataTexture path (black viewports on the
// WebGL2-only three pipeline).
function paletteStops(hexes) {
  const n = Math.min(8, hexes.length);
  const cols = [];
  const lums = [];
  for (let i = 0; i < n; i++) {
    const hex = hexes[Math.floor((i * (hexes.length - 1)) / Math.max(1, n - 1))];
    const v = parseInt(hex.slice(1), 16);
    const r = ((v >> 16) & 255) / 255;
    const g = ((v >> 8) & 255) / 255;
    const b = (v & 255) / 255;
    cols.push([r, g, b]);
    lums.push(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return { cols, lums, n };
}

export function makePostPipeline(THREE, renderer, scene, camera, { mode = "crt" } = {}) {
  const size = renderer.getSize(new THREE.Vector2());
  // ping-pong scene targets: `cur` holds this frame's raw scene render,
  // `prev` the last one — uPrev/uGhost (DMG ghosting) sample real history.
  const makeTarget = () =>
    new THREE.WebGLRenderTarget(
      Math.max(1, Math.floor(size.x / 2)),
      Math.max(1, Math.floor(size.y / 2))
    );
  let cur = makeTarget();
  let prev = makeTarget();

  const uniforms = {
    uTex: { value: cur.texture },
    uPrev: { value: prev.texture },
    uRes: { value: new THREE.Vector2(size.x / 2, size.y / 2) },
    uFg: { value: new THREE.Vector3(1, 1, 1) },
    uBg: { value: new THREE.Vector3(0, 0, 0) },
    uIntensity: { value: 1.0 },
    uGhost: { value: 0.0 },
    uRamp0: { value: new THREE.Vector3() },
    uRamp1: { value: new THREE.Vector3() },
    uRamp2: { value: new THREE.Vector3() },
    uRamp3: { value: new THREE.Vector3() },
    uPal: { value: Array.from({ length: 8 }, () => new THREE.Vector3()) }, // DMG/CGB palette
    uPalLum: { value: [0, 0, 0, 0, 0, 0, 0, 0] },
    uPalCount: { value: 0.0 },
    uPalEnabled: { value: 0.0 },
    uBivert: { value: 0.0 }, // see DMG_FRAG — bivert happens upstream now
  };

  function setPalette(hexArray) {
    // install the theme ramp as the DMG/CGB quantization palette (≤8 stops)
    const { cols, lums, n } = paletteStops(hexArray);
    for (let i = 0; i < 8; i++) {
      const c = cols[Math.min(i, n - 1)];
      uniforms.uPal.value[i].set(c[0], c[1], c[2]);
      uniforms.uPalLum.value[i] = lums[Math.min(i, n - 1)];
    }
    uniforms.uPalCount.value = n;
    uniforms.uPalEnabled.value = 1.0;
  }

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: QUAD_VERT,
    fragmentShader: MODE_FRAG[mode] || CRT_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const postScene = new THREE.Scene();
  const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

  let enabled = mode !== "off";
  let curMode = mode;

  function setMode(m) {
    curMode = m;
    enabled = m !== "off";
    const frag = MODE_FRAG[m];
    if (frag && material.fragmentShader !== frag) {
      material.fragmentShader = frag;
      material.needsUpdate = true;
    }
  }

  function setTheme(colors) {
    // colors: {fg, bg, ramp: [hex × n]} — ramp installs the palette LUT so
    // the DMG pass quantizes to the ACTIVE theme ramp (palette packs and
    // subtheme ramps included), not a hardcoded 4-shade set
    uniforms.uFg.value.copy(hexToVec(THREE, colors.fg));
    uniforms.uBg.value.copy(hexToVec(THREE, colors.bg));
    const ramp = colors.ramp || [colors.bg, colors.bg, colors.fg, colors.fg];
    for (let i = 0; i < 4; i++) {
      uniforms[`uRamp${i}`].value.copy(hexToVec(THREE, ramp[Math.min(i, ramp.length - 1)]));
    }
    setPalette(ramp);
  }

  function render() {
    if (!enabled || curMode === "off") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return;
    }
    renderer.setRenderTarget(cur);
    renderer.render(scene, camera);
    uniforms.uTex.value = cur.texture;
    uniforms.uPrev.value = prev.texture;
    uniforms.uGhost.value = ghostAmount();
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCam);
    const t = cur;
    cur = prev;
    prev = t;
  }

  function setSize(w, h) {
    const hw = Math.max(1, Math.floor(w / 2));
    const hh = Math.max(1, Math.floor(h / 2));
    cur.setSize(hw, hh);
    prev.setSize(hw, hh);
    uniforms.uRes.value.set(hw, hh);
  }

  function dispose() {
    cur.dispose();
    prev.dispose();
    material.dispose();
  }

  return {
    setMode,
    setTheme,
    setPalette,
    render,
    setSize,
    dispose,
    get mode() {
      return curMode;
    },
  };
}

// ---- raw-WebGL single-quad runner (no three.js needed) -------------------

export function makeShaderCanvas(canvas, fragSrc, uniformsSpec = {}) {
  const gl = canvas.getContext("webgl", { antialias: false, alpha: false });
  if (!gl) return null;

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error("shader compile:", gl.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  }

  // raw WebGL has no built-in `uv` attribute — supply our own
  const vert = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;
  const vs = compile(gl.VERTEX_SHADER, vert);
  const fs = compile(gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("shader link:", gl.getProgramInfoLog(prog));
    return null;
  }
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 3, -1, -1, 3]),
    gl.STATIC_DRAW
  );
  const loc = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  // ---- LCD ghosting (Feature 5) -----------------------------------------
  // When ghostAmount() > 0 (DMG theme), each frame is rendered offscreen and
  // blended with the previous blended frame via GHOST_FRAG. Three textures:
  // `raw` (current frame), `hist[0/1]` (ping-pong blended history). Ghost
  // sampling uses units 6/7 — data textures start at unit 0 and no caller
  // uses more than one.
  const gvs = compile(gl.VERTEX_SHADER, vert);
  const gfs = compile(gl.FRAGMENT_SHADER, GHOST_FRAG);
  let ghostProg = null;
  if (gvs && gfs) {
    ghostProg = gl.createProgram();
    gl.attachShader(ghostProg, gvs);
    gl.attachShader(ghostProg, gfs);
    gl.linkProgram(ghostProg);
    if (!gl.getProgramParameter(ghostProg, gl.LINK_STATUS)) {
      console.error("ghost link:", gl.getProgramInfoLog(ghostProg));
      ghostProg = null;
    }
  }
  const gLoc = ghostProg ? gl.getAttribLocation(ghostProg, "aPos") : -1;
  const gu = {};
  if (ghostProg) {
    gl.useProgram(ghostProg);
    for (const name of ["uTex", "uPrev", "uGhost"]) {
      gu[name] = gl.getUniformLocation(ghostProg, name);
    }
    gl.uniform1i(gu.uTex, 6);
    gl.uniform1i(gu.uPrev, 7);
    gl.useProgram(prog);
  }

  function makeFrameTarget(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    for (const [pname, pval] of [
      [gl.TEXTURE_MIN_FILTER, gl.NEAREST],
      [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
      [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
    ]) {
      gl.texParameteri(gl.TEXTURE_2D, pname, pval);
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, w, h, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo };
  }

  let ghostRt = null; // {raw, hist: [t0, t1], cur: 0|1, w, h}

  function ensureGhostRt() {
    const w = canvas.width;
    const h = canvas.height;
    if (ghostRt && ghostRt.w === w && ghostRt.h === h) return;
    ghostRt = {
      raw: makeFrameTarget(w, h),
      hist: [makeFrameTarget(w, h), makeFrameTarget(w, h)],
      cur: 0,
      w,
      h,
    };
  }

  function drawGhostPass(srcTex, prevTex, ghost, targetFbo) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    gl.useProgram(ghostProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(gLoc);
    gl.vertexAttribPointer(gLoc, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, prevTex);
    gl.uniform1f(gu.uGhost, ghost);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  const uniforms = { ...uniformsSpec };
  const locations = {};

  function setUniform(name, value) {
    uniforms[name] = value;
    if (!(name in locations)) {
      locations[name] = gl.getUniformLocation(prog, name);
    }
    const l = locations[name];
    if (!l) return;
    if (typeof value === "number") gl.uniform1f(l, value);
    else if (value.length === 2) gl.uniform2f(l, value[0], value[1]);
    else if (value.length === 3) gl.uniform3f(l, value[0], value[1], value[2]);
  }

  function render(time) {
    gl.useProgram(prog);
    setUniform("uTime", time);
    setUniform("uRes", [canvas.width, canvas.height]);
    for (const [name, value] of Object.entries(uniforms)) {
      if (name !== "uTime" && name !== "uRes") setUniform(name, value);
    }
    const ghost = ghostProg ? ghostAmount() : 0;
    if (ghost <= 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return;
    }
    ensureGhostRt();
    // 1. current frame → raw target
    gl.bindFramebuffer(gl.FRAMEBUFFER, ghostRt.raw.fbo);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // 2. blend raw with previous history → next history target
    const prev = ghostRt.hist[ghostRt.cur];
    const next = ghostRt.hist[1 - ghostRt.cur];
    drawGhostPass(ghostRt.raw.tex, prev.tex, ghost, next.fbo);
    // 3. copy blended frame to screen (uGhost=0 → plain copy)
    drawGhostPass(next.tex, prev.tex, 0, null);
    ghostRt.cur = 1 - ghostRt.cur;
  }

  // data textures (e.g. flow fields) — uploaded only on new data, never per frame
  const textures = {};
  let nextUnit = 0;

  function setTexture(name, w, h, data, filter) {
    // data: Uint8Array (w*h*3) RGB; CLAMP_TO_EDGE; LINEAR unless filter given
    let entry = textures[name];
    if (!entry) {
      const loc = gl.getUniformLocation(prog, name);
      if (!loc) return; // shader doesn't use this texture — skip
      const tex = gl.createTexture();
      const unit = nextUnit++;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      for (const [pname, pval] of [
        [gl.TEXTURE_MIN_FILTER, filter || gl.LINEAR],
        [gl.TEXTURE_MAG_FILTER, filter || gl.LINEAR],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
        [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
      ]) {
        gl.texParameteri(gl.TEXTURE_2D, pname, pval);
      }
      gl.uniform1i(loc, unit);
      entry = { tex, unit };
      textures[name] = entry;
    }
    gl.activeTexture(gl.TEXTURE0 + entry.unit);
    gl.bindTexture(gl.TEXTURE_2D, entry.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, w, h, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
  }

  return { render, setUniform, setTexture, gl };
}

export function hexToRgb01(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
