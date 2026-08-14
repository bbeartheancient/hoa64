// Theme system (Phase 4.5, extended in Phase 6) — JS mirror of css/themes.css.
// setTheme(id) → <html data-theme> + localStorage + window "themechange".
// getSetting/setSetting → persisted display/processing settings (localStorage
// "hoa64-settings") + window "settingschange"; vgaSubtheme/cgbVariant are
// mirrored to <html data-subtheme>/<html data-variant> for the CSS blocks.
// recolorCanvas() LUT-recolors server PNGs (green-on-black renders):
//   quantize themes (mono/plasma/dmg/cgb): intensity (max RGB channel —
//     server ink is single-hue, Rec.601 luma would cap pure green at ~58%
//     and never reach the lightest stop) → nearest ramp stop —
//     4 stops for mono/plasma/dmg, 56 for cgb ("palette56" gradient; palette
//     packs from /api/palettes replace it via setCgbPalette). LUT ramps are
//     always FORWARD (lum 0 → darkest stop) even on inverted themes —
//     server PNGs are fixed ink-on-black renders, and walking backwards
//     turned mostly-dark PNGs into a solid light rectangle (micromag energy
//     viewport bug).
//   DMG/CGB bivert = FULL in-palette theme reversal of the entire screen
//     (a real bivert mod flips the whole LCD): the theme's CSS custom
//     properties are swapped on <html> (fg↔bg, dim↔accent — _applyChrome),
//     themeColor()/themeRampSample() follow, canvas LUT ramps reverse
//     (bivertInLut), and three.js consumers re-render on themechange. No
//     invert(1) filter for dmg/cgb — colors stay exactly in-palette.
//     Plasma's bivert keeps the global #app invert(1) filter (settings.js).
//   themeRampSample walks the ramp backwards on inverted themes so client
//     heatmaps/vertex colors follow the page chrome; bivert un-reverses
//     the walk (chrome is reversed too). Inverted-family sampling is
//     lifted to the upper 60% of the walked ramp so 3D views never
//     collapse to near-bg darkness.
//   lerp themes (green/amber/vga): two-color bg→fg by intensity; for vga,
//   themeColor("bg"/"fg") resolves the active subtheme's colors.
// mono/green/amber themeRamps are pure bg→fg luminance ramps (one hue, no
// RGB spread) — every visualizer stays inside the phosphor's hue.
// The LUT applies ONLY theme color + quantization. Display adjustments
// (brightness/contrast/saturation — and plasma's bivert invert) live in ONE
// place — a global CSS filter on #app set by settings.js — so canvases and
// text chrome are processed exactly once, together. DMG/CGB bivert is NOT a
// filter: it is the in-palette chrome swap above. Brightness additionally
// drives the phosphor glow: --glow-eff text-shadow alpha scales with
// brightness on the CRT themes (mono/green/amber/plasma/vga; the LCD
// themes dmg/cgb have no phosphor and keep glow:null).
// retintCanvas() is the tab-facing API: call after drawing a server PNG —
// it caches the pristine ImageData, registers the canvas, and re-tints on
// every themechange from the cached source (never from a tinted copy).

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function _hex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0"))
      .join("")
  );
}

function _lerpStops(anchors, n) {
  // n evenly-spaced hex stops RGB-interpolated through the anchor hexes —
  // a pure single-hue luminance ramp when the anchors share a hue
  const pts = anchors.map(hexRgb);
  if (pts.length === 1) return Array.from({ length: n }, () => anchors[0]);
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (pts.length - 1);
    const j = Math.min(pts.length - 2, Math.floor(x));
    const f = x - j;
    out.push(
      _hex(
        pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f,
        pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f,
        pts[j][2] + (pts[j + 1][2] - pts[j][2]) * f
      )
    );
  }
  return out;
}

// User-specified DMG greenscale anchors (Item 7) — exact DMG palette, and
// the anchor family the CGB 56-step backlight gradient interpolates (Item 8).
const DMG_ANCHORS = ["#1b2a09", "#0e450b", "#496b22", "#9a9e3f"]; // dark → light

const THEMES = {
  mono: {
    label: "MONO", fg: "#f2f2f2", bg: "#000000", dim: "#8a8a8a",
    faint: "#262626", accent: "#ffffff", glow: "#f2f2f233", inverted: false,
    quantize: true, ramp: _lerpStops(["#000000", "#f2f2f2"], 4),
  },
  green: {
    label: "P1", fg: "#33ff66", bg: "#020603", dim: "#1f7a3d",
    faint: "#0a2614", accent: "#baffd0", glow: "#33ff6644", inverted: false,
    quantize: false, ramp: _lerpStops(["#020603", "#33ff66"], 4),
  },
  amber: {
    label: "AMB", fg: "#ffb000", bg: "#080400", dim: "#8a5c00",
    faint: "#241500", accent: "#ffe1a0", glow: "#ffb00044", inverted: false,
    quantize: false, ramp: _lerpStops(["#080400", "#ffb000"], 4),
  },
  plasma: {
    label: "PLS", fg: "#ff4a1f", bg: "#0d0200", dim: "#7a2410",
    faint: "#260801", accent: "#ffb59d", glow: "#ff5a2a66", inverted: false,
    quantize: true, ramp: ["#1a0500", "#8a1f08", "#ff4a1f", "#ffb59d"],
    supportsBivert: true, // global invert(1) filter — plasma has no fixed palette
  },
  dmg: {
    // Exact DMG palette (Item 7). Bivert = FULL in-palette theme reversal
    // (Item 3): chrome vars swap on <html> (_applyChrome) AND the LUT ramp
    // reverses (bivertInLut) — inverted pixels stay inside the 4 shades.
    label: "DMG", fg: "#1b2a09", bg: "#9a9e3f", dim: "#0e450b",
    faint: "#496b22", accent: "#1b2a09", glow: null, inverted: true,
    quantize: true, ramp: [...DMG_ANCHORS],
    supportsBivert: true, bivertInLut: true,
  },
  cgb: {
    // Game Boy Color LCD (Item 8): the DMG anchor family expanded to a
    // 56-step gradient. quantize:"palette56" maps luminance → nearest of the
    // 56 stops (the old 5-bit channel snap is gone). Palette packs replace
    // the whole derivation via setCgbPalette; bivert reverses the ramp.
    label: "CGB", fg: "#1b2a09", bg: "#9a9e3f", dim: "#0e450b",
    faint: "#496b22", accent: "#0e450b", glow: null, inverted: true,
    quantize: "palette56", ramp: _lerpStops(DMG_ANCHORS, 56),
    supportsBivert: true, bivertInLut: true,
  },
  vga: {
    label: "VGA", fg: "#c0c0c0", bg: "#0000a8", dim: "#7a7ac0",
    faint: "#10106a", accent: "#ffff55", glow: "#c0c0c033", inverted: false,
    quantize: false, ramp: ["#000055", "#5555ff", "#55ffff", "#ffffff"],
  },
};

// VGA subtheme visualizer ramps (Feature 8) + viewport colors (Item 7) —
// each subtheme carries its own bg/fg so themeColor("bg")/themeRamp() and
// the 2D LUT follow the subtheme, not just the CSS chrome (whose overrides
// live in themes.css [data-theme="vga"][data-subtheme="…"] blocks).
const VGA_SUBTHEMES = {
  blue: { bg: "#0000a8", fg: "#c0c0c0", ramp: ["#000055", "#5555ff", "#55ffff", "#ffffff"] }, // classic VGA
  cyberpunk: { bg: "#000000", fg: "#ffe600", ramp: ["#000000", "#004038", "#00e0c0", "#b0fff0"] }, // teal grid
  thirdman: { bg: "#000000", fg: "#f0f0f0", ramp: ["#e02020", "#f0f0f0", "#2050e0"] }, // diverging: red/white/blue
  evangelion: { bg: "#000000", fg: "#ffe600", ramp: ["#2a0a4a", "#6a1a9a", "#c02050", "#ff4020"] }, // accent #ff2020
};

const LS_KEY = "hoa64-theme";
let _current = "mono";

// ---- persisted settings (Features 7/8) -----------------------------------

const SETTINGS_KEY = "hoa64-settings";
const DEFAULTS = {
  fx: true, // master post-processing toggle (crt-fx / shader post passes)
  renderScale: 1.0, // multiplier on the per-tab pixelRatio cap
  motionBlur: 0.65, // DMG LCD ghosting amount (0–0.9)
  brightness: 1.0,
  contrast: 1.0,
  saturation: 1.0,
  bivert: false, // plasma: full-screen invert(1) filter; dmg/cgb: full in-palette theme reversal
  cgbPalette: "", // "category/name" of a /api/palettes pack ("" = base CGB)
  cgbVariant: "light", // "light" | "dark" — <html data-variant>
  vgaSubtheme: "blue", // key into VGA_SUBTHEMES — <html data-subtheme>
};
let _settings = { ...DEFAULTS };

function _loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) _settings = { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* private mode, corrupt JSON — keep defaults */
  }
}

function _saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(_settings));
  } catch {
    /* ignore */
  }
}

function _applySettingAttrs() {
  const root = document.documentElement;
  root.setAttribute("data-subtheme", _settings.vgaSubtheme);
  root.setAttribute("data-variant", _settings.cgbVariant);
}

export function getSetting(key) {
  return _settings[key] ?? DEFAULTS[key];
}

export function setSetting(key, value) {
  if (!(key in DEFAULTS)) return;
  _settings[key] = value;
  _saveSettings();
  if (key === "vgaSubtheme" || key === "cgbVariant" || key === "bivert") {
    _applySettingAttrs();
    _applyChrome(); // bivert swaps the dmg/cgb chrome vars — before dispatch
    // these change the visual identity (CSS chrome vars + themeRamp/LUT) —
    // canvas/three consumers re-render on themechange, so fire it too
    // (otherwise e.g. VGA subtheme switches leave panels blue until the
    // next full theme switch, and a DMG/CGB bivert never re-tints).
    window.dispatchEvent(new CustomEvent("themechange", { detail: _current }));
  }
  window.dispatchEvent(
    new CustomEvent("settingschange", { detail: { key, value } }),
  );
}

export { THEMES, VGA_SUBTHEMES };

export function currentTheme() {
  return _current;
}

export function setTheme(id) {
  if (!THEMES[id]) return;
  _current = id;
  document.documentElement.setAttribute("data-theme", id);
  _applyChrome(); // palette-pack + bivert chrome overrides, before dispatch
  _applyGlow();
  try {
    localStorage.setItem(LS_KEY, id);
  } catch {
    /* private mode etc. */
  }
  window.dispatchEvent(new CustomEvent("themechange", { detail: id }));
}

export function themeColor(name) {
  if (_current === "vga" && (name === "bg" || name === "fg")) {
    // viewport colors follow the subtheme (Item 7) — THEMES.vga.bg/.fg are
    // only the "blue" subtheme's values
    const sub = VGA_SUBTHEMES[_settings.vgaSubtheme] || VGA_SUBTHEMES.blue;
    return sub[name];
  }
  // effective colors: base theme → CGB palette pack → bivert swap (Item 3)
  const cols = _chromeColors();
  return cols[name] ?? cols.fg;
}

export function themeRamp() {
  // Visualizer color stops for the active theme (hex array, dark→light —
  // or diverging neg→zero→pos for 3-stop ramps like vga/thirdman). VGA
  // follows the active subtheme, CGB the selected palette pack; other
  // themes return their own ramp (56 stops for the base CGB gradient).
  if (_current === "vga") {
    const sub = VGA_SUBTHEMES[_settings.vgaSubtheme] || VGA_SUBTHEMES.blue;
    return [...sub.ramp];
  }
  if (_current === "cgb" && _cgbPalette) return [..._cgbPalette.ramp];
  return [...THEMES[_current].ramp];
}

// ---- CGB palette packs (Item 8) + DMG/CGB bivert chrome swap (Item 3) ------
// A palette pack (GET /api/palettes — parsed emulator .pal files) replaces
// the whole CGB color derivation: darkest color → fg, lightest → bg, dim and
// faint at the lower/upper thirds, and the ramp is the palette sorted by
// luminance then RGB-interpolated to 56 stops.
// DMG/CGB bivert is a FULL in-palette theme reversal (like a bivert mod
// flips the whole LCD): fg↔bg and dim↔accent swap, the LUT ramp reverses
// (_lutForTheme), and themeRampSample un-reverses its inverted walk.
// Chrome overrides (palette and/or bivert) ride as inline
// --fg/--bg/--dim/--faint/--accent on <html>, beating themes.css.
let _cgbPalette = null; // {ramp, fg, bg, dim, faint} | null

function _luminance(hex) {
  const [r, g, b] = hexRgb(hex);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function _biverted() {
  // full theme-reversal bivert — only the fixed-palette LCD themes; plasma's
  // bivert stays the settings.js #app invert(1) filter
  return !!(THEMES[_current].bivertInLut && _settings.bivert);
}

function _chromeColors() {
  // effective theme colors: base theme → CGB palette pack → bivert swap
  const t = THEMES[_current];
  let c = { fg: t.fg, bg: t.bg, dim: t.dim, faint: t.faint, accent: t.accent };
  if (_current === "cgb" && _cgbPalette) {
    c = {
      fg: _cgbPalette.fg,
      bg: _cgbPalette.bg,
      dim: _cgbPalette.dim,
      faint: _cgbPalette.faint,
      accent: _cgbPalette.dim,
    };
  }
  if (_biverted()) {
    c = { fg: c.bg, bg: c.fg, dim: c.accent, faint: c.faint, accent: c.dim };
  }
  return c;
}

function _applyChrome() {
  const root = document.documentElement;
  const override = (_current === "cgb" && _cgbPalette) || _biverted();
  const cols = override ? _chromeColors() : null;
  for (const n of ["fg", "bg", "dim", "faint", "accent"]) {
    if (cols) root.style.setProperty(`--${n}`, cols[n]);
    else root.style.removeProperty(`--${n}`);
  }
}

export function setCgbPalette(colors) {
  // colors: hex array from a palette pack (≥ 2 entries), or null → base CGB
  if (Array.isArray(colors) && colors.length >= 2) {
    const sorted = [...colors].sort((a, b) => _luminance(a) - _luminance(b));
    _cgbPalette = {
      ramp: _lerpStops(sorted, 56),
      fg: sorted[0],
      bg: sorted[sorted.length - 1],
      dim: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length / 3))],
      faint: sorted[Math.min(sorted.length - 1, Math.floor((2 * sorted.length) / 3))],
    };
  } else {
    _cgbPalette = null;
  }
  _applyChrome();
  window.dispatchEvent(new CustomEvent("themechange", { detail: _current }));
}

// ---- brightness-driven phosphor glow (Item 5) -------------------------------
// CRT phosphor halos scale with the Brightness setting: --glow-eff is the
// theme's glow color with its alpha multiplied by 2·clamp(b−0.5, 0, 1) —
// b=1 → the theme's base glow, b=1.5 → double, b=0.5 → none. LCD themes
// (dmg/cgb, glow:null) have no phosphor and keep the shadow transparent.
function _applyGlow() {
  const root = document.documentElement;
  const g = THEMES[_current].glow;
  if (!g) {
    root.style.setProperty("--glow-eff", "transparent");
    return;
  }
  const b = Number(_settings.brightness) || 1;
  const s = 2 * Math.max(0, Math.min(1, b - 0.5));
  const base = parseInt(g.slice(7), 16); // alpha byte of the #rrggbbaa glow
  const a = Math.round(base * s);
  root.style.setProperty(
    "--glow-eff",
    a > 0 ? g.slice(0, 7) + a.toString(16).padStart(2, "0") : "transparent"
  );
}

window.addEventListener("settingschange", (e) => {
  if (e.detail && e.detail.key === "brightness") _applyGlow();
});

export function themeRampSample(u) {
  // Sample the active themeRamp() at u ∈ [0,1], piecewise-linear through
  // the stops; returns [r, g, b] floats in 0–1 (visualizer vertex colors,
  // heatmaps). Inverted themes (dmg/cgb) walk the ramp backwards, so u=0
  // lands near the (light) background and u=1 on the dark ink — matching
  // the old bg→fg lerp semantics. Bivert reverses the chrome too, so it
  // UN-reverses this walk (u=0 near the now-dark background again).
  // Inverted-family themes additionally lift the sampling range to the
  // upper 60% of the walked ramp: u always measures distance from the
  // bg-adjacent end, and low values must not collapse to near-bg darkness
  // (3D views on DMG/CGB were nearly invisible).
  const t = THEMES[_current];
  const inv = t.inverted !== _biverted(); // XOR — bivert swaps the chrome
  const stops = (inv ? [...themeRamp()].reverse() : themeRamp()).map(hexRgb);
  if (stops.length === 1) return stops[0].map((v) => v / 255);
  let xu = Math.max(0, Math.min(1, u));
  if (t.inverted) xu = 0.4 + 0.6 * xu; // min contrast: ≥40% away from bg
  const x = xu * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    (a[0] + (b[0] - a[0]) * f) / 255,
    (a[1] + (b[1] - a[1]) * f) / 255,
    (a[2] + (b[2] - a[2]) * f) / 255,
  ];
}

export function ghostAmount() {
  // LCD ghosting factor for shader canvases (Feature 5): only the DMG theme
  // emulates panel response; the Motion Blur slider sets the amount (0–0.9).
  return _current === "dmg" ? Number(_settings.motionBlur) || 0 : 0;
}

// restore persisted theme + settings at module load (before first paint)
(function initTheme() {
  _loadSettings();
  let saved = null;
  try {
    saved = localStorage.getItem(LS_KEY);
  } catch {
    /* ignore */
  }
  if (saved && THEMES[saved]) _current = saved;
  document.documentElement.setAttribute("data-theme", _current);
  _applySettingAttrs();
  _applyChrome(); // persisted bivert / restored palette chrome swap
  _applyGlow();
})();

// ---- canvas recoloring --------------------------------------------------

const _registered = new Set();

function _lutForTheme() {
  const t = THEMES[_current];
  const lut = new Uint8Array(256 * 3);
  let ramp = themeRamp(); // includes vga subtheme + cgb palette-pack ramps
  // The LUT ramp is always FORWARD (lum 0 → darkest stop): server PNGs are
  // rendered ink-on-black by fixed module code, so dark pixels must map to
  // the theme's dark end on every theme, inverted (dmg/cgb) included — an
  // inverted walk turned mostly-dark PNGs into a solid light rectangle (the
  // micromag energy-viewport bug). Only bivert reverses the ramp, keeping
  // inverted pixels inside the palette.
  if (t.bivertInLut && _settings.bivert) ramp.reverse(); // DMG/CGB bivert
  if (t.quantize) {
    // luminance → nearest ramp stop: 4 shades for mono/plasma/dmg, the 56
    // gradient stops for cgb "palette56" (the old 5-bit snap is gone)
    const cols = ramp.map(hexRgb);
    for (let i = 0; i < 256; i++) {
      const k = Math.min(cols.length - 1, Math.floor((i / 256) * cols.length));
      const [r, g, b] = cols[k];
      lut[i * 3] = r;
      lut[i * 3 + 1] = g;
      lut[i * 3 + 2] = b;
    }
  } else {
    // lerp themes (green/amber/vga): themeColor resolves the vga SUBTHEME's
    // bg/fg so e.g. cyberpunk viewports go black→yellow instead of blue.
    const bg = hexRgb(themeColor("bg"));
    const fg = hexRgb(themeColor("fg"));
    for (let i = 0; i < 256; i++) {
      const f = i / 255;
      lut[i * 3] = Math.round(bg[0] + (fg[0] - bg[0]) * f);
      lut[i * 3 + 1] = Math.round(bg[1] + (fg[1] - bg[1]) * f);
      lut[i * 3 + 2] = Math.round(bg[2] + (fg[2] - bg[2]) * f);
    }
  }
  return lut;
}

// DMG visualizer LUT semantics, exposed for tests and for any code that
// needs the exact 4-shade mapping without going through a canvas. Server
// PNGs are rendered ink-on-black, so the mapping is always lum 0 → Dark
// (DMG_ANCHORS[0]) … lum 255 → Light (DMG_ANCHORS[3]); bivert reverses it.
// This never depends on the active theme or page chrome.
export function dmgLut(bivert = false) {
  const ramp = bivert ? [...DMG_ANCHORS].reverse() : [...DMG_ANCHORS];
  return (lum) =>
    ramp[Math.min(3, Math.floor((Math.max(0, Math.min(255, lum)) / 256) * 4))];
}

function _applyTint(canvas) {
  const src = canvas._retintSrc;
  if (!src) return;
  const ctx = canvas.getContext("2d");
  if (_current === "green") {
    // server renders are already P1 green-on-black — show pristine
    ctx.putImageData(src, 0, 0);
    return;
  }
  const lut = _lutForTheme();
  const out = new ImageData(src.width, src.height);
  const s = src.data;
  const d = out.data;
  for (let i = 0; i < s.length; i += 4) {
    // intensity = max channel, NOT Rec.601 luma: server renders are
    // single-hue ink on black, and luma caps pure green (0,255,0) at ~149 —
    // full-bright ink never reached the lightest DMG shade (Item 3)
    const lum = Math.max(s[i], s[i + 1], s[i + 2]);
    const k = lum * 3;
    d[i] = lut[k];
    d[i + 1] = lut[k + 1];
    d[i + 2] = lut[k + 2];
    d[i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

export function recolorCanvas(canvas, { src = null } = {}) {
  // low-level: recolor `canvas` (or the given ImageData) with the current theme
  if (src) canvas._retintSrc = src;
  else if (!canvas._retintSrc) {
    canvas._retintSrc = canvas
      .getContext("2d")
      .getImageData(0, 0, canvas.width, canvas.height);
  }
  _applyTint(canvas);
}

export function retintCanvas(canvas) {
  // call right after drawing a server PNG onto `canvas`: captures the
  // pristine pixels, registers for themechange re-tints, applies now
  canvas._retintSrc = canvas
    .getContext("2d")
    .getImageData(0, 0, canvas.width, canvas.height);
  _registered.add(canvas);
  _applyTint(canvas);
}

function _retintAll() {
  for (const c of _registered) {
    if (c.isConnected) _applyTint(c);
    else _registered.delete(c);
  }
}

window.addEventListener("themechange", _retintAll);
