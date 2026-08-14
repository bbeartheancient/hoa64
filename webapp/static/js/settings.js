// Settings panel (Phase 6, Features 7/8) — slide-in display + processing
// settings, opened from the [SET] top-bar button and dismissed via its [X]
// button, click-outside, or Esc. The theme buttons live here (moved out of
// the top bar); the Display section is rebuilt per theme — each retro
// display exposes the controls its hardware would have had — and Processing
// holds the global FX/render switches. All state persists through theme.js
// getSetting/setSetting (localStorage "hoa64-settings"). Display
// adjustments (brightness/contrast/saturation/bivert) apply to the ENTIRE
// screen through one CSS filter on #app (see applyGlobalFilter) — canvas
// LUTs apply only theme color + quantization, nothing twice.

import {
  THEMES,
  currentTheme,
  setTheme,
  getSetting,
  setSetting,
  setCgbPalette,
} from "/js/theme.js";

const THEME_TITLES = {
  mono: "Monochrome terminal",
  green: "P1 phosphor green CRT",
  amber: "P3 amber CRT",
  plasma: "Compaq gas-plasma red/orange",
  dmg: "GameBoy DMG greenscale LCD",
  cgb: "GameBoy Color 5-bit LCD",
  vga: "256-color DOS IDE",
};

const SUBTHEME_LABELS = {
  blue: "Blue",
  cyberpunk: "Cyberpunk",
  thirdman: "Third Man",
  evangelion: "Evangelion",
};

// per-display controls (Feature 8): what each theme's Display section shows
const DISPLAY_CONTROLS = {
  mono: ["brightness", "contrast"],
  green: ["brightness", "contrast"],
  amber: ["brightness", "contrast"],
  plasma: ["brightness", "contrast", "bivert"],
  dmg: ["contrast", "motionBlur", "bivert"],
  cgb: ["brightness", "contrast", "cgbVariant", "bivert", "cgbPalette"],
  vga: ["saturation", "brightness", "contrast", "vgaSubtheme"],
};

// ---- CGB palette packs (Item 8) --------------------------------------------
// /api/palettes serves parsed emulator .pal files grouped by category; the
// selected pack ("category/name", persisted as cgbPalette) replaces the CGB
// ramp/fg/bg derivation in theme.js (setCgbPalette). _appliedPalette guards
// the re-apply on themechange against the themechange setCgbPalette fires.
let _paletteCache = null; // Promise<palette[]> — fetched once
let _appliedPalette = null;

function loadPalettes() {
  if (!_paletteCache) {
    _paletteCache = fetch("/api/palettes")
      .then((r) => (r.ok ? r.json() : { palettes: [] }))
      .then((d) => d.palettes || [])
      .catch(() => []);
  }
  return _paletteCache;
}

function applyPersistedPalette() {
  const key = currentTheme() === "cgb" ? String(getSetting("cgbPalette") || "") : "";
  if (key === _appliedPalette) return;
  _appliedPalette = key;
  if (!key) {
    setCgbPalette(null);
    return;
  }
  loadPalettes().then((list) => {
    const entry = list.find((p) => `${p.category}/${p.name}` === key);
    setCgbPalette(entry ? entry.colors : null);
  });
}

function paletteControl() {
  const select = el("select", {});
  select.append(el("option", { value: "" }, "(base CGB gradient)"));
  tag(select, () => {
    select.value = String(getSetting("cgbPalette") || "");
  });
  loadPalettes().then((list) => {
    const groups = new Map(); // category → optgroup (document order kept)
    for (const p of list) {
      const key = `${p.category}/${p.name}`;
      if (!groups.has(p.category)) {
        const g = el("optgroup", { label: p.category });
        groups.set(p.category, g);
        select.append(g);
      }
      groups.get(p.category).append(el("option", { value: key }, p.name));
    }
    select.value = String(getSetting("cgbPalette") || "");
  });
  select.addEventListener("change", () => {
    const key = select.value;
    setSetting("cgbPalette", key);
    _appliedPalette = null; // force re-apply for the new selection
    applyPersistedPalette();
  });
  return el("label", { class: "set-row" }, el("span", {}, "Palette pack"), select);
}

const SLIDERS = {
  brightness: { label: "Brightness", min: 0.25, max: 2, step: 0.05 },
  contrast: { label: "Contrast", min: 0.25, max: 2, step: 0.05 },
  saturation: { label: "Saturation", min: 0, max: 2, step: 0.05 },
  motionBlur: { label: "Motion Blur", min: 0, max: 0.9, step: 0.05 },
};

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid);
  return n;
}

const panel = document.getElementById("settings-panel");

// Every control input carries data-setting + a _refresh() that re-reads the
// setting; one settingschange listener refreshes whatever is in the DOM, so
// duplicated controls (e.g. Motion Blur in Display-dmg and Processing) and
// rebuilt sections never drift apart.
function tag(input, refresh) {
  input.setAttribute("data-setting", "");
  input._refresh = refresh;
  refresh();
  return input;
}

function sliderControl(key) {
  const spec = SLIDERS[key];
  const val = el("span", { class: "set-val" });
  const input = el("input", {
    type: "range",
    min: spec.min,
    max: spec.max,
    step: spec.step,
  });
  // NB: tag() runs the refresher immediately — `input` must already be
  // initialized before tag() is called (const TDZ, see toggleControl too)
  tag(input, () => {
    const v = Number(getSetting(key));
    input.value = v;
    val.textContent = v.toFixed(2);
  });
  input.addEventListener("input", () => setSetting(key, parseFloat(input.value)));
  return el("label", { class: "set-row" }, el("span", {}, spec.label), input, val);
}

function toggleControl(key, label, { on = true, off = false } = {}) {
  const input = el("input", { type: "checkbox" });
  tag(input, () => {
    input.checked = getSetting(key) === on;
  });
  input.addEventListener("change", () =>
    setSetting(key, input.checked ? on : off),
  );
  return el("label", { class: "set-row" }, el("span", {}, label), input);
}

function selectControl(key, label, options) {
  const select = el("select", {});
  for (const [value, text] of options) {
    select.append(el("option", { value }, text));
  }
  tag(select, () => {
    select.value = String(getSetting(key));
  });
  select.addEventListener("change", () => {
    const raw = select.value;
    const num = parseFloat(raw);
    setSetting(key, Number.isNaN(num) ? raw : num);
  });
  return el("label", { class: "set-row" }, el("span", {}, label), select);
}

// ---- sections ------------------------------------------------------------

const themeSection = el("section", { class: "set-section" });
themeSection.append(el("h3", {}, "Theme"));
const themeRow = el("div", { class: "set-themes" });
for (const [id, t] of Object.entries(THEMES)) {
  themeRow.append(
    el(
      "button",
      { class: "ts-btn", "data-theme-id": id, title: THEME_TITLES[id] || id },
      `[${t.label}]`,
    ),
  );
}
themeSection.append(themeRow);
themeRow.addEventListener("click", (e) => {
  const b = e.target.closest("button.ts-btn");
  if (b && THEMES[b.dataset.themeId]) setTheme(b.dataset.themeId);
});

function syncThemeButtons() {
  themeRow
    .querySelectorAll(".ts-btn")
    .forEach((b) =>
      b.classList.toggle("active", b.dataset.themeId === currentTheme()),
    );
}

const displaySection = el("section", { class: "set-section" });

function buildDisplaySection() {
  displaySection.innerHTML = "";
  displaySection.append(
    el("h3", {}, `Display — ${THEMES[currentTheme()].label}`),
  );
  for (const key of DISPLAY_CONTROLS[currentTheme()] || []) {
    if (SLIDERS[key]) displaySection.append(sliderControl(key));
    else if (key === "bivert") {
      displaySection.append(toggleControl("bivert", "Bivert mod"));
    } else if (key === "cgbVariant") {
      displaySection.append(
        toggleControl("cgbVariant", "Dark variant", {
          on: "dark",
          off: "light",
        }),
      );
    } else if (key === "vgaSubtheme") {
      displaySection.append(
        selectControl("vgaSubtheme", "Subtheme", Object.entries(SUBTHEME_LABELS)),
      );
    } else if (key === "cgbPalette") {
      displaySection.append(paletteControl());
    }
  }
}

const procSection = el("section", { class: "set-section" });
procSection.append(el("h3", {}, "Processing"));
procSection.append(toggleControl("fx", "FX master"));
procSection.append(
  selectControl("renderScale", "Render scale", [
    ["1", "1.0×"],
    ["0.75", "0.75×"],
    ["0.5", "0.5×"],
  ]),
);
procSection.append(sliderControl("motionBlur"));

// guard: missing panel markup must not take down main.js's module graph —
// the controls above build detached elements harmlessly and simply never show
const closeBtn = el(
  "button",
  { class: "ts-btn", id: "settings-close", title: "Close settings" },
  "[X]",
);
if (panel) {
  panel.append(
    el(
      "div",
      { class: "set-head" },
      el("h2", { class: "set-title" }, "Settings"),
      closeBtn,
    ),
    themeSection,
    displaySection,
    procSection,
  );
}

// ---- global display filter ------------------------------------------------
// brightness/contrast/saturation (+ bivert inversion on themes that expose
// the toggle) apply to the ENTIRE screen through one CSS filter on #app —
// canvases, text chrome, overlays alike, exactly once. #app wraps the whole
// UI incl. .crt-fx and this panel because filter creates a containing block
// for fixed-position descendants.

function applyGlobalFilter() {
  const app = document.getElementById("app");
  if (!app) return;
  const b = getSetting("brightness");
  const c = getSetting("contrast");
  const s = getSetting("saturation");
  // plasma inverts via the filter; dmg/cgb bivert is a FULL in-palette theme
  // reversal in theme.js (chrome var swap on <html> + LUT ramp reversal) —
  // no filter invert there, colors stay exactly in-palette
  const inv =
    getSetting("bivert") &&
    THEMES[currentTheme()].supportsBivert &&
    !THEMES[currentTheme()].bivertInLut;
  const parts = [];
  if (b !== 1) parts.push(`brightness(${b})`);
  if (c !== 1) parts.push(`contrast(${c})`);
  if (s !== 1) parts.push(`saturate(${s})`);
  if (inv) parts.push("invert(1)");
  app.style.filter = parts.length ? parts.join(" ") : "";
}

// ---- wiring ---------------------------------------------------------------

const settingsBtn = document.getElementById("settings-btn");
if (settingsBtn && panel) {
  // [SET] toggles; [X], click-outside and Esc all dismiss
  settingsBtn.addEventListener("click", () => panel.classList.toggle("open"));
  closeBtn.addEventListener("click", () => panel.classList.remove("open"));
  document.addEventListener("click", (e) => {
    if (!panel.classList.contains("open")) return;
    const t = e.target;
    if (t && t.closest && t.closest("#settings-panel, #settings-btn")) return;
    panel.classList.remove("open");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") panel.classList.remove("open");
  });
}

window.addEventListener("themechange", () => {
  syncThemeButtons();
  buildDisplaySection();
  applyGlobalFilter(); // bivert applicability depends on the theme
  applyPersistedPalette(); // no-op unless the cgb selection actually changed
});

window.addEventListener("settingschange", (e) => {
  const key = e.detail && e.detail.key;
  if (panel) {
    panel.querySelectorAll("[data-setting]").forEach((n) => {
      if (n._refresh) n._refresh();
    });
  }
  if (["brightness", "contrast", "saturation", "bivert"].includes(key)) {
    applyGlobalFilter();
  }
});

syncThemeButtons();
buildDisplaySection();
applyGlobalFilter();
applyPersistedPalette();
