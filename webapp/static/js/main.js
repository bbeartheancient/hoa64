// Tab switching + lazy dynamic import of tab modules.
// Each tab module at /js/tabs/<name>.js exports init(container).
// Theme switching + display settings live in /js/settings.js (Phase 6).
// After every tab init, controls.js upgrades the fresh DOM (number steppers,
// file browse buttons) to the themed HUD widgets.

import "/js/settings.js";
import { enhanceControls } from "/js/controls.js";

const content = document.getElementById("tab-content");
const loaded = {};
let activeName = null;

async function activate(name) {
  // let the outgoing tab release resources (WebGL contexts, sockets…)
  if (activeName && activeName !== name && loaded[activeName]?.deactivate) {
    try {
      loaded[activeName].deactivate();
    } catch (e) {
      console.error(e);
    }
  }
  activeName = name;
  document
    .querySelectorAll(".tab")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  content.innerHTML = "";
  try {
    if (!loaded[name]) loaded[name] = await import(`/js/tabs/${name}.js`);
    loaded[name].init(content);
    enhanceControls(content); // themed number steppers + file browse buttons
  } catch (e) {
    content.textContent = `tab ${name} failed to load: ${e}`;
    console.error(e);
  }
}

// guard: a missing #tabs must never take down the rest of the wiring
const tabsNav = document.getElementById("tabs");
if (tabsNav) {
  tabsNav.addEventListener("click", (e) => {
    const b = e.target.closest("button.tab");
    if (!b || b.disabled) return;
    activate(b.dataset.tab);
  });
}

// ---- cross-tab deep links -------------------------------------------------
// Tabs dispatch "hoa64:open-tab" {tab, ...payload}; after activation we
// re-dispatch as "hoa64:payload" so the target tab's DOM is ready.
window.addEventListener("hoa64:open-tab", (e) => {
  const d = e.detail || {};
  if (!d.tab) return;
  activate(d.tab).then(() => {
    window.dispatchEvent(new CustomEvent("hoa64:payload", { detail: d }));
  });
});

activate("matrix_lab");
