// Themed form controls (Phase 7) — native number-input spinners and the
// file-input browse button ignore the retro themes, so they are replaced
// with HUD widgets built from the /assets/icons SVGs (mask-image tinted by
// currentColor, inheriting the theme fg):
//   input[type=number] → wrapped in .num-wrap with two stepper buttons wired
//     to input.stepUp()/stepDown() (native min/max/step clamping); both
//     "input" and "change" are re-dispatched so existing listeners fire.
//   input[type=file] → visually hidden; a text-only .file-btn HUD button
//     ("[BROWSE]") proxies input.click() and shows the selected file's
//     name. (The icon variant was dropped — it clashed with the retro
//     text chrome.)
// enhanceControls(root) is idempotent per element and runs after every tab
// init (hooked in main.js — tabs build their DOM dynamically).

export function enhanceControls(root) {
  if (!root || !root.querySelectorAll) return;
  for (const input of root.querySelectorAll('input[type="number"]')) {
    _wrapNumber(input);
  }
  for (const input of root.querySelectorAll('input[type="file"]')) {
    _wrapFile(input);
  }
}

function _btn(className, icoClass, title) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = className;
  b.title = title;
  const ico = document.createElement("span");
  ico.className = `ctl-ico ${icoClass}`;
  b.append(ico);
  return b;
}

function _wrapNumber(input) {
  if (input.dataset.numEnhanced) return;
  input.dataset.numEnhanced = "1";
  const wrap = document.createElement("span");
  wrap.className = "num-wrap";
  input.replaceWith(wrap);
  wrap.append(input);
  const up = _btn("num-btn", "ico-up", "Step up");
  const down = _btn("num-btn", "ico-down", "Step down");
  const step = (dir) => {
    try {
      if (dir > 0) input.stepUp();
      else input.stepDown();
    } catch {
      return; // no steppable value (empty/invalid) — leave the input alone
    }
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  up.addEventListener("click", () => step(1));
  down.addEventListener("click", () => step(-1));
  wrap.append(up, down);
}

function _wrapFile(input) {
  if (input.dataset.fileEnhanced) return;
  input.dataset.fileEnhanced = "1";
  input.classList.add("file-hidden");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn file-btn";
  btn.title = "Browse…";
  const label = document.createElement("span");
  label.className = "file-label";
  label.textContent = "[BROWSE]";
  btn.append(label);
  btn.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    label.textContent = input.files && input.files.length ? input.files[0].name : "[BROWSE]";
  });
  input.insertAdjacentElement("afterend", btn);
}
