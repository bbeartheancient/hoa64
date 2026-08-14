// Heatmap — render a 2D array onto a canvas, two-sided ramp around zero:
// values are scaled by max |value| and mapped through the active themeRamp
// (theme.js themeRampSample): −1 → ramp start, 0 → ramp middle, +1 → ramp
// end — diverging ramps (vga/thirdman) give red/zero/blue, ordered ramps a
// signed luminance sweep. Chrome colors follow the active HUD theme;
// re-renders on themechange from the cached grid. Shared by the HOA
// decode-matrix view (Phase 4) and later tabs (terrain).

import { themeRampSample } from "/js/theme.js";

export function makeHeatmap(canvas) {
  const ctx = canvas.getContext("2d");
  let lastGrid = null;

  function render(grid) {
    lastGrid = grid;
    if (!canvas.isConnected) {
      window.removeEventListener("themechange", onTheme);
      return;
    }
    const rows = grid.length;
    const cols = rows ? grid[0].length : 0;
    canvas.width = cols || 1;
    canvas.height = rows || 1;
    const img = ctx.createImageData(canvas.width, canvas.height);
    let max = 0;
    for (const row of grid) for (const v of row) max = Math.max(max, Math.abs(v));
    const inv = max > 0 ? 1 / max : 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = Math.max(-1, Math.min(1, grid[r][c] * inv));
        const rgb = themeRampSample((t + 1) / 2);
        const k = (r * cols + c) * 4;
        img.data[k] = Math.round(rgb[0] * 255);
        img.data[k + 1] = Math.round(rgb[1] * 255);
        img.data[k + 2] = Math.round(rgb[2] * 255);
        img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function onTheme() {
    if (lastGrid) render(lastGrid);
  }
  window.addEventListener("themechange", onTheme);

  return { render };
}
