// Strip chart — hand-rolled scrolling line plot on a canvas (no chart lib).
// makeStripChart(canvas, {seriesName: color}, {maxPoints}) -> { push, clear,
// setVisible }. push({seriesName: value}) appends one sample; each series
// keeps the last maxPoints values and autoscales y across all VISIBLE
// series. x is the sample index (step frames arrive unevenly — wall-clock
// spacing lies). setVisible(name, on) toggles a series (lines, autoscale
// and legend) without dropping its data. Chrome colors (bg, text,
// zero-line) follow the active HUD theme and the chart re-renders on
// themechange. Series colors are caller-chosen; pass colors = null to let
// the chart pick them from the active themeRamp (Feature 8) — defaulted
// series are discovered on first push and re-picked on every themechange.

import { themeColor, themeRamp } from "/js/theme.js";

export function makeStripChart(canvas, colors = null, { maxPoints = 600 } = {}) {
  const ctx = canvas.getContext("2d");
  const defaulted = !colors;
  const explicit = colors || {};
  const data = defaulted
    ? {}
    : Object.fromEntries(Object.keys(explicit).map((n) => [n, []]));
  const hidden = new Set();

  function names() {
    return Object.keys(data).filter((n) => !hidden.has(n));
  }

  function colorFor(name, idx) {
    if (!defaulted) return explicit[name];
    // skip the first (near-bg) stop so lines stay visible
    const stops = themeRamp();
    return stops[(idx % (stops.length - 1)) + 1];
  }

  function draw() {
    if (!canvas.isConnected) {
      window.removeEventListener("themechange", draw);
      return;
    }
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = themeColor("bg");
    ctx.fillRect(0, 0, w, h);

    let lo = Infinity;
    let hi = -Infinity;
    for (const n of names()) {
      for (const v of data[n]) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) {
      ctx.fillStyle = themeColor("dim");
      ctx.font = "11px monospace";
      ctx.fillText("no data", 8, h / 2);
      return;
    }
    if (hi - lo < 1e-12) {
      hi = lo + 1;
      lo -= 1;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    const yOf = (v) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8);

    // zero line when in range
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = themeColor("faint");
      ctx.beginPath();
      ctx.moveTo(0, yOf(0));
      ctx.lineTo(w, yOf(0));
      ctx.stroke();
    }

    names().forEach((n) => {
      const idx = Object.keys(data).indexOf(n); // stable color slot
      const pts = data[n];
      if (pts.length < 2) return;
      ctx.strokeStyle = colorFor(n, idx);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const x = (i / (maxPoints - 1)) * (w - 4) + 2;
        const y = yOf(pts[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // legend + y range
    ctx.font = "10px monospace";
    let lx = 8;
    names().forEach((n) => {
      const idx = Object.keys(data).indexOf(n); // stable color slot
      ctx.fillStyle = colorFor(n, idx);
      const last = data[n].length ? data[n][data[n].length - 1] : null;
      const label = last === null ? n : `${n} ${fmt(last)}`;
      ctx.fillText(label, lx, 12);
      lx += ctx.measureText(label).width + 14;
    });
    ctx.fillStyle = themeColor("dim");
    ctx.fillText(fmt(hi), 8, 26);
    ctx.fillText(fmt(lo), 8, h - 6);
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-2)) return v.toExponential(1);
    return a >= 100 ? v.toFixed(0) : v.toFixed(2);
  }

  window.addEventListener("themechange", draw);
  draw();

  return {
    push(sample) {
      for (const n of Object.keys(sample)) {
        if (!defaulted && !(n in explicit)) continue;
        if (!(n in data)) data[n] = [];
        const v = sample[n];
        if (typeof v === "number" && Number.isFinite(v)) {
          data[n].push(v);
          if (data[n].length > maxPoints) data[n].shift();
        }
      }
      draw();
    },
    clear() {
      for (const n of Object.keys(data)) data[n].length = 0; // incl. hidden
      draw();
    },
    setVisible(name, on) {
      // toggle a series without dropping its samples; autoscale + legend
      // follow the visible set (names() filters hidden series)
      if (on) hidden.delete(name);
      else hidden.add(name);
      draw();
    },
    isVisible(name) {
      return !hidden.has(name);
    },
  };
}
