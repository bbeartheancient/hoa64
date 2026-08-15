// Fixed KiCad copper palette for footprint/PCB previews.
// These canvases do NOT follow the retro theme LUT — F.Cu / B.Cu have
// to stay distinguishable on every theme.  Inner layers are reserved
// for a future 4-layer exporter (none implemented yet).
//
//   Red    F.Cu     top
//   Blue   B.Cu     bottom
//   Green  In1.Cu   inner 1 (reserved)
//   Orange In2.Cu   inner 2 (reserved)
//   —      polarity 0 / W=½ edge   unfilled (dielectric gap)

export const KICAD_BG = "#0b0d10";
export const KICAD_INK = "#d0d0d0";
export const KICAD_MUTED = "#6a6e74";

export const KICAD_CU = {
  "F.Cu": "#e31c23",
  "B.Cu": "#3b6fd8",
  "In1.Cu": "#3cb44b",
  "In2.Cu": "#e67e22",
};

export const KICAD_KEY = [
  { layer: "F.Cu", hex: KICAD_CU["F.Cu"], name: "F.Cu", means: "top copper (red)" },
  { layer: "B.Cu", hex: KICAD_CU["B.Cu"], name: "B.Cu", means: "bottom copper (blue)" },
  { layer: "In1.Cu", hex: KICAD_CU["In1.Cu"], name: "In1.Cu", means: "inner 1 (green) — reserved" },
  { layer: "In2.Cu", hex: KICAD_CU["In2.Cu"], name: "In2.Cu", means: "inner 2 (orange) — reserved" },
  { layer: "", hex: "transparent", name: "0 / gap", means: "W=½ edge — unfilled dielectric" },
];

export function copperColor(layer, kind) {
  if (kind === "keepout" || kind === "zone") return "rgba(160,160,160,0.28)";
  const L = String(layer || "");
  if (L.includes("F.Cu")) return KICAD_CU["F.Cu"];
  if (L.includes("B.Cu")) return KICAD_CU["B.Cu"];
  if (L.includes("In1.Cu")) return KICAD_CU["In1.Cu"];
  if (L.includes("In2.Cu")) return KICAD_CU["In2.Cu"];
  if (L.includes("Edge")) return "#d4c24a";
  if (L.includes("Silk") || L.includes("CrtYd") || L.includes("Fab")) return KICAD_MUTED;
  return KICAD_CU["F.Cu"];
}

/** Draw a kicad_gen / materials preview {prims,bbox} with the fixed palette.
 *  polarity === 0 cells are skipped (unfilled).  role==="feed" pads are
 *  small labeled circles. */
export function drawKicadPrims(canvas, preview, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = KICAD_BG;
  ctx.fillRect(0, 0, W, H);
  if (!preview || !preview.prims || !preview.prims.length) {
    ctx.fillStyle = KICAD_MUTED;
    ctx.font = "11px monospace";
    ctx.fillText(opts.empty || "no footprint yet", 8, H / 2);
    return;
  }
  const b = preview.bbox;
  const span = Math.max(b.xmax - b.xmin, b.ymax - b.ymin, 1e-3);
  const sc = (Math.min(W, H) - 48) / span;
  const x0 = (W - (b.xmax - b.xmin) * sc) / 2 - b.xmin * sc;
  const y0 = (H - (b.ymax - b.ymin) * sc) / 2 - b.ymin * sc;
  const X = (x) => x0 + x * sc;
  const Y = (y) => H - (y0 + y * sc);
  for (const p of preview.prims) {
    if (p.polarity === 0 && p.role !== "feed") continue; // unfilled dielectric
    const c = copperColor(p.layer, p.kind);
    ctx.strokeStyle = c;
    ctx.fillStyle = c;
    ctx.lineWidth = Math.max(1, (p.w || 0.15) * sc);
    if (p.kind === "line" && p.a && p.b) {
      ctx.beginPath();
      ctx.moveTo(X(p.a[0]), Y(p.a[1]));
      ctx.lineTo(X(p.b[0]), Y(p.b[1]));
      ctx.stroke();
    } else if ((p.kind === "poly" || p.kind === "keepout" || p.kind === "zone") && p.pts) {
      ctx.globalAlpha = p.kind === "keepout" ? 0.22 : (p.kind === "zone" ? 0.18 : 0.85);
      ctx.beginPath();
      p.pts.forEach((pt, i) => (i ? ctx.lineTo(X(pt[0]), Y(pt[1])) : ctx.moveTo(X(pt[0]), Y(pt[1]))));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    } else if (p.kind === "rect" && p.a && p.b) {
      ctx.fillRect(
        X(Math.min(p.a[0], p.b[0])),
        Y(Math.max(p.a[1], p.b[1])),
        Math.abs(p.b[0] - p.a[0]) * sc,
        Math.abs(p.b[1] - p.a[1]) * sc
      );
    } else if (p.kind === "circle" && p.c) {
      ctx.beginPath();
      ctx.arc(X(p.c[0]), Y(p.c[1]), (p.r || 0) * sc, 0, Math.PI * 2);
      ctx.stroke();
    } else if (p.kind === "pad" && p.c && p.role === "feed") {
      const r = Math.max(2.5, Math.min(p.size ? p.size[0] : 0.3, p.size ? p.size[1] : 0.3) * sc * 0.5);
      ctx.beginPath();
      ctx.arc(X(p.c[0]), Y(p.c[1]), r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = KICAD_BG;
      ctx.lineWidth = 1;
      ctx.stroke();
      if (p.name) {
        ctx.fillStyle = KICAD_INK;
        ctx.font = "9px monospace";
        ctx.fillText(p.name, X(p.c[0]) + r + 2, Y(p.c[1]) + 3);
      }
    } else if (p.kind === "pad" && p.c) {
      const sx = (p.size ? p.size[0] : 1) * sc, sy = (p.size ? p.size[1] : 1) * sc;
      ctx.fillRect(X(p.c[0]) - sx / 2, Y(p.c[1]) - sy / 2, sx, sy);
    }
  }
  ctx.fillStyle = KICAD_MUTED;
  ctx.font = "10px monospace";
  const caption = opts.caption
    || `${(b.xmax - b.xmin).toFixed(2)} × ${(b.ymax - b.ymin).toFixed(2)} mm`;
  ctx.fillText(caption, 8, H - 8);
  if (opts.legend !== false) drawLegend(ctx, W, H);
}

function drawLegend(ctx, W, H) {
  const items = [
    { hex: KICAD_CU["F.Cu"], lab: "F.Cu" },
    { hex: KICAD_CU["B.Cu"], lab: "B.Cu" },
  ];
  let x = 8, y = H - 26;
  ctx.font = "9px monospace";
  for (const it of items) {
    ctx.fillStyle = it.hex;
    ctx.fillRect(x, y, 10, 10);
    ctx.fillStyle = KICAD_INK;
    ctx.fillText(it.lab, x + 13, y + 9);
    x += 48;
  }
  ctx.strokeStyle = KICAD_MUTED;
  ctx.strokeRect(x, y, 10, 10);
  ctx.fillStyle = KICAD_INK;
  ctx.fillText("gap", x + 13, y + 9);
}
