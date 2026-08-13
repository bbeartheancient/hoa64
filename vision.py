"""Phase 3 — vision tower: project boxes/rays onto the HOA-7 sphere.

No neural net. Detections become soft spherical Gaussians in Ambix SN3D
space (same 64-D geometry as audio). Suitable as conditioning / tool output
for language models and later diffusion control.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .basis import MAX_ORDER, N_CHANNELS, sh_sn3d, unit_vector
from .encode import mix
from .report import SpatialReport, report_from_hoa


def _as_box(b: Mapping[str, Any]) -> dict:
    """Normalize a box/ray dict."""
    if "az" in b or "azimuth" in b or "azimuth_deg" in b:
        az = float(b.get("az", b.get("azimuth", b.get("azimuth_deg", 0.0))))
        el = float(b.get("el", b.get("elevation", b.get("elevation_deg", 0.0))))
    elif "x" in b and "y" in b:
        # normalized image coords: x in [0,1] left→right, y in [0,1] top→bottom
        # map to az in [-180,180] around center, el in [-90,90]
        x = float(b["x"])
        y = float(b["y"])
        # simple equirectangular pinhole-ish: center (0.5,0.5) = front
        # az: left of image = +az (left), right = -az if we want Ambix left=+Y
        # Convention: image x=0 left → az=+hfov/2, x=1 right → az=-hfov/2
        hfov = float(b.get("hfov_deg", 90.0))
        vfov = float(b.get("vfov_deg", 60.0))
        az = (0.5 - x) * hfov
        el = (0.5 - y) * vfov
    else:
        raise ValueError(f"box needs az/el or x/y: {b}")

    return {
        "az": az,
        "el": el,
        "w_deg": float(b.get("w_deg", b.get("width_deg", b.get("sigma_deg", 8.0)))),
        "h_deg": float(b.get("h_deg", b.get("height_deg", b.get("sigma_deg", 8.0)))),
        "weight": float(b.get("weight", b.get("score", b.get("confidence", 1.0)))),
        "label": str(b.get("label", b.get("class", b.get("name", "")))),
        "kind": str(b.get("kind", "box")),  # box | ray | point
    }


def angular_gaussian_weights(
    center_az: float,
    center_el: float,
    sample_az: np.ndarray,
    sample_el: np.ndarray,
    sigma_az_deg: float,
    sigma_el_deg: float,
) -> np.ndarray:
    """Soft lobe on the sphere (product of wrapped azimuth + elevation Gaussians)."""
    # great-circle-ish separation using unit vectors is better
    c = unit_vector(center_az, center_el, degrees=True)
    # sample_az, sample_el may be mesh
    u = unit_vector(sample_az, sample_el, degrees=True)
    # cos gamma = c · u
    cosg = np.clip(np.sum(u * c, axis=-1), -1.0, 1.0)
    gamma = np.rad2deg(np.arccos(cosg))
    # isotropic sigma from geometric mean of width axes
    sig = max(0.5, math.sqrt(max(sigma_az_deg, 0.5) * max(sigma_el_deg, 0.5)))
    return np.exp(-0.5 * (gamma / sig) ** 2)


def encode_boxes_to_hoa(
    boxes: Sequence[Mapping[str, Any]],
    *,
    max_order: int = MAX_ORDER,
    n_azi: int = 72,
    n_el: int = 36,
) -> np.ndarray:
    """Project one or more vision boxes/rays into a static HOA-7 field (C,).

    Each box is a soft spherical Gaussian; the field is the weighted sum of
    SN3D plane-wave encodings of the discrete sphere samples of that lobe.
    """
    if not boxes:
        return np.zeros((max_order + 1) ** 2, dtype=np.float64)

    azi = np.linspace(-180.0, 180.0, n_azi, endpoint=False)
    el = np.linspace(-90.0, 90.0, n_el)
    AA, EE = np.meshgrid(azi, el, indexing="ij")
    # solid-angle-ish weights
    w_el = np.cos(np.deg2rad(el))
    w_el = np.maximum(w_el, 0.0)
    dA = (2.0 * math.pi / n_azi) * (math.pi / max(n_el - 1, 1))
    sa = dA * w_el[None, :]  # (1,E) broadcast to (A,E)

    field = np.zeros((max_order + 1) ** 2, dtype=np.float64)
    for raw in boxes:
        b = _as_box(raw)
        if b["kind"] == "ray" or b["kind"] == "point":
            # delta: single direction plane wave
            Y = sh_sn3d(b["az"], b["el"], degrees=True, max_order=max_order)
            field = field + b["weight"] * Y
            continue
        lobe = angular_gaussian_weights(
            b["az"], b["el"], AA, EE, b["w_deg"], b["h_deg"]
        )
        # discrete HOA analysis of the lobe function f(Ω)
        # a = sum f(Ω) Y(Ω) sa(Ω) * (2n+1)/4π   — use same scale as rotate dense
        Y = sh_sn3d(AA, EE, degrees=True, max_order=max_order)  # (A,E,C)
        f = lobe * b["weight"]
        # a_c = sum_{ae} f_ae * sa_ae * Y_ae,c
        weighted = (f * sa)[..., None] * Y
        a = np.sum(weighted, axis=(0, 1))
        # SN3D analysis scale per order
        nch = a.shape[0]
        scale = np.empty(nch, dtype=np.float64)
        for n in range(max_order + 1):
            s = float(2 * n + 1)
            for m in range(-n, n + 1):
                scale[n * (n + 1) + m] = s
        a = a * scale / (4.0 * math.pi)
        field = field + a

    # pad to 64
    out = np.zeros(N_CHANNELS, dtype=np.float64)
    out[: field.shape[0]] = field
    return out


def report_from_boxes(
    boxes: Sequence[Mapping[str, Any]],
    *,
    max_order: int = MAX_ORDER,
    sample_rate: int = 48000,
    n_samples: int = 1,
    meta: Optional[dict] = None,
) -> SpatialReport:
    """Vision-only spatial report from detection boxes/rays."""
    hoa = encode_boxes_to_hoa(boxes, max_order=max_order)
    # report_from_hoa expects (C,T) — use a constant "frame"
    stream = np.tile(hoa.reshape(-1, 1), (1, max(1, n_samples)))
    notes = [f"vision tower: {len(boxes)} box/ray event(s) on HOA sphere"]
    hints = []
    for i, raw in enumerate(boxes):
        try:
            b = _as_box(raw)
            hints.append(
                {
                    "label": b["label"] or f"box{i}",
                    "azimuth_deg": b["az"],
                    "elevation_deg": b["el"],
                    "weight": b["weight"],
                    "kind": b["kind"],
                }
            )
        except Exception:
            continue
    m = dict(meta or {})
    m["encode"] = "vision_boxes"
    m["n_boxes"] = len(boxes)
    rep = report_from_hoa(
        stream,
        sample_rate,
        max_order=max_order,
        include_frames=False,
        include_bands=False,
        include_peak_map=True,
        notes=notes,
        meta=m,
    )
    rep.kind = "spatial_vision"
    rep.sources_hint = hints
    return rep


def fuse_reports(audio: Mapping[str, Any], vision: Mapping[str, Any]) -> dict:
    """Merge audio + vision spatial reports into one agent-facing payload.

    Strategy:
      - Keep both DOAs
      - angular_separation_deg between them
      - agreement flag if within 15°
      - combined one_liner
    """
    from .analysis import angular_error_deg

    a_az = float(audio.get("doa_az_deg", 0.0))
    a_el = float(audio.get("doa_el_deg", 0.0))
    v_az = float(vision.get("doa_az_deg", vision.get("peak_az_deg", 0.0)))
    v_el = float(vision.get("doa_el_deg", vision.get("peak_el_deg", 0.0)))
    sep = angular_error_deg(a_az, a_el, v_az, v_el)
    agree = sep <= 15.0

    # energy-weighted blend of unit vectors if both have energy
    from .basis import az_el_from_unit, unit_vector

    ea = float(audio.get("energy", 0.0)) + 1e-12
    ev = float(vision.get("energy", 0.0)) + 1e-12
    ua = unit_vector(a_az, a_el, degrees=True)
    uv = unit_vector(v_az, v_el, degrees=True)
    um = (ea * ua + ev * uv) / (ea + ev)
    nm = float(np.linalg.norm(um))
    if nm > 1e-15:
        baz, bel = az_el_from_unit(um / nm, degrees=True)
        baz, bel = float(baz), float(bel)
    else:
        baz, bel = a_az, a_el

    one = (
        f"spatial-fuse: audio=({a_az:.1f},{a_el:.1f}) vision=({v_az:.1f},{v_el:.1f}) "
        f"sep={sep:.1f}° agree={agree} blend=({baz:.1f},{bel:.1f})"
    )
    return {
        "schema": "spatial-hoa.fuse.v1",
        "kind": "spatial_av_fuse",
        "audio_doa_az_deg": a_az,
        "audio_doa_el_deg": a_el,
        "vision_doa_az_deg": v_az,
        "vision_doa_el_deg": v_el,
        "angular_separation_deg": sep,
        "agreement": agree,
        "blend_az_deg": baz,
        "blend_el_deg": bel,
        "audio_energy": float(audio.get("energy", 0.0)),
        "vision_energy": float(vision.get("energy", 0.0)),
        "audio_one_liner": audio.get("one_liner")
        or f"audio DOA ({a_az:.1f},{a_el:.1f})",
        "vision_one_liner": vision.get("one_liner")
        or f"vision DOA ({v_az:.1f},{v_el:.1f})",
        "vision_sources": vision.get("sources_hint") or [],
        "audio_bands": audio.get("bands") or [],
        "one_liner": one,
        "notes": [
            "Fused audio intensity DOA with vision spherical presence peak.",
            "agreement true if separation ≤ 15°.",
        ],
    }
