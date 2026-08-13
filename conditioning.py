"""Diffusion / generative conditioning from spatial reports.

Turns HOA calculator output into:
  * plain-text control lines for T2I / T2V prompts
  * structured JSON for ControlNet-style / custom nodes
  * optional ComfyUI API prompt payload (if Comfy is running on :8188)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, Path]


def _get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def panner_report(
    az_deg: float = 0.0,
    el_deg: float = 0.0,
    w_amplitude: float = 0.5,
) -> dict:
    """Synthetic spatial report from a UI spherical panner + W gain.

    W (omnidirectional HOA channel) maps to field size / POV:
      low W  → tight / subject-focused / narrow FOV
      high W → wide / environmental / immersive FOV
    """
    w = max(0.0, float(w_amplitude))
    return {
        "kind": "spatial_panner",
        "doa_az_deg": float(az_deg),
        "doa_el_deg": float(el_deg),
        "w_amplitude": w,
        "energy": w,
        "one_liner": (
            f"panner az={float(az_deg):.0f}° el={float(el_deg):.0f}° W={w:.2f}"
        ),
        "meta": {
            "source": "ui_panner",
            "field_width_deg": _w_to_field_width_deg(w),
        },
    }


def _w_to_field_width_deg(w: float) -> float:
    """Map W amplitude (0..1+) to an angular field width in degrees."""
    w = max(0.0, min(2.0, float(w)))
    # 8° pin-point → ~160° ultra-wide at W=1, and beyond at W>1
    return 8.0 + 152.0 * min(1.0, w) + 40.0 * max(0.0, w - 1.0)


def _w_to_field_language(w: float, *, style: str = "natural") -> str:
    """Natural language for field size / camera POV from W amplitude."""
    w = max(0.0, float(w))
    width = _w_to_field_width_deg(w)
    if style == "tags":
        if w < 0.25:
            return f"spatial-fov-tight, spatial-w-{w:.2f}, field-{width:.0f}deg"
        if w < 0.5:
            return f"spatial-fov-medium, spatial-w-{w:.2f}, field-{width:.0f}deg"
        if w < 0.75:
            return f"spatial-fov-wide, spatial-w-{w:.2f}, field-{width:.0f}deg"
        return f"spatial-fov-immersive, spatial-w-{w:.2f}, field-{width:.0f}deg"
    if style == "technical":
        return f"W={w:.3f} field_width_deg={width:.1f}"
    if w < 0.2:
        return (
            f"tight close-up POV, narrow field of view (~{width:.0f}°), "
            "subject fills the frame, shallow spatial field"
        )
    if w < 0.4:
        return (
            f"medium-close framing, moderate field of view (~{width:.0f}°), "
            "subject-focused with limited environment"
        )
    if w < 0.6:
        return (
            f"natural mid-shot POV, balanced field of view (~{width:.0f}°), "
            "subject and surrounding space equally present"
        )
    if w < 0.8:
        return (
            f"wide environmental framing (~{width:.0f}°), expansive field, "
            "subject placed in a larger spatial context"
        )
    return (
        f"ultra-wide immersive POV (~{width:.0f}°), large ambient field, "
        "surrounding space dominates over any single subject"
    )


def spatial_prompt_fragment(
    report: Mapping[str, Any],
    *,
    style: str = "natural",
) -> str:
    """Short natural-language spatial control for diffusion prompts.

    style: natural | tags | technical

    When ``w_amplitude`` (or fallback ``energy``) is present, appends field
    size / POV language so UI panners can drive framing as well as direction.
    """
    kind = str(report.get("kind", "spatial_field"))
    w_raw = _get(report, "w_amplitude", default=None)
    # Only treat energy as W when the report is from a UI panner (or W is explicit).
    if w_raw is None and kind == "spatial_panner":
        w_raw = _get(report, "energy", default=None)

    if kind == "spatial_av_fuse" or "blend_az_deg" in report:
        a_az = float(_get(report, "audio_doa_az_deg", default=0))
        a_el = float(_get(report, "audio_doa_el_deg", default=0))
        v_az = float(_get(report, "vision_doa_az_deg", default=0))
        v_el = float(_get(report, "vision_doa_el_deg", default=0))
        sep = float(_get(report, "angular_separation_deg", default=0))
        agree = bool(_get(report, "agreement", default=False))
        b_az = float(_get(report, "blend_az_deg", default=a_az))
        b_el = float(_get(report, "blend_el_deg", default=a_el))
        if style == "tags":
            base = (
                f"spatial-az-{b_az:.0f}, spatial-el-{b_el:.0f}, "
                f"av-{'aligned' if agree else 'offset'}-{sep:.0f}deg"
            )
        elif style == "technical":
            base = (
                f"HOA control: blend_az={b_az:.1f} blend_el={b_el:.1f} "
                f"audio=({a_az:.1f},{a_el:.1f}) vision=({v_az:.1f},{v_el:.1f}) "
                f"sep={sep:.1f} agree={agree}"
            )
        else:
            side = _az_to_side(b_az)
            height = _el_to_height(b_el)
            align = (
                "sound and subject co-located"
                if agree
                else f"sound and subject separated by {sep:.0f} degrees"
            )
            base = (
                f"camera/listener facing forward; primary subject {side}, {height}; "
                f"{align}; spatial azimuth {b_az:.0f}°, elevation {b_el:.0f}°"
            )
        if w_raw is not None:
            base = f"{base}; {_w_to_field_language(float(w_raw), style=style)}"
        return base

    az = float(_get(report, "doa_az_deg", "peak_az_deg", default=0))
    el = float(_get(report, "doa_el_deg", "peak_el_deg", default=0))
    if style == "tags":
        base = f"spatial-az-{az:.0f}, spatial-el-{el:.0f}"
    elif style == "technical":
        base = f"HOA control: az={az:.1f} el={el:.1f} kind={kind}"
    else:
        base = (
            f"primary direction {_az_to_side(az)}, {_el_to_height(el)}; "
            f"azimuth {az:.0f}°, elevation {el:.0f}°"
        )
    if w_raw is not None:
        base = f"{base}; {_w_to_field_language(float(w_raw), style=style)}"
    return base


def _az_to_side(az: float) -> str:
    # Ambix: +az = left
    if -20 <= az <= 20:
        return "in front of the camera"
    if 20 < az <= 70:
        return "to the front-left"
    if 70 < az <= 110:
        return "on the left"
    if az > 110 or az < -110:
        return "behind the camera"
    if -70 <= az < -20:
        return "to the front-right"
    return "on the right"


def _el_to_height(el: float) -> str:
    if el > 25:
        return "above eye level"
    if el < -25:
        return "below eye level"
    return "near eye level"


def build_conditioning(
    report: Mapping[str, Any],
    *,
    base_prompt: str = "",
    negative_prompt: str = "",
    style: str = "natural",
) -> dict:
    """Structured conditioning payload for generative pipelines."""
    frag = spatial_prompt_fragment(report, style=style)
    if base_prompt:
        positive = f"{base_prompt.rstrip(', ').rstrip()}, {frag}"
    else:
        positive = frag
    control = {
        "schema": "spatial-hoa.conditioning.v1",
        "spatial_fragment": frag,
        "positive_prompt": positive,
        "negative_prompt": negative_prompt,
        "control_vector": {
            "az_deg": float(
                _get(
                    report,
                    "blend_az_deg",
                    "doa_az_deg",
                    "peak_az_deg",
                    default=0.0,
                )
            ),
            "el_deg": float(
                _get(
                    report,
                    "blend_el_deg",
                    "doa_el_deg",
                    "peak_el_deg",
                    default=0.0,
                )
            ),
            "energy": float(_get(report, "energy", "audio_energy", default=0.0)),
            "agreement": _get(report, "agreement", default=None),
            "angular_separation_deg": _get(
                report, "angular_separation_deg", default=None
            ),
        },
        "source_report_kind": report.get("kind"),
        "one_liner": report.get("one_liner") or frag,
    }
    return control


def list_comfy_checkpoints(base_url: str = "http://127.0.0.1:8188") -> list[str]:
    """Ask ComfyUI which ckpt_name values are valid."""
    url = base_url.rstrip("/") + "/object_info/CheckpointLoaderSimple"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        node = data.get("CheckpointLoaderSimple") or data
        choices = (
            node.get("input", {})
            .get("required", {})
            .get("ckpt_name", [[]])[0]
        )
        # filter non-checkpoint junk (e.g. sam *.pth)
        return [
            c
            for c in choices
            if isinstance(c, str)
            and c.endswith((".safetensors", ".ckpt"))
            and "sam_" not in c.lower()
        ]
    except Exception:
        return []


def resolve_comfy_checkpoint(
    preferred: str | None = None,
    *,
    base_url: str = "http://127.0.0.1:8188",
) -> str:
    """Pick a checkpoint that exists on this ComfyUI install."""
    available = list_comfy_checkpoints(base_url)
    if preferred and preferred in available:
        return preferred
    # Prefer SDXL base, then any non-pony XL, then first available
    for name in available:
        if name == "sd_xl_base_1.0.safetensors":
            return name
    for name in available:
        if "xl" in name.lower() or "sdxl" in name.lower():
            return name
    if available:
        return available[0]
    # Offline fallback — may 400 if not installed
    return preferred or "sd_xl_base_1.0.safetensors"


def comfy_txt2img_payload(
    conditioning: Mapping[str, Any],
    *,
    checkpoint: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int = 20,
    seed: int = 0,
    cfg: float = 7.0,
    base_url: str = "http://127.0.0.1:8188",
    auto_checkpoint: bool = True,
) -> dict:
    """Minimal ComfyUI API workflow dict (checkpoint + CLIP + KSampler).

    Load via: POST http://127.0.0.1:8188/prompt  {"prompt": <this>}
    """
    if auto_checkpoint or not checkpoint:
        checkpoint = resolve_comfy_checkpoint(checkpoint, base_url=base_url)
    # SDXL wants larger latents; SD1.5 512 is fine
    is_xl = any(t in checkpoint.lower() for t in ("xl", "sdxl", "pony", "zimage"))
    if width is None:
        width = 1024 if is_xl else 512
    if height is None:
        height = 1024 if is_xl else 512

    positive = conditioning.get("positive_prompt", "")
    negative = conditioning.get("negative_prompt", "") or (
        "blurry, low quality, deformed, watermark"
    )
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "spatial_hoa", "images": ["8", 0]},
        },
    }


def submit_comfy_prompt(
    workflow: Mapping[str, Any],
    *,
    base_url: str = "http://127.0.0.1:8188",
    client_id: str = "spatial-hoa",
) -> dict:
    """POST workflow to ComfyUI. Returns API JSON or error dict with body."""
    # Accept either raw graph or already-wrapped {"prompt": ...}
    if "prompt" in workflow and isinstance(workflow.get("prompt"), dict):
        payload = dict(workflow)
        payload.setdefault("client_id", client_id)
    else:
        payload = {"prompt": dict(workflow), "client_id": client_id}

    url = base_url.rstrip("/") + "/prompt"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(err_body)
        except Exception:
            detail = {"raw": err_body[:2000]}
        ckpts = list_comfy_checkpoints(base_url)
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "detail": detail,
            "available_checkpoints": ckpts,
            "hint": (
                "Use --checkpoint <name> from available_checkpoints, "
                "or omit it to auto-select."
            ),
        }
    except urllib.error.URLError as e:
        return {"error": str(e), "hint": "Is ComfyUI running on :8188?"}
    except Exception as e:
        return {"error": str(e)}


def save_conditioning(cond: Mapping[str, Any], path: PathLike) -> None:
    Path(path).write_text(json.dumps(cond, indent=2) + "\n", encoding="utf-8")


def load_report(path: PathLike) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition_from_report_file(
    report_path: PathLike,
    *,
    base_prompt: str = "cinematic still, photoreal",
    style: str = "natural",
) -> dict:
    rep = load_report(report_path)
    return build_conditioning(rep, base_prompt=base_prompt, style=style)
