"""Minimal HTTP API so Qwythos / agents can request spatial reports.

Stdlib only (no Flask). Default bind: 127.0.0.1:8765

Endpoints
---------
GET  /health
GET  /v1/spatial/schema          — OpenAI-style tool descriptor
POST /v1/spatial/analyze        — JSON body → SpatialReport
POST /v1/spatial/analyze_file   — {path, mode, az, el, ...}
POST /v1/spatial/demo_scene     — synthetic multi-source report
POST /v1/spatial/vision         — Phase 3: boxes/rays → spatial report
POST /v1/spatial/fuse           — merge audio + vision report dicts

OpenAI tools (for agents with bash): also install ``spatial-report`` on PATH.
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .report import (
    REPORT_SCHEMA_VERSION,
    report_from_ambix_wav,
    report_from_hoa,
    report_from_mono_wav,
    report_from_scene,
)
from .stream import SourceSpec
from .synth import envelope_adsr, tone

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spatial_analyze",
        "description": (
            "Analyze spatial audio (Ambix HOA or mono plane-wave) or visual "
            "bounding boxes on the sphere. Returns a compact spatial report "
            "(DOA, bands, frames) from the HOA-7 calculator — not an LLM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["ambix_file", "mono_file", "demo_scene", "vision", "fuse"],
                    "description": "Analysis mode",
                },
                "path": {
                    "type": "string",
                    "description": "WAV path for ambix_file or mono_file",
                },
                "az": {
                    "type": "number",
                    "description": "Azimuth degrees for mono plane-wave encode (0=front, +90=left)",
                },
                "el": {
                    "type": "number",
                    "description": "Elevation degrees (0=horizon, +90=zenith)",
                },
                "order": {
                    "type": "integer",
                    "description": "Max HOA order 0..7 (default 3 for speed, 7 full)",
                    "default": 3,
                },
                "boxes": {
                    "type": "array",
                    "description": "Vision boxes: [{az, el, w_deg?, h_deg?, weight?, label?}]",
                    "items": {"type": "object"},
                },
                "audio_report": {
                    "type": "object",
                    "description": "Existing audio SpatialReport dict for fuse mode",
                },
                "vision_report": {
                    "type": "object",
                    "description": "Existing vision SpatialReport dict for fuse mode",
                },
            },
            "required": ["mode"],
        },
    },
}


def _json_response(handler: BaseHTTPRequestHandler, code: int, obj: Any) -> None:
    body = json.dumps(obj, indent=None).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    n = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(n) if n else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def handle_analyze(body: dict) -> dict:
    mode = body.get("mode", "demo_scene")
    order = int(body.get("order", 3))
    order = max(0, min(7, order))

    if mode == "demo_scene":
        sr = int(body.get("sample_rate", 48000))
        dur = float(body.get("duration", 0.4))
        n = int(sr * dur)
        env = envelope_adsr(n, sr)
        sources = [
            SourceSpec(0.0, 0.0, tone(440, dur, sr, amplitude=0.4) * env, "front"),
            SourceSpec(90.0, 10.0, tone(660, dur, sr, amplitude=0.25) * env, "left"),
        ]
        rep = report_from_scene(sources, sr, max_order=order)
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    if mode == "ambix_file":
        path = body.get("path") or body.get("file")
        if not path:
            raise ValueError("path required for ambix_file")
        rep = report_from_ambix_wav(path, max_order=order)
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    if mode == "mono_file":
        path = body.get("path") or body.get("file")
        if not path:
            raise ValueError("path required for mono_file")
        if body.get("az") is None:
            raise ValueError("az required for mono_file plane-wave encode")
        rep = report_from_mono_wav(
            path,
            float(body["az"]),
            float(body.get("el", 0.0)),
            max_order=order,
        )
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    if mode == "vision":
        from .vision import report_from_boxes

        boxes = body.get("boxes") or []
        rep = report_from_boxes(boxes, max_order=order)
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    if mode == "fuse":
        from .vision import fuse_reports

        ar = body.get("audio_report") or {}
        vr = body.get("vision_report") or {}
        return fuse_reports(ar, vr)

    if mode == "detect":
        from .detector import detect_to_sphere, load_boxes_json
        from .vision import report_from_boxes

        if body.get("boxes_path"):
            boxes = load_boxes_json(body["boxes_path"])
        elif body.get("image"):
            boxes = detect_to_sphere(
                body["image"],
                backend=body.get("backend", "auto"),
                score_thresh=float(body.get("score", 0.5)),
                hfov_deg=float(body.get("hfov", 90)),
                vfov_deg=float(body.get("vfov", 60)),
            )
        else:
            boxes = body.get("boxes") or []
        rep = report_from_boxes(boxes, max_order=order)
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        d["boxes"] = boxes
        return d

    if mode == "live":
        from .live_audio import live_report

        rep = live_report(
            duration_sec=float(body.get("duration", 2.0)),
            sample_rate=int(body.get("sample_rate", 48000)),
            channels=int(body.get("channels", 1)),
            source=body.get("source"),
            az_deg=float(body.get("az", 0.0)),
            el_deg=float(body.get("el", 0.0)),
            max_order=order,
            keep_wav=body.get("write_wav"),
        )
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    if mode == "panner":
        # UI spherical panner: az/el + W amplitude → spatial report (+ optional condition)
        from .conditioning import build_conditioning, panner_report

        rep = panner_report(
            float(body.get("az", body.get("az_deg", 0.0))),
            float(body.get("el", body.get("el_deg", 0.0))),
            float(body.get("w", body.get("w_amplitude", body.get("energy", 0.5)))),
        )
        if body.get("condition") or body.get("prompt"):
            return build_conditioning(
                rep,
                base_prompt=str(body.get("prompt", "")),
                style=str(body.get("style", "natural")),
            )
        return rep

    if mode == "condition":
        from .conditioning import build_conditioning, panner_report

        rep = body.get("report")
        if rep is None and (
            "az" in body or "az_deg" in body or "w" in body or "w_amplitude" in body
        ):
            # Convenience: condition directly from panner knobs
            rep = panner_report(
                float(body.get("az", body.get("az_deg", 0.0))),
                float(body.get("el", body.get("el_deg", 0.0))),
                float(body.get("w", body.get("w_amplitude", body.get("energy", 0.5)))),
            )
        else:
            rep = rep or body
        return build_conditioning(
            rep,
            base_prompt=str(body.get("prompt", "")),
            style=str(body.get("style", "natural")),
        )

    if mode == "hoa_vector":
        # raw coefficients list
        import numpy as np

        coeffs = body.get("hoa") or body.get("coefficients")
        if coeffs is None:
            raise ValueError("hoa coefficients required")
        sr = int(body.get("sample_rate", 48000))
        a = np.asarray(coeffs, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        rep = report_from_hoa(a, sr, max_order=order)
        d = rep.to_dict()
        d["one_liner"] = rep.one_liner()
        return d

    raise ValueError(f"unknown mode: {mode}")


class SpatialHandler(BaseHTTPRequestHandler):
    server_version = f"spatial-hoa/{__version__}"

    def log_message(self, fmt: str, *args) -> None:
        # quieter default
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "service": "spatial-hoa",
                    "version": __version__,
                    "schema": REPORT_SCHEMA_VERSION,
                },
            )
            return
        if path in ("/v1/spatial/schema", "/v1/tools"):
            _json_response(
                self,
                200,
                {
                    "tools": [TOOL_SCHEMA],
                    "report_schema": REPORT_SCHEMA_VERSION,
                },
            )
            return
        _json_response(self, 404, {"error": "not found", "path": path})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = _read_json(self)
        except Exception as e:
            _json_response(self, 400, {"error": f"invalid json: {e}"})
            return
        try:
            if path in (
                "/v1/spatial/analyze",
                "/v1/spatial/analyze_file",
                "/v1/spatial/demo_scene",
                "/v1/spatial/vision",
                "/v1/spatial/fuse",
            ):
                # map path to mode if not set
                if path.endswith("demo_scene") and "mode" not in body:
                    body["mode"] = "demo_scene"
                elif path.endswith("vision") and "mode" not in body:
                    body["mode"] = "vision"
                elif path.endswith("fuse") and "mode" not in body:
                    body["mode"] = "fuse"
                elif path.endswith("analyze_file") and "mode" not in body:
                    body["mode"] = "ambix_file" if body.get("ambix") else "mono_file"
                elif "mode" not in body:
                    body["mode"] = "demo_scene"
                result = handle_analyze(body)
                _json_response(self, 200, result)
                return
            _json_response(self, 404, {"error": "not found", "path": path})
        except Exception as e:
            _json_response(
                self,
                400,
                {"error": str(e), "trace": traceback.format_exc()[-800:]},
            )


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    httpd = ThreadingHTTPServer((host, port), SpatialHandler)
    print(f"spatial-hoa API http://{host}:{port}  (hoa64 {__version__})")
    print("  GET  /health")
    print("  GET  /v1/spatial/schema")
    print("  POST /v1/spatial/analyze")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="spatial-hoa HTTP API for Qwythos/agents")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
