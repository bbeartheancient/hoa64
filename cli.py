"""CLI: spatial-hoa analyze — produce JSON reports for other models.

Examples
--------
  python -m hoa64.cli analyze scene.wav --ambix
  python -m hoa64.cli analyze mono.wav --az 30 --el 0 -o report.json
  python -m hoa64.cli demo-scene -o /tmp/scene_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .audio_io import write_wav
from .report import (
    report_from_ambix_wav,
    report_from_mono_wav,
    report_from_scene,
)
from .stream import SourceSpec, encode_scene
from .synth import envelope_adsr, tone


def _cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    kwargs = dict(
        max_order=args.order,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        include_frames=not args.no_frames,
        include_bands=not args.no_bands,
    )
    if args.ambix:
        rep = report_from_ambix_wav(path, **kwargs)
    else:
        if args.az is None:
            print(
                "error: mono plane-wave encode needs --az (or pass --ambix)",
                file=sys.stderr,
            )
            return 2
        rep = report_from_mono_wav(
            path, args.az, args.el if args.el is not None else 0.0, **kwargs
        )

    text = rep.to_json(indent=2 if not args.compact else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(rep.one_liner())
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def _cmd_demo_scene(args: argparse.Namespace) -> int:
    sr = args.sr
    dur = args.duration
    n = int(sr * dur)
    env = envelope_adsr(n, sr)
    s1 = tone(440.0, dur, sr, amplitude=0.4) * env
    s2 = tone(660.0, dur, sr, amplitude=0.25) * env
    sources = [
        SourceSpec(0.0, 0.0, s1, label="front_A4"),
        SourceSpec(90.0, 15.0, s2, label="left_E5"),
    ]
    rep = report_from_scene(
        sources,
        sr,
        max_order=args.order,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
    )

    if args.write_wav:
        hoa = encode_scene(sources, max_order=min(args.order, 3))
        # Write order-1 B-format-ish (first 4 ch) for portability
        write_wav(args.write_wav, hoa[:4], sr)
        print(f"wrote ambix-lite WAV {args.write_wav} (4 ch)")

    text = rep.to_json(indent=2 if not args.compact else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(rep.one_liner())
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(args.host, args.port)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    import json
    from .detector import (
        detect_to_sphere,
        detections_to_sphere_boxes,
        load_boxes_json,
        load_yolo_labels,
        write_demo_image_with_box,
        Detection,
    )
    from .vision import report_from_boxes

    if args.demo_image:
        path = Path(args.demo_image)
        det = write_demo_image_with_box(path)
        boxes = detections_to_sphere_boxes([det], hfov_deg=args.hfov, vfov_deg=args.vfov)
        print(f"wrote demo image {path}")
    elif args.yolo_labels:
        dets = load_yolo_labels(args.yolo_labels)
        boxes = detections_to_sphere_boxes(dets, hfov_deg=args.hfov, vfov_deg=args.vfov)
    elif args.boxes_json:
        boxes = load_boxes_json(args.boxes_json)
    elif args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(
                f"error: image not found: {img_path}\n"
                "  /path/to/photo.jpg was only an example placeholder.\n"
                "  Try a real file, or:\n"
                "    spatial-report detect --demo-image /tmp/demo.png -o /tmp/det.json\n"
                "    spatial-report detect --image /tmp/spatial_hoa_e2e/demo_frame.png -o /tmp/det.json",
                file=sys.stderr,
            )
            return 2
        try:
            boxes = detect_to_sphere(
                img_path,
                backend=args.backend,
                score_thresh=args.score,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
            )
        except Exception as e:
            print(f"error: detector failed: {e}", file=sys.stderr)
            print(
                "  Fallback without neural net:\n"
                "    spatial-report detect --demo-image /tmp/demo.png -o /tmp/det.json",
                file=sys.stderr,
            )
            return 1
        if not boxes:
            print(
                "warning: no detections above score threshold "
                "(try --score 0.2, or pass --boxes-json / --demo-image)",
                file=sys.stderr,
            )
    else:
        print(
            "error: need --image, --boxes-json, --yolo-labels, or --demo-image\n"
            "  Example: spatial-report detect --demo-image /tmp/demo.png -o /tmp/det.json",
            file=sys.stderr,
        )
        return 2

    rep = report_from_boxes(boxes, max_order=args.order)
    text = rep.to_json(indent=2 if not args.compact else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(rep.one_liner())
        print(f"wrote {args.output}")
    else:
        print(text)
    if args.write_boxes:
        Path(args.write_boxes).write_text(json.dumps(boxes, indent=2) + "\n")
        print(f"wrote boxes {args.write_boxes}")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    from .live_audio import live_report, list_pulse_sources

    if args.list_sources:
        for s in list_pulse_sources():
            print(s)
        return 0
    rep = live_report(
        duration_sec=args.duration,
        sample_rate=args.sr,
        channels=args.channels,
        source=args.source,
        az_deg=args.az,
        el_deg=args.el,
        max_order=args.order,
        keep_wav=args.write_wav,
    )
    text = rep.to_json(indent=2 if not args.compact else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(rep.one_liner())
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def _dump_matrix(H: np.ndarray, fmt: str) -> None:
    for row in H:
        if fmt == "binary":
            print(" ".join("1" if x == 1 else "0" for x in row))
        else:
            print(" ".join("+1" if x == 1 else "-1" for x in row))


def _cmd_hadamard(args: argparse.Namespace) -> int:
    from . import hadamard as hd

    if args.selftest:
        return hd.selftest()

    if args.orders is not None:
        print("constructible orders <= %d:" % args.orders)
        print(hd.hadamard_orders(args.orders))
        return 0

    if args.generate is not None:
        H = hd.hadamard_known(args.generate)
        if H is None:
            print(
                f"order {args.generate}: no Sylvester/Paley/kron construction; "
                f"open cases 668, 716, 892 are beyond this toolset"
            )
            return 1
        c = hd.check(H, det=True)
        print(
            f"order {args.generate}: constructed & verified "
            f"(max_off={c['max_off']}, det_log10={c['det_log10']:.4f}, "
            f"bound_log10={c['det_bound_log10']:.4f})"
        )
        if args.out:
            hd.save_npy(args.out, H)
            print(f"wrote {args.out}")
        if args.dump:
            _dump_matrix(H, args.format)
        return 0

    if args.verify is not None:
        H = hd.load_npy(args.verify)
        if H is None:
            print(f"error: cannot read {args.verify}", file=sys.stderr)
            return 2
        c = hd.check(H, det=True)
        print(
            f"order {c['n']}: is_hadamard={c['is_hadamard']} "
            f"max_off={c['max_off']} f={c['f']} "
            f"h2_balanced={c['h2_all_balanced']} "
            f"det_log10={c.get('det_log10'):.4f} "
            f"bound_log10={c['det_bound_log10']:.4f}"
        )
        if args.dump:
            _dump_matrix(H, args.format)
        return 0

    if args.modular_check is not None:
        H = hd.load_npy(args.modular_check)
        if H is None:
            print(f"error: cannot read {args.modular_check}", file=sys.stderr)
            return 2
        r = hd.modular_check(H, args.mod)
        print(
            f"order {H.shape[0]}: H H^T = n I (mod {args.mod}): {r['ok']} "
            f"max_residue={r['max_residue']}"
        )
        return 0

    if args.search is not None:
        seeds = None
        if args.seeds:
            seeds = [s for p in args.seeds if (s := hd.load_npy(p)) is not None]
        best, st = hd.ils_search(
            args.search,
            seeds=seeds,
            inner_flips=args.inner,
            outer_iters=args.outer,
            time_budget=args.time,
            frac=args.frac,
            seed_int=args.seed,
            print_progress=not args.quiet,
        )
        print(
            f"order {args.search}: f={st['f']} max_off={st['max_off']} "
            f"iters={st['iters']} "
            f"det_log10={st['det_log10']:.4f} bound_log10={st['det_bound_log10']:.4f} "
            f"is_hadamard={st['is_hadamard']} elapsed={st['elapsed_s']:.1f}s"
        )
        if args.out:
            hd.save_npy(args.out, best)
            print(f"wrote best to {args.out}")
        if args.dump:
            _dump_matrix(best, args.format)
        return 0

    print(
        "need one of --selftest / --generate / --orders / --verify / "
        "--modular-check / --search",
        file=sys.stderr,
    )
    return 2


def _cmd_condition(args: argparse.Namespace) -> int:
    import json
    from .conditioning import (
        build_conditioning,
        comfy_txt2img_payload,
        condition_from_report_file,
        save_conditioning,
        submit_comfy_prompt,
        load_report,
    )

    rep = load_report(args.report)
    cond = build_conditioning(
        rep, base_prompt=args.prompt or "", style=args.style
    )
    if args.output:
        save_conditioning(cond, args.output)
        print(cond["positive_prompt"])
        print(f"wrote {args.output}")
    else:
        print(json.dumps(cond, indent=2))

    if args.comfy or args.write_workflow:
        from .conditioning import resolve_comfy_checkpoint, list_comfy_checkpoints

        ckpt = args.checkpoint
        if not ckpt or args.auto_checkpoint:
            ckpt = resolve_comfy_checkpoint(ckpt, base_url=args.comfy_url)
            print(f"comfy checkpoint: {ckpt}")
        wf = comfy_txt2img_payload(
            cond,
            checkpoint=ckpt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            base_url=args.comfy_url,
            auto_checkpoint=False,
        )
        if args.write_workflow:
            # Write API-ready payload so `curl -d @file` works
            api_body = {"prompt": wf, "client_id": "spatial-hoa"}
            Path(args.write_workflow).write_text(
                json.dumps(api_body, indent=2) + "\n"
            )
            print(f"wrote API workflow {args.write_workflow}")
        if args.comfy:
            result = submit_comfy_prompt(wf, base_url=args.comfy_url)
            print(json.dumps(result, indent=2))
            if result.get("error") and result.get("available_checkpoints"):
                print(
                    "available checkpoints:\n  "
                    + "\n  ".join(result["available_checkpoints"]),
                    file=sys.stderr,
                )
    return 0


def _cmd_vision(args: argparse.Namespace) -> int:
    import json
    from .vision import report_from_boxes

    if args.boxes_json:
        boxes = json.loads(Path(args.boxes_json).read_text(encoding="utf-8"))
    elif args.boxes:
        boxes = json.loads(args.boxes)
    else:
        # demo boxes
        boxes = [
            {"az": 0, "el": 0, "w_deg": 10, "h_deg": 10, "weight": 1.0, "label": "front"},
            {"az": 90, "el": 5, "w_deg": 12, "h_deg": 12, "weight": 0.7, "label": "left"},
        ]
    if isinstance(boxes, dict) and "boxes" in boxes:
        boxes = boxes["boxes"]
    rep = report_from_boxes(boxes, max_order=args.order)
    text = rep.to_json(indent=2 if not args.compact else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(rep.one_liner())
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hoa64",
        description="HOA-7 spatial calculator (audio + vision + JSON for Qwythos)",
    )
    p.add_argument("--version", action="version", version=f"hoa64 {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze a WAV → spatial JSON report")
    a.add_argument("input", help="Path to WAV")
    a.add_argument(
        "--ambix",
        action="store_true",
        help="Input is multi-channel Ambix ACN (not mono plane-wave)",
    )
    a.add_argument("--az", type=float, default=None, help="Plane-wave azimuth (deg)")
    a.add_argument("--el", type=float, default=0.0, help="Plane-wave elevation (deg)")
    a.add_argument("--order", type=int, default=7, help="Max HOA order (default 7)")
    a.add_argument("--frame-ms", type=float, default=40.0)
    a.add_argument("--hop-ms", type=float, default=20.0)
    a.add_argument("--no-frames", action="store_true")
    a.add_argument("--no-bands", action="store_true")
    a.add_argument("-o", "--output", help="Write JSON to path")
    a.add_argument("--compact", action="store_true", help="Minified JSON")
    a.set_defaults(func=_cmd_analyze)

    d = sub.add_parser("demo-scene", help="Synthetic 2-source scene → report")
    d.add_argument("--sr", type=int, default=48000)
    d.add_argument("--duration", type=float, default=0.5)
    d.add_argument("--order", type=int, default=7)
    d.add_argument("--frame-ms", type=float, default=40.0)
    d.add_argument("--hop-ms", type=float, default=20.0)
    d.add_argument("-o", "--output", help="Write JSON to path")
    d.add_argument("--write-wav", help="Also write 4-ch Ambix-lite WAV")
    d.add_argument("--compact", action="store_true")
    d.set_defaults(func=_cmd_demo_scene)

    v = sub.add_parser("vision", help="Vision boxes/rays → spatial JSON report")
    v.add_argument(
        "--boxes",
        help='JSON array of boxes, e.g. \'[{"az":0,"el":0,"weight":1}]\'',
    )
    v.add_argument("--boxes-json", help="Path to JSON file with boxes array")
    v.add_argument("--order", type=int, default=3)
    v.add_argument("-o", "--output", help="Write JSON to path")
    v.add_argument("--compact", action="store_true")
    v.set_defaults(func=_cmd_vision)

    s = sub.add_parser("serve", help="HTTP API for Qwythos/agents (default :8765)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_cmd_serve)

    det = sub.add_parser("detect", help="Image/YOLO/JSON → vision spatial report")
    det.add_argument("--image", help="Image path (optional torchvision detector)")
    det.add_argument("--boxes-json", help="Precomputed boxes or detections JSON")
    det.add_argument("--yolo-labels", help="YOLO .txt labels for an image")
    det.add_argument("--demo-image", help="Write synthetic image+box to this path and analyze")
    det.add_argument("--backend", default="auto", choices=["auto", "torchvision", "none"])
    det.add_argument("--score", type=float, default=0.5)
    det.add_argument("--hfov", type=float, default=90.0)
    det.add_argument("--vfov", type=float, default=60.0)
    det.add_argument("--order", type=int, default=3)
    det.add_argument("-o", "--output", help="Spatial report JSON")
    det.add_argument("--write-boxes", help="Write sphere boxes JSON")
    det.add_argument("--compact", action="store_true")
    det.set_defaults(func=_cmd_detect)

    live = sub.add_parser("live", help="Capture mic → spatial report")
    live.add_argument("--duration", type=float, default=2.0)
    live.add_argument("--sr", type=int, default=48000)
    live.add_argument("--channels", type=int, default=1)
    live.add_argument("--source", help="Pulse source name (pactl list short sources)")
    live.add_argument("--list-sources", action="store_true")
    live.add_argument("--az", type=float, default=0.0, help="Plane-wave az if mono")
    live.add_argument("--el", type=float, default=0.0)
    live.add_argument("--order", type=int, default=3)
    live.add_argument("-o", "--output")
    live.add_argument("--write-wav", help="Keep captured WAV")
    live.add_argument("--compact", action="store_true")
    live.set_defaults(func=_cmd_live)

    h = sub.add_parser(
        "hadamard",
        help="Hadamard generator / verifier / max-determinant search",
    )
    h.add_argument("--generate", type=int, metavar="ORDER")
    h.add_argument(
        "--orders", type=int, metavar="MAX", help="list constructible orders <= MAX"
    )
    h.add_argument("--verify", metavar="NPY", help="verify a saved +-1 matrix")
    h.add_argument(
        "--search", type=int, metavar="ORDER", help="heuristic max-det search"
    )
    h.add_argument("--seeds", action="append", default=None, metavar="NPY")
    h.add_argument("--inner", type=int, default=100000, metavar="FLIPS")
    h.add_argument("--outer", type=int, default=20)
    h.add_argument("--time", type=float, default=None, metavar="SEC")
    h.add_argument("--frac", type=float, default=0.05)
    h.add_argument("--seed", type=int, default=None, help="RNG seed")
    h.add_argument("--out", default=None, help="save result matrix .npy")
    h.add_argument("--modular-check", metavar="NPY", help="check H H^T = n I (mod m)")
    h.add_argument("--mod", type=int, default=64)
    h.add_argument("--dump", action="store_true", help="print the matrix rows")
    h.add_argument(
        "--format",
        default="sign",
        choices=["sign", "binary"],
        help="dump format: +-1 signs or 1/0 (default sign)",
    )
    h.add_argument("--quiet", action="store_true")
    h.add_argument("--selftest", action="store_true")
    h.set_defaults(func=_cmd_hadamard)

    cond = sub.add_parser("condition", help="Spatial report → diffusion conditioning")
    cond.add_argument("report", help="Path to spatial/fuse report JSON")
    cond.add_argument("--prompt", default="cinematic still, photoreal")
    cond.add_argument("--style", default="natural", choices=["natural", "tags", "technical"])
    cond.add_argument("-o", "--output", help="Write conditioning JSON")
    cond.add_argument("--comfy", action="store_true", help="Submit minimal workflow to ComfyUI")
    cond.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    cond.add_argument(
        "--checkpoint",
        default=None,
        help="ComfyUI ckpt_name (default: auto-detect from /object_info)",
    )
    cond.add_argument(
        "--auto-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve checkpoint against ComfyUI's installed list (default: true)",
    )
    cond.add_argument("--width", type=int, default=None, help="Latent width (default: 1024 XL / 512 SD)")
    cond.add_argument("--height", type=int, default=None)
    cond.add_argument("--steps", type=int, default=20)
    cond.add_argument("--seed", type=int, default=0)
    cond.add_argument(
        "--write-workflow",
        help="Save ComfyUI API JSON ({prompt: graph}) for curl -d @file",
    )
    cond.set_defaults(func=_cmd_condition)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
