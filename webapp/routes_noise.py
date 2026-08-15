"""Noise-lab API routes — DiT noise classifier training & analysis.

Wraps `noise_data` (NOISEX-92 access + log-mel DSP) and `dit_noise`
(DiT-backbone classifier, lazy torch):

* ``GET  /classes`` — 19-class list (15 NOISEX-92 + 4 RF synth) +
  trained-model status + ``live_sources`` (`rf_capture.live_sources`:
  per-source availability of mic / local wifi / ble radio capture).
* ``POST /train``   — JobManager job running `dit_noise.train_model`
  (Muon default, AdamW optional); per-epoch ``{epoch, loss, acc,
  val_acc}`` frames stream over the existing ``WS /ws/job/{job_id}``;
  `_BudgetStop` becomes the stop_flag.
* ``POST /analyze`` — classify a WAV file path OR a live capture
  (``live_seconds`` + ``live_source``): ``mic`` goes through
  `live_audio.capture_wav` (trusted-local only like the other
  path-taking endpoints), the RF sources (wifi/ble) go through
  `rf_capture.capture` — measured radio-counter cadence rendered to
  baseband, no temp WAV, capture ``stats`` echoed in the response;
  returns class probabilities + the log-mel spectrogram as a heatmap
  PNG.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import audio_io, dit_noise, noise_data, rf_capture
from ._png import heatmap_png
from .jobs import JOBS, Job, report
from .routes_hadamard import _jsafe
from .routes_search import _BudgetStop

router = APIRouter(prefix="/api/noise")

_MODEL_PATH = Path(dit_noise.__file__).with_name("dit_noise.pt")


@router.get("/classes")
def classes() -> dict:
    return {
        "classes": noise_data.NOISE_CLASSES,
        "synth_classes": sorted(noise_data.SYNTH_CLASSES),
        "recorded_classes": list(noise_data.RECORDED_CLASSES),
        "model": {
            "trained": _MODEL_PATH.exists(),
            "path": str(_MODEL_PATH) if _MODEL_PATH.exists() else None,
        },
        "live_sources": rf_capture.live_sources(),
    }


# ---------------------------------------------------------------- train

class TrainReq(BaseModel):
    epochs: int = 8
    batch_size: int = 64
    max_windows_per_class: int | None = None
    budget_s: float = 1800.0
    optimizer: str = "muon"  # muon | adamw


def _run_train(job: Job) -> dict:
    p = job.params
    stop = _BudgetStop(job)

    # the dataset load (first run downloads NOISEX-92) is silent until the
    # first epoch callback — report a status line immediately so the client
    # shows activity during it
    report(job, engine="dit_noise", status_text="loading dataset…")

    def cb(frame: dict) -> None:
        report(job, engine="dit_noise", **frame)

    return _jsafe(dit_noise.train_model(
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        max_windows_per_class=p["max_windows_per_class"],
        callback=cb,
        stop_flag=stop,
        out_path=_MODEL_PATH,
        optimizer=p["optimizer"],
    ))


@router.post("/train")
def train(req: TrainReq) -> dict:
    if not (1 <= req.epochs <= 200):
        raise HTTPException(status_code=400, detail="epochs must be 1..200")
    if not (8 <= req.batch_size <= 512):
        raise HTTPException(status_code=400, detail="batch_size must be 8..512")
    if req.max_windows_per_class is not None and not (16 <= req.max_windows_per_class <= 5000):
        raise HTTPException(status_code=400, detail="max_windows_per_class must be 16..5000")
    if not (10.0 <= req.budget_s <= 86400.0):
        raise HTTPException(status_code=400, detail="budget_s must be 10..86400 s")
    if req.optimizer not in ("muon", "adamw"):
        raise HTTPException(status_code=400, detail="optimizer must be muon or adamw")
    job = JOBS.submit("dit_noise_train", _run_train, {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "max_windows_per_class": req.max_windows_per_class,
        "budget_s": req.budget_s,
        "optimizer": req.optimizer,
        "live": {},
    })
    return {"job_id": job.id}


# ---------------------------------------------------------------- analyze

class AnalyzeReq(BaseModel):
    path: str | None = None          # existing WAV on this machine (trusted-local)
    live_seconds: float | None = None  # record live instead (see live_source)
    live_source: str = "mic"         # mic | wifi | ble — rf_capture.live_sources() keys


def _mel_png_b64(mel: np.ndarray) -> str:
    m = np.asarray(mel, dtype=np.float64)
    m = np.clip((m + 4.0) / 8.0, 0.0, 1.0)  # undo noise_data normalization range
    return base64.b64encode(heatmap_png(m, 512)).decode("ascii")


@router.post("/analyze")
def analyze(req: AnalyzeReq) -> dict:
    if (req.path is None) == (req.live_seconds is None):
        raise HTTPException(status_code=400, detail="give exactly one of path / live_seconds")
    if not _MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="no trained model — run /api/noise/train first")
    tmp = None
    cap_stats = None
    try:
        if req.live_seconds is not None:
            if not (0.5 <= req.live_seconds <= 30.0):
                raise HTTPException(status_code=400, detail="live_seconds must be 0.5..30")
            if req.live_source == "mic":
                from .. import live_audio  # lazy: needs a capture backend
                tmp = Path(tempfile.mkdtemp(prefix="hoa64_noise_")) / "capture.wav"
                try:
                    live_audio.capture_wav(tmp, duration_sec=req.live_seconds)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"capture failed: {e}")
                x, fs = audio_io.read_wav(tmp)
            else:
                srcs = rf_capture.live_sources()
                info = srcs.get(req.live_source)
                if info is None:
                    raise HTTPException(status_code=400, detail=(
                        f"unknown live_source {req.live_source!r}; "
                        f"expected one of {sorted(srcs)}"))
                if not info.get("available"):
                    raise HTTPException(status_code=400, detail=(
                        f"live_source {req.live_source!r} unavailable: "
                        f"{info.get('reason', 'not detected')}"))
                try:
                    x, fs, cap_stats = rf_capture.capture(req.live_source,
                                                          req.live_seconds)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"capture failed: {e}")
        else:
            try:
                x, fs = audio_io.read_wav(req.path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"cannot read {req.path!r}: {e}")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=-1 if x.shape[-1] <= 8 else 0)  # downmix
        x = x.astype(np.float32)
        if x.size < 2048:
            raise HTTPException(status_code=400, detail="audio too short")
        out = dit_noise.classify(x, fs)
        res = {
            "top": out["top"],
            "probs": out["probs"],
            "mel_png_b64": _mel_png_b64(out["mel"]),
            "fs": fs,
            "duration_s": float(x.size) / float(fs),
        }
        if cap_stats is not None:
            res["capture"] = cap_stats
        return _jsafe(res)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()
