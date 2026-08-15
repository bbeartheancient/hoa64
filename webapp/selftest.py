"""Webapp selftest — endpoint and JobManager smoke checks.

Runs in-process against `fastapi.testclient.TestClient(create_app())`;
no server needed.  Mirrors the `hadamard.selftest` reporting style
(expect/raise, PASS lines, single summary line).

    cd /home/bbear && python -m hoa64.webapp.selftest

Note: the selftest uses paley(108) rather than 92 — 92 is not a Paley
order (`paley(92)` returns None).  GCP builds powers of two, so the GCP
check uses order 4.
"""

from __future__ import annotations

import base64
import time

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def selftest() -> int:
    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    from fastapi.testclient import TestClient

    from ..hadamard import paley
    from .app import create_app
    from .jobs import JobManager

    client = TestClient(create_app())

    r = client.get("/api/health")
    expect(r.status_code == 200 and r.json()["status"] == "ok", "health endpoint")
    print("PASS health")

    r = client.post("/api/construct", json={"order": 64, "method": "sylvester"})
    d = r.json()
    expect(r.status_code == 200 and d["ok"], "construct sylvester(64)")
    expect(d["stats"]["is_hadamard"], "sylvester(64) not hadamard")
    expect(base64.b64decode(d["png_b64"])[:8] == PNG_MAGIC, "png_b64 bad magic")
    print("PASS construct sylvester(64)")

    r = client.post("/api/construct", json={"order": 108, "method": "paley"})
    d = r.json()
    expect(d["ok"] and d["stats"]["is_hadamard"], "construct paley(108)")
    print("PASS construct paley(108)")

    r = client.post("/api/construct", json={"order": 4, "method": "gcp", "seed": 1})
    d = r.json()
    expect(d["ok"] and d["stats"]["is_hadamard"], "construct gcp(4, seed=1)")
    print("PASS construct gcp(4)")

    r = client.post("/api/verify", json={"matrix": paley(12).tolist()})
    d = r.json()
    expect(d["ok"] and d["stats"]["is_hadamard"], "verify paley(12) matrix")
    print("PASS verify paley(12)")

    r = client.get("/api/orders", params={"max": 100})
    d = r.json()
    expect(r.status_code == 200 and 64 in d["known"], "orders list missing 64")
    print("PASS orders max=100")

    r = client.get("/")
    expect(r.status_code == 200 and "hoa64" in r.text, "index page")
    print("PASS index page")

    jm = JobManager()
    job = jm.submit("trivial", lambda j: 40 + 2, {})
    deadline = time.time() + 5.0
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.02)
    expect(job.status == "done" and job.result == 42, "JobManager trivial job")
    print("PASS JobManager")

    # ------------------------------------------------ Phase 2: search API
    from .jobs import JOBS

    def wait_terminal(job_id: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = client.get(f"/api/search/{job_id}").json()
            if d["status"] in ("done", "error", "cancelled"):
                return d
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} did not reach terminal state in {timeout}s")

    r = client.post("/api/search", json={"engine": "maxdet", "order": 8, "budget_s": 2})
    expect(r.status_code == 200 and "job_id" in r.json(), "POST /api/search maxdet")
    d = wait_terminal(r.json()["job_id"], 15.0)
    expect(d["status"] == "done", f"maxdet job status {d['status']}")
    expect(d["result"]["ok"] and d["result"]["stats"]["is_hadamard"], "maxdet result not hadamard")
    expect(base64.b64decode(d["result"]["png_b64"])[:8] == PNG_MAGIC, "maxdet png bad magic")
    print("PASS search maxdet(8)")
    done_job = r.json()["job_id"]

    r = client.post(
        "/api/search", json={"engine": "micromag", "order": 4, "mode": "sa", "budget_s": 2}
    )
    expect(r.status_code == 200, "POST /api/search micromag sa")
    jid = r.json()["job_id"]
    time.sleep(1.0)  # TestClient is in-process: the JOBS singleton is shared
    job = JOBS.get(jid)
    expect(job is not None, "micromag job missing from JOBS")
    progress = [m for m in job.history if m["type"] == "progress"]
    expect(len(progress) >= 1, "micromag sa emitted no progress frames")
    expect(
        all(k in progress[0] for k in ("step", "T", "E", "best_E", "accepts")),
        "micromag progress frame missing keys",
    )
    d = wait_terminal(jid, 15.0)
    expect(d["status"] == "done", f"micromag job status {d['status']}")
    print(f"PASS search micromag sa(4) — {len(progress)} progress frames")

    # ILS-mode engines must stream per-iteration frames ({"iter","f","best_f"})
    for eng, order in (("micromag", 4), ("tile", 4), ("williamson", 8),
                       ("gs", 8), ("circulant", 4)):
        r = client.post("/api/search", json={"engine": eng, "order": order, "budget_s": 3})
        expect(r.status_code == 200, f"POST /api/search {eng} ils")
        jid = r.json()["job_id"]
        d = wait_terminal(jid, 20.0)
        job = JOBS.get(jid)
        it_frames = [m for m in job.history
                     if m["type"] == "progress" and "iter" in m]
        expect(len(it_frames) >= 1, f"{eng} ils emitted no iteration frames")
        expect(all("best_f" in m for m in it_frames),
               f"{eng} iter frames missing best_f")
        print(f"PASS search {eng} ils({order}) — {len(it_frames)} iteration frames")

    r = client.post(
        "/api/search", json={"engine": "tile", "order": 128, "mode": "sa", "budget_s": 60}
    )
    expect(r.status_code == 200, "POST /api/search tile sa")
    jid = r.json()["job_id"]
    time.sleep(0.5)
    r = client.post(f"/api/search/{jid}/cancel")
    expect(r.status_code == 200 and r.json()["cancelled"], "cancel request failed")
    d = wait_terminal(jid, 10.0)
    expect(d["status"] in ("cancelled", "done"), f"tile cancel status {d['status']}")
    print(f"PASS search cancel (tile sa(128) → {d['status']})")

    with client.websocket_connect(f"/ws/job/{done_job}") as ws:
        snap = ws.receive_json()
        expect(snap["type"] == "snapshot" and snap["status"] == "done", "WS snapshot bad")
        expect(len(snap["history"]) >= 1, "WS snapshot history empty")
    print("PASS ws snapshot")

    r = client.get("/api/search")
    expect(r.status_code == 200 and any(j["id"] == done_job for j in r.json()["jobs"]),
           "search list missing job")
    print("PASS search list")

    # ------------------------------------------------ Phase 3: micromag sim
    import numpy as np

    from ..micromag import site_energy, total_energy

    rng = np.random.default_rng(3)
    Hr = rng.choice([-1, 1], size=(16, 16)).astype(np.int8)
    S = site_energy(Hr)
    _, _, E_dem, _ = total_energy(Hr)
    expect(abs(float(S.sum()) - E_dem) < 1e-6, "site_energy sum != demag energy")
    print("PASS site_energy sum == E_dem (16×16 random)")

    # sylvester start makes convergence deterministic (random 8 is ~20%)
    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 2, "field_every_steps": 500, "start": "sylvester"},
    )
    expect(r.status_code == 200 and "job_id" in r.json(), "POST /api/sim/micromag")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 15.0)
    expect(d["status"] == "done", f"sim job status {d['status']}")
    expect(d["result"]["ok"], "sim result not ok (sylvester start should converge)")
    job = JOBS.get(jid)
    field_frames = [m for m in job.history if "field_png_b64" in m]
    expect(len(field_frames) >= 1, "sim emitted no field_png_b64 frame")
    expect(
        base64.b64decode(field_frames[0]["field_png_b64"])[:8] == PNG_MAGIC,
        "field_png_b64 bad magic",
    )
    expect(
        base64.b64decode(field_frames[0]["grad_png_b64"])[:8] == PNG_MAGIC,
        "grad_png_b64 bad magic",
    )
    expect(
        any("E_dem" in m for m in job.history if m["type"] == "progress"),
        "sim progress frames missing energy decomposition",
    )
    flux_frames = [m for m in job.history if "flux_png_b64" in m]
    expect(len(flux_frames) >= 1, "sim emitted no flux_png_b64 frame")
    expect(
        base64.b64decode(flux_frames[0]["flux_png_b64"])[:8] == PNG_MAGIC,
        "flux_png_b64 bad magic",
    )
    print(f"PASS sim micromag(8, sylvester) — {len(field_frames)} field frames")

    # flux_map: wall density vs brute-force broken-bond count
    from ..hadamard import sylvester as _syl
    from ..micromag import flux_map

    H8 = _syl(8)
    W = flux_map(H8)
    bh = int(np.sum(H8 != np.roll(H8, -1, axis=1)))
    bv = int(np.sum(H8 != np.roll(H8, -1, axis=0)))
    expect(abs(float(W.sum()) - (bh + bv) / 2.0) < 1e-9, "flux_map wall count mismatch")
    expect(set(np.unique(W).tolist()) <= {0.0, 0.5, 1.0}, "flux_map values outside {0,½,1}")
    print(f"PASS flux_map sylvester(8) — {bh + bv} broken bonds, W sum {W.sum():.0f}")

    from ..micromag import flux_tiles as _flux_tiles
    from ..hadamard import paley as _paley, hadamard_product as _kron
    t8 = _flux_tiles(H8)
    expect(t8["n_tiles"] == 1 and t8["h8_agree"] == 1.0, "H8 is not its own tile")
    expect(abs(t8["mean_w"] - 0.5) < 1e-12, "H8 mean W != 1/2")
    for n in (16, 32, 64):
        tn = _flux_tiles(_syl(n))
        expect(tn["n_tiles"] == 4, f"sylvester({n}) tiles {tn['n_tiles']} != 4")
        expect(tn["kronecker_h8"] is True, f"sylvester({n}) not flagged kronecker_h8")
        expect(0.85 < tn["h8_agree"] < 0.95, f"sylvester({n}) H8 agree {tn['h8_agree']}")
        expect(abs(tn["mean_w"] - 0.5) < 1e-12, f"sylvester({n}) mean W != 1/2")
    P12 = _paley(12)
    expect(P12 is not None, "paley(12) missing")
    k = _flux_tiles(_kron(P12, H8))
    expect(k["n_tiles"] == 4 and k["kronecker_h8"], "P12⊗H8 should be 4 H8-tiles")
    ptiles = _flux_tiles(P12, tile=4)
    expect(ptiles["n_tiles"] > 4, "paley(12) unexpectedly 4-tile at 4×4")
    t256 = _flux_tiles(_syl(256))
    expect(t256["n_tiles"] == 4 and t256["nested"] is True,
           "H256 not nested 4-tile")
    expect(t256["scales"] == {"8": 4, "16": 4, "32": 4, "64": 4, "128": 4},
           f"H256 scales {t256['scales']}")
    expect(t256["counts"] == [341, 341, 171, 171],
           f"H256 counts {t256['counts']}")
    print("PASS flux_tiles H.8 tessellation (Sylvester 4-tile, Paley does not, "
          "P12⊗H8 does; H256 nested 4-at-every-scale, counts 341/171)")

    r = client.get("/api/sim/flux-tiles?order=16&start=sylvester")
    expect(r.status_code == 200, f"GET /api/sim/flux-tiles: {r.status_code}")
    ft = r.json()["flux_tiles"]
    expect(ft["n_tiles"] == 4 and ft["kronecker_h8"], f"GET flux-tiles 16: {ft}")
    expect(r.json().get("flux_png_b64"), "GET flux-tiles missing png")
    r = client.get("/js/tabs/micromag_sim.js")
    expect("sim-flux-read" in r.text and "renderFluxTiles" in r.text,
           "micromag_sim.js missing flux-tile panel")
    r = client.get("/js/kicad_layers.js")
    expect(r.status_code == 200 and "#e31c23" in r.text and "In2.Cu" in r.text,
           "kicad_layers.js palette")
    expect("shouldFill" in r.text and "paintRank" in r.text,
           "kicad_layers.js missing overlay paint order")
    for tab in ("antenna.js", "filter.js", "materials.js"):
        src = client.get(f"/js/tabs/{tab}").text
        expect("kicad_layers.js" in src, f"{tab} missing kicad_layers import")
    print("PASS GET /api/sim/flux-tiles + FLUX TILES panel")

    from ..micromag import _e_tile as _et
    expect(_et(H8, 1.0) == 0.0, "E_tile(H8) != 0")
    expect(_et(_syl(16), 1.0) < 0.2, "E_tile(H16) too large")
    print(f"PASS E_tile prior (H8=0, H16={_et(_syl(16), 1.0):.3f})")

    # ------------------------------------- Item 2: micromag goal attraction
    from ..hadamard import verify as _verify
    from ..micromag import micromag_sa

    goal4 = _syl(4)
    Hg, infog = micromag_sa(
        4, T_start=2.0, T_end=0.01, cooling=0.9995, max_steps=60000,
        goal=goal4, lam_goal=5.0, rng=np.random.default_rng(7),
    )
    expect(
        _verify(Hg) and infog["goal_agree"] == 1.0 and infog["E_goal"] == 0.0,
        "micromag_sa goal path did not anneal onto ±goal",
    )
    expect(
        np.array_equal(Hg, goal4) or np.array_equal(Hg, -goal4),
        "micromag_sa goal result is not ±goal",
    )
    _, info0 = micromag_sa(4, max_steps=2000, rng=np.random.default_rng(7))
    expect(
        "goal_agree" not in info0 and "E_goal" not in info0,
        "goal=None info dict gained goal keys",
    )
    try:
        micromag_sa(4, max_steps=10, goal=_syl(8))
        raise AssertionError("goal shape mismatch did not raise")
    except ValueError:
        pass
    print("PASS micromag_sa goal attraction (order 4)")

    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 2, "start": "sylvester", "goal_order": 12},
    )
    expect(r.status_code == 400, "non-multiple goal_order not rejected")
    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 2, "start": "library", "goal_order": 8},
    )
    expect(r.status_code == 400, "start=library + equal-order goal not rejected")

    # goal ABOVE the start order: library H(8) Kronecker-lifted to order 16,
    # annealed toward library H(16) — the lift is already Hadamard
    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 3, "start": "library", "goal_order": 16,
              "lam_goal": 1.0, "seed": 9},
    )
    expect(r.status_code == 200 and "job_id" in r.json(), "POST sim lifted-goal")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 20.0)
    expect(d["status"] == "done", f"lifted-goal sim job status {d['status']}")
    expect(
        d["result"]["ok"] and d["result"]["order"] == 16,
        f"lifted-goal result unexpected: {d['result'].get('order')}, ok={d['result'].get('ok')}",
    )
    print("PASS sim micromag(8→16, library start lifted to goal order)")

    # sylvester starts lift the same way — a non-power-of-2 goal (8×12=96,
    # quotient 12 in the library) must be accepted; a quotient that is not
    # a Hadamard order (8×3=24 → 3) must still be rejected
    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 2, "start": "sylvester", "goal_order": 96},
    )
    expect(r.status_code == 200, "sylvester start + non-pow2 goal rejected")
    client.post(f"/api/search/{r.json()['job_id']}/cancel")
    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 2, "start": "sylvester", "goal_order": 24},
    )
    expect(r.status_code == 400, "goal with non-Hadamard quotient not rejected")
    print("PASS sim lifted-goal validation (sylvester 8→96 ok, 8→24 rejected)")

    r = client.post(
        "/api/sim/micromag",
        json={"order": 8, "budget_s": 3, "start": "random", "goal_order": 8,
              "lam_goal": 5.0, "seed": 5},
    )
    expect(r.status_code == 200 and "job_id" in r.json(), "POST sim with goal_order")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 15.0)
    expect(d["status"] == "done", f"goal sim job status {d['status']}")
    job = JOBS.get(jid)
    gprog = [m for m in job.history if m["type"] == "progress" and "E" in m]
    expect(
        gprog and all("E_goal" in m for m in gprog),
        "goal sim progress frames missing E_goal",
    )
    expect(
        any("goal_agree" in m for m in gprog),
        "goal sim progress frames missing goal_agree",
    )
    print(f"PASS sim micromag(8, goal) — {len(gprog)} frames with E_goal")

    # ------------------------------------------------ Phase 4: HOA studio
    from ..analysis import angular_error_deg

    r = client.post("/api/hoa/speakers", json={"preset": "ring8", "order": 3})
    d = r.json()
    expect(r.status_code == 200 and len(d["positions"]) == 8, "speakers ring8 positions")
    expect(
        len(d["decode_matrix"]) == 8 and len(d["decode_matrix"][0]) == 16,
        "decode_matrix not 8×16",
    )
    expect(d["cond"] > 0 and d["n_channels"] == 16, "speakers cond/n_channels")
    print("PASS hoa speakers ring8 order 3")

    r = client.post(
        "/api/hoa/scene",
        json={
            "sources": [{"az": 30, "el": 10, "freq": 440}],
            "order": 3,
            "duration": 0.05,
            "wav": True,
        },
    )
    d = r.json()
    expect(r.status_code == 200, "scene POST")
    pm = d["power_map"]
    expect(len(pm["power"]) == 36 and len(pm["power"][0]) == 72, "power_map not 36×72")
    err = angular_error_deg(d["doa"]["az"], d["doa"]["el"], 30.0, 10.0)
    expect(err <= 15.0, f"doa off by {err:.1f}°")
    expect(base64.b64decode(d["power_png_b64"])[:8] == PNG_MAGIC, "power png bad magic")
    expect("wav_token" in d, "scene wav_token missing")
    r2 = client.get(f"/api/hoa/wav/{d['wav_token']}")
    expect(r2.status_code == 200 and r2.content[:4] == b"RIFF", "wav download bad")
    r3 = client.get(f"/api/hoa/wav/{d['wav_token']}")
    expect(r3.status_code == 404, "wav token should be one-shot")
    print(f"PASS hoa scene (doa err {err:.2f}°, wav one-shot OK)")

    r = client.post(
        "/api/hoa/decode-grid", json={"hoa": [1.0] + [0.0] * 15, "n_azi": 18, "n_el": 9}
    )
    d = r.json()
    expect(
        r.status_code == 200 and len(d["samples"]) == 18 and len(d["samples"][0]) == 9,
        "decode-grid samples not 18×9",
    )
    expect(base64.b64decode(d["png_b64"])[:8] == PNG_MAGIC, "decode-grid png bad magic")
    print("PASS hoa decode-grid 18×9")

    from pathlib import Path as _P

    vendor = _P(__file__).parent / "static" / "vendor"
    three = (vendor / "three.module.js").read_text()[:200]
    expect("three" in three.lower() and len(three) > 100, "three.module.js header")
    expect((vendor / "OrbitControls.js").stat().st_size > 10000, "OrbitControls.js size")
    print("PASS vendor three.js files")

    # ------------------------------------------------ Phase 5a: generative lab
    from .. import orbitals as _orb

    expect(_orb.selftest() == 0, "orbitals selftest")  # prints its own PASS lines

    r = client.post("/api/gen/terrain", json={"size": 64, "order": 16, "octaves": 3, "seed": 1})
    d = r.json()
    expect(r.status_code == 200, "POST /api/gen/terrain")
    expect(len(d["heightmap"]) == 64 and len(d["heightmap"][0]) == 64, "heightmap not 64²")
    st = d["stats"]
    expect(0.0 <= st["min"] <= st["mean"] <= st["max"] <= 1.0, "terrain stats out of bounds")
    expect(len(d["layers"]) == 3, "terrain layers != octaves")
    expect(base64.b64decode(d["png_b64"])[:8] == PNG_MAGIC, "terrain png bad magic")
    for lb in d["layers"]:
        expect(base64.b64decode(lb)[:8] == PNG_MAGIC, "terrain layer png bad magic")
    # layers_f32: signed amp-weighted octave contribs on the heightmap grid
    expect(len(d["layers_f32"]) == 3, "layers_f32 != octaves")
    for lf in d["layers_f32"]:
        expect(len(lf) == 64 and len(lf[0]) == 64, "layers_f32 grid != heightmap grid")
    print("PASS gen terrain (64², 3 layers + layers_f32)")

    r = client.post("/api/gen/terrain", json={"size": 32, "order": 16, "octaves": 9})
    expect(r.status_code == 400, "octaves > 8 should be 400")

    r = client.post("/api/gen/orbital", json={"n": 2, "l": 1, "m": 0, "samples": 2000, "seed": 1})
    d = r.json()
    expect(r.status_code == 200, "POST /api/gen/orbital")
    expect(len(d["points"]) == 2000 and len(d["points"][0]) == 3, "orbital points not N×3")
    expect(len(d["weights"]) == 2000, "orbital weights length")
    expect(d["extent"] > 0.0, "orbital extent <= 0")
    expect(base64.b64decode(d["proj_png_b64"])[:8] == PNG_MAGIC, "proj png bad magic")
    print("PASS gen orbital (2p, 2000 pts)")

    r = client.post("/api/gen/noise-field", json={"size": 64, "order": 16, "seed": 1})
    d = r.json()
    expect(r.status_code == 200, "POST /api/gen/noise-field")
    expect(len(d["grid"]) == 64 and len(d["grid"][0]) == 64, "noise grid not 64²")
    expect(base64.b64decode(d["png_b64"])[:8] == PNG_MAGIC, "noise png bad magic")
    print("PASS gen noise-field (64²)")

    # ------------------------------------------------ Phase 5b: library / DAG
    from ..game_of_hadamard import classify_orders

    cd = classify_orders(128)
    expect(64 in cd["built"], "classify_orders(128): 64 not built")
    expect(64 not in cd["gaps"], "classify_orders(128): 64 in gaps")
    expect(64 in cd["labels"] and 64 in cd["depths"], "classify_orders labels/depths")
    print(f"PASS classify_orders(128) — {len(cd['built'])} built, {len(cd['gaps'])} gaps")

    # 1212 = 4·303 is the first true gap (1211 = 7·173 kills Paley I)
    r = client.get("/api/dag", params={"max": 1300})
    d = r.json()
    expect(r.status_code == 200 and 64 in d["built"], "GET /api/dag")
    expect(len(d["gaps"]) > 0 and not (set(d["gaps"]) & set(d["built"])), "dag gaps bad")
    expect(1212 in d["gaps"], "dag: 1212 should be a gap")
    print(f"PASS dag max=1300 — {len(d['built'])} built, gaps {d['gaps'][:3]}")

    r = client.get("/api/detbounds", params={"max": 64})
    d = r.json()
    expect(r.status_code == 200 and len(d["entries"]) >= 8, "GET /api/detbounds")
    by_order = {e["order"]: e for e in d["entries"]}
    for n in (1, 2, 4, 8, 16, 32, 64):  # Sylvester attains the Hadamard bound
        e = by_order[n]
        expect(
            abs(e["det_log10"] - e["det_bound_log10"]) < 1e-6,
            f"sylvester({n}) det {e['det_log10']} != bound {e['det_bound_log10']}",
        )
    for e in d["entries"]:
        if e["det_log10"] is not None:
            expect(
                e["det_log10"] <= e["det_bound_log10"] + 1e-6,
                f"order {e['order']} det over the Hadamard bound",
            )
    print(f"PASS detbounds max=64 — {len(d['entries'])} entries, sylvester at bound")

    r = client.get("/api/challenges")
    expect(r.status_code == 200, "GET /api/challenges")
    ch = r.json()
    expect(len(ch["challenges"]) == 23 and len(ch["alignment"]) == 23,
           "challenges payload must be 23+23")
    expect(ch["summary"]["total"] == 23 and ch["summary"]["counts"]["active"] == 1,
           "challenges summary")
    expect(any(c["n"] == 19 and "Riemann" in c["title"] for c in ch["challenges"]),
           "challenge 19 (Riemann) missing")
    expect(any(a["n"] == 19 and a["status"] == "active" for a in ch["alignment"]),
           "challenge 19 must be the active rh.py anchor")
    expect(any(e["module"] == "muon.py"
               for a in ch["alignment"] for e in a.get("engines") or []),
           "muon.py must appear in the challenges alignment")
    r = client.get("/js/tabs/library.js")
    expect(r.status_code == 200 and "DARPA" in r.text and "challenges" in r.text,
           "library.js missing CHALLENGES layer")
    expect("lib-layer-select" in r.text, "library.js missing layer select")
    print("PASS DARPA-23 challenges frame (API + Library CHALLENGES layer)")

    # ------------------------------------------------ ℍ³ hadamard space
    from .. import hadamard_space as _hs

    expect(_hs.selftest() == 0, "hadamard_space selftest")  # own PASS lines

    r = client.post("/api/viz/hadamard-space", json={"order": 16, "mode": "rows", "kappa": 1})
    d = r.json()
    expect(r.status_code == 200, "POST /api/viz/hadamard-space rows")
    expect(len(d["points"]) == 16 and len(d["points"][0]) == 3, "space points not 16×3")
    expect(
        all(sum(x * x for x in p) < 1.0 for p in d["points"]),
        "poincaré point outside the ball",
    )
    expect(len(d["geodesics"]) >= 1, "no geodesics returned")
    expect(
        all(len(g) >= 8 and len(g[0]) == 3 for g in d["geodesics"]),
        "geodesic polylines malformed",
    )
    print(f"PASS viz hadamard-space rows (16 pts, {len(d['geodesics'])} geodesics)")

    r = client.post("/api/viz/hadamard-space", json={"order": 16, "mode": "lattice", "kappa": 1})
    d = r.json()
    expect(r.status_code == 200, "POST /api/viz/hadamard-space lattice")
    expect(
        len(d["verts"]) == 16 and len(d["verts"][0]) == 16 and len(d["verts"][0][0]) == 3,
        "lattice verts not 16×16×3",
    )
    expect(set(sum(d["colors"], [])) <= {0, 1}, "lattice colors not ±1-derived")
    print("PASS viz hadamard-space lattice (16² grid)")

    r = client.post("/api/viz/hadamard-space", json={"order": 3})
    expect(r.status_code == 400, "order 3 should be 400")
    print("PASS viz hadamard-space rejects order 3")

    # ------------------------------------------------ Antenna lab
    r = client.post("/api/antenna/design", json={
        "f_lo_mhz": 2400, "f_hi_mhz": 2485, "medium": "air",
        "site": {"mounting": "pcb", "max_size_m": 0.04},
    })
    d = r.json()
    expect(r.status_code == 200, "POST /api/antenna/design")
    entries = d["entries"]
    expect(len(entries) >= 1, "design entries empty")
    scores = [e["score"] for e in entries]
    expect(all(0.0 < s <= 1.0 for s in scores), "design scores outside (0,1]")
    expect(
        all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
        "design entries not sorted by score desc",
    )
    top = entries[0]
    expect(top["design"]["dimensions_m"], "top design missing dimensions_m")
    expect(isinstance(top["explain"], str) and top["explain"], "top explain missing")
    expect(isinstance(top["reasons"], list) and top["reasons"], "top reasons missing")
    expect(abs(d["required_bw_frac"] - 0.0348) < 0.001, "required_bw_frac off")
    expect(d["f_center_mhz"] == 2442.5, "f_center_mhz != 2442.5")
    # physics pin: a 2.4 GHz half-wave dipole in air is ≈ 58 mm long
    dip = next(e for e in entries if e["type"] == "dipole")
    dip_dims = [float(v) for v in dip["design"]["dimensions_m"].values()]
    expect(
        any(0.0561 <= v <= 0.0601 for v in dip_dims),
        f"dipole has no ≈58 mm dimension: {dip_dims}",
    )
    print(f"PASS antenna design ({len(entries)} entries, top={top['type']}, "
          f"dipole dims pinned)")

    r = client.post("/api/antenna/design", json={"f_lo_mhz": 2485, "f_hi_mhz": 2400})
    expect(r.status_code == 400, "design f_lo >= f_hi should be 400")
    print("PASS antenna design rejects f_lo >= f_hi")

    r = client.post("/api/antenna/parts", json={
        "f_lo_mhz": 2400, "f_hi_mhz": 2485, "gain_dbi_min": 0})
    d = r.json()
    expect(r.status_code == 200, "POST /api/antenna/parts")
    matches = d["matches"]
    expect(len(matches) >= 4, "parts matches < 4")
    expect(
        all(matches[i]["score"] >= matches[i + 1]["score"]
            for i in range(len(matches) - 1)),
        "parts matches not sorted by score desc",
    )
    expect(
        all(all(k in m for k in ("part", "mfr", "erf_url")) for m in matches),
        "parts match missing part/mfr/erf_url",
    )
    expect(
        all(m["freq_lo_mhz"] <= 2400 and m["freq_hi_mhz"] >= 2485 for m in matches),
        "parts match does not cover the band",
    )
    print(f"PASS antenna parts ({len(matches)} matches, all cover 2400–2485)")

    r = client.post("/api/antenna/parts", json={"f_lo_mhz": 2400, "f_hi_mhz": 5800})
    d = r.json()
    expect(r.status_code == 200 and len(d["matches"]) >= 4,
           "wide 2400–5800 should fall back to overlapping parts")
    expect(d.get("coverage") == "overlap",
           "2400–5800 coverage mode should be overlap")
    print(f"PASS antenna parts wide-band fallback ({len(d['matches'])} overlaps)")

    r = client.post("/api/antenna/parts", json={"f_lo_mhz": 38000, "f_hi_mhz": 39000})
    d = r.json()
    expect(r.status_code == 200 and d["matches"] == [], "38–39 GHz should match nothing")
    print("PASS antenna parts empty band (38–39 GHz)")

    r = client.post("/api/antenna/kicad", json={"design_type": "patch", "f_mhz": 2450})
    d = r.json()
    expect(r.status_code == 200 and d.get("token"), "POST /api/antenna/kicad")
    files = d["files"]
    expect(len(files) >= 2, "kicad export < 2 files")
    mod = [f for f in files if f.endswith(".kicad_mod")]
    pcb = [f for f in files if f.endswith(".kicad_pcb")]
    expect(mod and pcb, "kicad export missing .kicad_mod/.kicad_pcb")
    for name, magic in ((mod[0], "(footprint"), (pcb[0], "(kicad_pcb")):
        rg = client.get(f"/api/antenna/kicad/{d['token']}/{name}")
        expect(rg.status_code == 200 and rg.text.startswith(magic),
               f"kicad file {name} bad")
        rg2 = client.get(f"/api/antenna/kicad/{d['token']}/{name}")
        expect(rg2.status_code == 404, f"kicad file {name} not one-shot")
    print(f"PASS antenna kicad patch ({len(files)} files, one-shot OK)")

    r = client.post("/api/antenna/kicad", json={"design_type": "bogus", "f_mhz": 2450})
    expect(r.status_code == 400, "kicad bogus design_type should be 400")
    print("PASS antenna kicad rejects bogus design_type")

    r = client.post("/api/antenna/kicad", json={"design_type": "mifa", "f_mhz": 2450})
    d = r.json()
    expect(r.status_code == 200 and d.get("preview") and d.get("params"),
           "POST /api/antenna/kicad mifa missing preview/params")
    expect(d["params"]["mifa_trace_mm"] == 0.508, "mifa_trace_mm != 20 mil")
    expect(d["params"]["return_loss_target_db"] == 10.0, "RL target != 10 dB")
    expect(len(d["preview"]["prims"]) >= 4, "mifa preview has too few prims")
    expect(any(f.endswith(".kicad_pcb") for f in d["files"]), "mifa missing board")
    expect(d.get("preview_board") and d["preview_board"]["prims"],
           "mifa missing preview_board")
    print(f"PASS antenna kicad MIFA ({len(d['preview']['prims'])} prims, "
          f"{len(d['files'])} files)")

    r = client.get("/api/antenna/kicad/library")
    expect(r.status_code == 200, "GET /api/antenna/kicad/library")
    lib = r.json().get("footprints") or []
    print(f"PASS antenna kicad library ({len(lib)} RF_Antenna footprints)")
    if lib:
        name = next((e["name"] for e in lib if "SWRA117D" in e["name"]), lib[0]["name"])
        r = client.post("/api/antenna/kicad", json={
            "design_type": "lib", "f_mhz": 2450, "opts": {"lib_name": name}})
        d = r.json()
        expect(r.status_code == 200 and d.get("preview"), f"lib {name} export failed")
        print(f"PASS antenna kicad library scale ({name})")

    walk = [[0, 0, 0], [0.005, 0, 0], [0.005, 0.005, 0], [0.01, 0.005, 0]]
    r = client.post("/api/antenna/kicad", json={
        "design_type": "evolved", "f_mhz": 2450, "opts": {"points": walk}})
    d = r.json()
    expect(r.status_code == 200 and d.get("preview") and d.get("preview_board"),
           "evolved kicad missing preview")
    print("PASS antenna kicad evolved walk")

    # fresh export kept unconsumed for the frontend-path pinning below
    rp = client.post("/api/antenna/kicad", json={"design_type": "patch", "f_mhz": 2450})
    kicad_pin_token = rp.json()["token"]
    kicad_pin_name = rp.json()["files"][0]

    r = client.post("/api/antenna/fields", json={
        "f_mhz": 150, "medium": "water", "interface": False,
        "n": 24, "max_steps": 30, "frame_every": 10, "budget_s": 120,
    })
    expect(r.status_code == 200 and "job_id" in r.json(), "POST /api/antenna/fields")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 60.0)
    expect(d["status"] == "done", f"fields job status {d['status']}")
    job = JOBS.get(jid)
    fframes = [m for m in job.history
               if all(k in m for k in ("e_xy_png_b64", "e_xz_png_b64", "ar_png_b64"))]
    expect(len(fframes) >= 1, "fields emitted no e_xy/e_xz/ar frame")
    for key in ("e_xy_png_b64", "e_xz_png_b64", "ar_png_b64"):
        expect(base64.b64decode(fframes[0][key])[:8] == PNG_MAGIC,
               f"fields {key} bad magic")
    res = d["result"]
    expect(res["alpha_theory"] > 0.0, "fields alpha_theory <= 0 (water)")
    expect(res["dx_m"] > 0.0, "fields dx_m <= 0")
    print(f"PASS antenna fields (water 150 MHz, {len(fframes)} heatmap frames, "
          f"α={res['alpha_theory']:.3g} Np/m)")

    r = client.post("/api/antenna/evolve", json={
        "f_mhz": 2450, "medium": "air", "hadamard_order": 64,
        "max_steps": 150, "budget_s": 120,
    })
    expect(r.status_code == 200 and "job_id" in r.json(), "POST /api/antenna/evolve")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 60.0)
    expect(d["status"] == "done", f"evolve job status {d['status']}")
    job = JOBS.get(jid)
    pframes = [m for m in job.history if "points" in m]
    expect(len(pframes) >= 1, "evolve emitted no points frame")
    expect(
        any(len(m["points"]) >= 1 and len(m["points"][0]) == 3 for m in pframes),
        "evolve points not [x,y,z] triples",
    )
    expect(
        any("E" in m and "best_E" in m for m in pframes),
        "evolve frames missing E/best_E",
    )
    pat = [m for m in job.history if "pattern_png_b64" in m]
    expect(len(pat) >= 1, "evolve emitted no pattern_png_b64 frame")
    expect(base64.b64decode(pat[-1]["pattern_png_b64"])[:8] == PNG_MAGIC,
           "evolve pattern png bad magic")
    res = d["result"]
    expect(np.isfinite(res["gain_dbi"]), "evolve gain_dbi not finite")
    expect("re" in res["z_in"] and "im" in res["z_in"], "evolve z_in missing re/im")
    print(f"PASS antenna evolve (H64 seed, gain {res['gain_dbi']:.2f} dBi)")

    r = client.post("/api/antenna/evolve", json={
        "f_mhz": 2450, "medium": "air", "topology": "pcb",
        "hadamard_order": 32, "max_steps": 40, "budget_s": 60,
    })
    expect(r.status_code == 200 and "job_id" in r.json(), "POST evolve topology=pcb")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 60.0)
    expect(d["status"] == "done", f"pcb evolve status {d['status']}")
    res = d["result"]
    expect(res.get("kind") == "pcb", f"pcb evolve kind={res.get('kind')}")
    expect("E_dfm" in (res.get("terms") or {}), "pcb evolve missing E_dfm")
    print(f"PASS antenna evolve pcb (kind=pcb, S11 {res['s11_db']:.1f} dB)")

    r = client.get("/js/tabs/antenna.js")
    expect(r.status_code == 200 and "ant-layer-select" in r.text, "antenna.js tab")
    expect("smith-canvas" in r.text and "drawSmith" in r.text, "antenna.js smith panel")
    expect("kicad-canvas" in r.text and "drawKicadPreview" in r.text,
           "antenna.js kicad preview")
    expect("ant-e-topo" in r.text and "ant-k-lib" in r.text, "antenna.js pcb/lib controls")
    r = client.get("/")
    expect('data-tab="antenna"' in r.text, "index antenna tab button")
    print("PASS antenna static integrity (tab js + index button)")

    r = client.post("/api/antenna/smith", json={
        "f_lo_mhz": 950.0, "f_hi_mhz": 1050.0, "n_points": 5, "source": "dipole"})
    expect(r.status_code == 200, f"smith dipole sweep: {r.status_code} {r.text[:200]}")
    sw = r.json()
    expect(len(sw["sweep"]) == 5 and sw["z0"] == 50.0, "smith sweep shape")
    zmid = sw["sweep"][2]["z"]
    expect(40.0 < zmid["re"] < 120.0, f"smith dipole Re(Z) off: {zmid}")
    for p in sw["sweep"]:
        if "gamma" in p:
            expect(abs(p["gamma"]["re"]) <= 1.0 + 1e-6
                   and abs(p["gamma"]["im"]) <= 1.0 + 1e-6, "gamma off chart")
    pts = [[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0.02, 0.01, 0], [0.03, 0, 0]]
    r = client.post("/api/antenna/smith", json={
        "f_lo_mhz": 2400.0, "f_hi_mhz": 2500.0, "n_points": 3,
        "source": "wire", "points": pts})
    expect(r.status_code == 200 and all("gamma" in p for p in r.json()["sweep"]),
           "smith wire sweep")
    r = client.post("/api/antenna/smith", json={
        "f_lo_mhz": 900.0, "f_hi_mhz": 1000.0, "source": "bogus"})
    expect(r.status_code == 400, "smith rejects bogus source")
    print(f"PASS antenna smith (dipole Z_in mid = {zmid['re']:.1f}{zmid['im']:+.1f}j Ω)")

    # survey: validation paths only (tile fetch needs network)
    r = client.post("/api/antenna/survey", json={
        "tx": {"lat": 46.6, "lon": 8.0, "h_m": 9999},
        "rx": {"lat": 46.62, "lon": 8.02, "h_m": 15}, "f_mhz": 2450})
    expect(r.status_code == 400, "survey rejects h_m out of range")
    r = client.post("/api/antenna/survey", json={
        "tx": {"lat": 146.6, "lon": 8.0, "h_m": 15},
        "rx": {"lat": 46.62, "lon": 8.02, "h_m": 15}, "f_mhz": 2450})
    expect(r.status_code == 400, "survey rejects lat out of range")
    print("PASS antenna survey validation (400s, offline)")

    r = client.get("/api/noise/classes")
    expect(r.status_code == 200, "noise classes endpoint")
    nd = r.json()
    expect(len(nd["classes"]) == 19 and isinstance(nd["model"]["trained"], bool),
           "noise classes payload")
    expect(nd["classes"][:15] == [
        "white", "pink", "babble", "factory1", "factory2",
        "buccaneer1", "buccaneer2", "f16", "destroyerengine", "destroyerops",
        "leopard", "m109", "machinegun", "volvo", "hfchannel"],
           "recorded class order must stay pinned")
    expect(nd["classes"][15:] == ["ble", "wifi", "zigbee", "lora"],
           "RF synth classes must be appended")
    expect(set(nd["synth_classes"]) == {"ble", "wifi", "zigbee", "lora"},
           "synth_classes payload")
    expect(len(nd["recorded_classes"]) == 15, "recorded_classes payload")
    r = client.post("/api/noise/analyze", json={})
    expect(r.status_code == 400, "noise analyze needs path xor live_seconds")
    r = client.post("/api/noise/analyze", json={"path": "/nonexistent.wav"})
    expect(r.status_code == 400, "noise analyze bad input → 400")
    r = client.post("/api/noise/train", json={"epochs": 1, "optimizer": "sgd"})
    expect(r.status_code == 400, "noise train rejects unknown optimizer")
    r = client.get("/js/tabs/noise.js")
    expect(r.status_code == 200 and "NOISE LAB" in r.text, "noise.js tab")
    expect("noi-t-opt" in r.text and "MUON" in r.text, "noise.js muon selector")
    expect("noi-class-list" in r.text, "noise.js class list")
    r = client.get("/")
    expect('data-tab="noise"' in r.text, "index noise tab button")
    print(f"PASS noise lab (19 classes, 4 RF synth, model trained: {nd['model']['trained']})")

    r = client.post("/api/mcu/firmware", json={"board": "esp32", "w": 8, "h": 8})
    expect(r.status_code == 200, "mcu firmware generate")
    md = r.json()
    expect(md["token"] and any(f.endswith(".ino") for f in md["files"]),
           "mcu firmware payload")
    ino = next(f for f in md["files"] if f.endswith(".ino"))
    rf = client.get(f"/api/mcu/file/{md['token']}/{ino}")
    expect(rf.status_code == 200 and b"FastLED" in rf.content, "mcu firmware download")
    r = client.post("/api/mcu/firmware", json={"board": "pic32"})
    expect(r.status_code == 400, "mcu firmware rejects unknown board")
    r = client.post("/api/mcu/firmware", json={"kind": "bogus"})
    expect(r.status_code == 400, "mcu firmware rejects unknown kind")
    r = client.post("/api/mcu/export",
                    json={"engine": "hadamard_core", "target": "rust_no_std"})
    me = r.json()
    expect(r.status_code == 200 and "lib.rs" in me["files"], "mcu edge export")
    r = client.post("/api/mcu/export", json={"engine": "bogus", "target": "c_baremetal"})
    expect(r.status_code == 400, "mcu export rejects unknown engine")
    r = client.get("/js/tabs/microcontroller.js")
    expect(r.status_code == 200 and "Microcontroller Lab" in r.text,
           "microcontroller.js tab")
    r = client.get("/")
    expect('data-tab="microcontroller"' in r.text, "index mcu tab button")
    print("PASS mcu lab (firmware + edge export + tab)")


    r = client.post("/api/filter/design", json={
        "kind": "lpf", "proto": "butterworth", "n": 5, "f_c_mhz": 1000})
    expect(r.status_code == 200, f"filter design: {r.status_code} {r.text[:200]}")
    fd = r.json()
    expect(fd["design"]["kind"] == "lpf" and fd["sweep"]["s21_db"], "filter design payload")
    expect(fd["metrics"]["il_db"] < 3.0, f"LPF passband IL {fd['metrics']['il_db']}")
    expect(fd["preview"]["prims"], "filter design missing preview")
    print(f"PASS filter design LPF n=5 (IL {fd['metrics']['il_db']:.2f} dB, "
          f"{len(fd['preview']['prims'])} prims)")

    r = client.post("/api/filter/design", json={
        "kind": "bpf", "n": 3, "f_lo_mhz": 2300, "f_hi_mhz": 2600})
    expect(r.status_code == 200 and r.json()["design"]["kind"] == "bpf", "filter bpf")
    r = client.post("/api/filter/kicad", json={"kind": "lpf", "n": 5, "f_c_mhz": 2450})
    kd = r.json()
    expect(r.status_code == 200 and kd.get("token") and kd.get("files"), "filter kicad")
    expect(any(f.endswith(".kicad_mod") for f in kd["files"]), "filter kicad missing mod")
    print(f"PASS filter kicad ({len(kd['files'])} files)")

    r = client.post("/api/filter/evolve", json={
        "kind": "lpf", "n": 3, "f_c_mhz": 1000, "hadamard_order": 16,
        "max_steps": 30, "budget_s": 30})
    expect(r.status_code == 200 and "job_id" in r.json(), "POST /api/filter/evolve")
    jid = r.json()["job_id"]
    d = wait_terminal(jid, 30.0)
    expect(d["status"] == "done", f"filter evolve status {d['status']}")
    expect(np.isfinite(d["result"]["best_E"]), "filter evolve best_E")
    expect(d["result"]["design"]["kind"] == "lpf", "filter evolve design kind")
    print(f"PASS filter evolve (best_E {d['result']['best_E']:.3f}, "
          f"IL {d['result']['metrics']['il_db']:.2f} dB)")

    r = client.get("/js/tabs/filter.js")
    expect(r.status_code == 200 and "FILTER LAB" in r.text, "filter.js tab")
    expect("flt-layer-select" in r.text and "drawSweep" in r.text, "filter.js layers")
    r = client.get("/")
    expect('data-tab="filter"' in r.text, "index filter tab button")
    print("PASS filter static integrity")

    r = client.post("/api/materials/design", json={
        "kind": "cloth", "order": 16, "start": "sylvester", "pitch_mm": 1.0})
    expect(r.status_code == 200, f"materials cloth: {r.status_code} {r.text[:200]}")
    md = r.json()
    expect(md["stats"]["warp_runs"] > 0 and md["preview"]["prims"], "cloth empty")
    expect("2-layer" in (md["stats"].get("stack") or ""), "cloth not 2-layer")
    expect(md["stats"]["sites_zero"] > 0, "cloth missing 0-polarity sites")
    expect(md.get("key") and md["key"]["fill"], "cloth missing map key")
    expect(md["tiles"]["kronecker_h8"], "cloth H16 not 4-tile")
    r = client.post("/api/materials/design", json={
        "kind": "touchpad", "order": 8, "start": "sylvester"})
    expect(r.status_code == 200 and r.json()["stats"]["n_caps"] > 0, "touchpad caps")
    r = client.post("/api/materials/design", json={
        "kind": "metamaterial", "order": 16, "start": "sylvester"})
    expect(r.status_code == 200 and r.json()["stats"]["n_atoms"] == 4, "meta atoms")
    r = client.post("/api/materials/kicad", json={
        "kind": "cloth", "order": 8, "start": "sylvester"})
    mk = r.json()
    expect(r.status_code == 200 and any(f.endswith(".kicad_mod") for f in mk["files"]),
           "materials kicad")
    expect(mk.get("preview_board") and mk["preview_board"].get("prims"),
           "materials kicad missing preview_board")
    board_layers = {p.get("layer") for p in mk["preview_board"]["prims"]}
    expect("F.Cu" in board_layers and "B.Cu" in board_layers,
           f"materials board preview missing 2-layer copper: {board_layers}")
    expect(not any(p.get("kind") == "zone" for p in mk["preview_board"]["prims"]),
           "materials board preview still has a B.Cu GND pour")
    pcb_name = next(f for f in mk["files"] if f.endswith(".kicad_pcb"))
    pcb = client.get(f"/api/materials/kicad/{mk['token']}/{pcb_name}")
    expect(pcb.status_code == 200 and "(zone" not in pcb.text,
           "materials .kicad_pcb still contains a zone/pour")
    expect('(layers "B.Cu")' in pcb.text and '(layers "F.Cu")' in pcb.text,
           "materials .kicad_pcb missing per-layer pads")
    r = client.get("/js/tabs/materials.js")
    expect(r.status_code == 200 and "MATERIALS LAB" in r.text, "materials.js tab")
    expect("mat-prev-toggle" in r.text and "lastKicadPreviewBoard" in r.text,
           "materials.js missing FOOTPRINT/BOARD toggle")
    r = client.get("/")
    expect('data-tab="materials"' in r.text, "index materials tab")
    print(f"PASS materials lab (cloth/touch/meta + kicad {len(mk['files'])} files, "
          "2-layer no pour)")

    # ------------------------------------------------ Phase 4.5: HUD themes
    r = client.get("/css/themes.css")
    expect(r.status_code == 200 and '[data-theme="dmg"]' in r.text, "themes.css dmg")
    expect('[data-theme="plasma"]' in r.text and "--ramp-3" in r.text, "themes.css ramps")
    print("PASS themes.css")

    r = client.get("/js/theme.js")
    expect(r.status_code == 200 and "THEMES" in r.text and "recolorCanvas" in r.text,
           "theme.js exports")
    print("PASS theme.js")

    r = client.get("/js/viz/shaders.js")
    expect(r.status_code == 200 and "CRT_FRAG" in r.text and "DMG_FRAG" in r.text,
           "shaders.js fragments")
    expect("ELECTRIC_FRAG" in r.text and "QUANTUM_FRAG" in r.text, "shaders.js reserved frags")
    # Item 1 fix: DMG_FRAG quantizes through a nearest-stop palette carried
    # in uniform arrays (uPal[8]/uPalLum[8]/uPalCount + uPalEnabled,
    # installed by setPalette() from paletteStops()) — the earlier 256×1
    # DataTexture LUT went black on three r170's WebGL2 renderer
    # (RGBFormat/UNSIGNED_BYTE → unsized gl.RGB); the pixel-grid term
    # stays; QUANTUM_FRAG rides the cloud's uDensity
    expect("uPal[8]" in r.text and "uPalLum" in r.text and "uPalCount" in r.text
           and "uPalEnabled" in r.text and "uBivert" in r.text
           and "setPalette" in r.text and "paletteStops" in r.text,
           "shaders.js DMG palette uniform arrays")
    expect("uniform sampler2D uPalTex" not in r.text
           and "paletteLutBytes" not in r.text
           and "setPaletteTexture" not in r.text,
           "shaders.js old palette LUT texture gone")
    expect("fract(vUv * uRes)" in r.text, "DMG_FRAG pixel grid kept")
    expect("uDensity" in r.text and "uDensityOn" in r.text,
           "QUANTUM_FRAG radial density coupling")
    print("PASS shaders.js")

    # ------------------------------------------- Phase 6: CGB + settings panel
    r = client.get("/css/themes.css")
    expect('[data-theme="cgb"]' in r.text, "themes.css cgb block")
    expect('[data-theme="cgb"][data-variant="dark"]' in r.text, "themes.css cgb dark variant")
    expect('[data-theme="vga"][data-subtheme="cyberpunk"]' in r.text
           and '[data-theme="vga"][data-subtheme="thirdman"]' in r.text
           and '[data-theme="vga"][data-subtheme="evangelion"]' in r.text,
           "themes.css vga subthemes")
    # each VGA subtheme block must override --bg — otherwise switching
    # subthemes leaves the previous theme's background in place
    for _sub in ("cyberpunk", "thirdman", "evangelion"):
        _i = r.text.find(f'[data-theme="vga"][data-subtheme="{_sub}"]')
        expect(_i >= 0 and "--bg" in r.text[_i:_i + 400],
               f"themes.css vga {_sub} --bg override")
    print("PASS themes.css phase 6 blocks")

    r = client.get("/js/theme.js")
    expect('cgb: {' in r.text and '"palette56"' in r.text
           and "_lerpStops(DMG_ANCHORS, 56)" in r.text,
           "theme.js CGB 56-stop gradient (5bit snap replaced)")
    expect("themeRamp" in r.text and "ghostAmount" in r.text
           and "getSetting" in r.text, "theme.js phase 6 exports")
    # Item 9: display adjustments moved out of the LUT into a global CSS
    # filter on #app (settings.js) — the old per-canvas API must be gone
    expect("applyDisplayAdjustments" not in r.text, "theme.js LUT adjustments removed")
    # Item 8: subtheme/variant switches must re-fire themechange so
    # canvas/three consumers re-render under the new palette
    expect('key === "vgaSubtheme" || key === "cgbVariant"' in r.text
           and '"themechange"' in r.text, "theme.js subtheme fires themechange")
    print("PASS theme.js phase 6")

    # ---------------- Phase 7: exact DMG palette, mono ramps, glow, packs ---
    expect('"#1b2a09", "#0e450b", "#496b22", "#9a9e3f"' in r.text,
           "theme.js DMG exact 4-shade palette")
    expect("setCgbPalette" in r.text and "cgbPalette" in r.text,
           "theme.js CGB palette-pack support")
    expect("bivertInLut" in r.text, "theme.js DMG/CGB bivert in LUT")
    expect('--glow-eff' in r.text and "_applyGlow" in r.text,
           "theme.js brightness-driven phosphor glow")
    r = client.get("/css/themes.css")
    _i = r.text.find('[data-theme="dmg"]')
    expect(_i >= 0 and "--fg: #1b2a09" in r.text[_i:_i + 400]
           and "--bg: #9a9e3f" in r.text[_i:_i + 400],
           "themes.css DMG exact palette assignment")
    r = client.get("/css/app.css")
    expect("var(--glow-eff, var(--glow))" in r.text, "app.css glow-eff text-shadow")
    print("PASS phase 7 theme core")

    # ----- Item 5/7: forward LUT semantics, dmgLut, VGA subtheme colors ----
    r = client.get("/js/theme.js")
    expect("export function dmgLut" in r.text, "theme.js dmgLut helper")
    # the old inverted-theme LUT reversal made mostly-dark server PNGs a
    # solid light rectangle (micromag energy viewport) — it must stay gone;
    # only bivertInLut may reverse the ramp now
    expect("if (t.inverted) ramp" not in r.text, "theme.js inverted LUT walk removed")
    expect("bivertInLut" in r.text, "theme.js bivert still reverses the LUT")
    expect('blue: { bg: "#0000a8"' in r.text
           and 'cyberpunk: { bg: "#000000"' in r.text,
           "theme.js VGA subthemes carry bg/fg")
    expect('if (_current === "vga" && (name === "bg" || name === "fg"))' in r.text,
           "theme.js themeColor resolves vga subtheme bg/fg")
    print("PASS theme.js item 5/7 LUT semantics")

    # ----- Item 3: bivert full theme reversal + max-channel LUT + 3D lift ---
    expect("_biverted" in r.text and "fg: c.bg, bg: c.fg" in r.text
           and "_applyChrome" in r.text,
           "theme.js bivert swaps chrome vars on <html> (in-palette)")
    # intensity = max channel: Rec.601 luma capped pure green at ~149 and
    # never reached the lightest DMG shade — the coefficients must be gone
    expect("Math.max(s[i], s[i + 1], s[i + 2])" in r.text
           and "s[i + 1] * 150" not in r.text,
           "theme.js canvas LUT uses max-channel intensity")
    # themeRampSample: bivert un-reverses the inverted walk (chrome is
    # swapped too); inverted-family sampling lifted to the upper 60% of the
    # ramp so 3D views keep min contrast against the background
    expect("t.inverted !== _biverted()" in r.text
           and "0.4 + 0.6 * xu" in r.text,
           "theme.js themeRampSample bivert walk + contrast lift")
    print("PASS theme.js item 3 bivert/LUT/lift")

    # icons + controls.js (themed number steppers / file browse)
    for icon in ("arrow-up.svg", "arrow-down.svg", "browse.svg"):
        ri = client.get(f"/assets/icons/{icon}")
        expect(ri.status_code == 200 and "currentColor" in ri.text,
               f"icon {icon} serves, currentColor")
    r = client.get("/js/controls.js")
    expect(r.status_code == 200 and "enhanceControls" in r.text
           and "stepUp" in r.text, "controls.js served")
    # Item 1: the file browse button is text-only now — no icon mask
    expect("browse.svg" not in r.text and "ico-browse" not in r.text,
           "controls.js text-only browse button")
    r = client.get("/js/main.js")
    expect("enhanceControls" in r.text, "main.js enhanceControls hook")
    print("PASS icons + controls.js")

    # palette packs endpoint: parsed emulator .pal library
    r = client.get("/api/palettes")
    expect(r.status_code == 200, "GET /api/palettes")
    _pals = r.json()["palettes"]
    expect(len(_pals) >= 1 and len(_pals[0]["colors"]) >= 2
           and all(c.startswith("#") and len(c) == 7 for c in _pals[0]["colors"]),
           "/api/palettes ≥1 parsed palette with ≥2 #hex colors")
    expect(_pals[0]["category"] and _pals[0]["name"], "/api/palettes category/name")
    print(f"PASS /api/palettes ({len(_pals)} palettes)")

    r = client.get("/js/viz/shaders.js")
    expect("GHOST_FRAG" in r.text and "ghostAmount" in r.text, "shaders.js ghosting")
    print("PASS shaders.js ghosting")

    r = client.get("/")
    expect('id="settings-btn"' in r.text and 'id="settings-panel"' in r.text,
           "settings panel markup")
    expect('class="crt-fx"' in r.text and "themes.css" in r.text, "crt overlay / themes link")
    expect("theme-switch" not in r.text, "theme buttons moved into the settings panel")
    print("PASS settings panel markup")

    r = client.get("/js/settings.js")
    expect(r.status_code == 200 and "data-theme-id" in r.text and "setSetting" in r.text,
           "settings.js theme buttons + settings wiring")
    expect("DISPLAY_CONTROLS" in r.text and "vgaSubtheme" in r.text
           and "cgbVariant" in r.text, "settings.js per-display controls")
    # dmg/cgb bivert is the in-palette chrome swap in theme.js — the #app
    # invert(1) filter must stay EXCLUSIVE to plasma (bivertInLut exclusion)
    expect("!THEMES[currentTheme()].bivertInLut" in r.text,
           "settings.js invert(1) filter excluded for dmg/cgb")
    print("PASS settings.js")

    # ---------------- UI overhaul: single-viewport tabs + global filter -----
    r = client.get("/")
    expect('id="app"' in r.text, "index #app wrapper (global filter target)")
    r = client.get("/js/settings.js")
    expect("applyGlobalFilter" in r.text and "#app" in r.text,
           "settings.js global display filter on #app")
    expect("settings-close" in r.text and "Escape" in r.text,
           "settings.js dismissal (X / Esc / click-outside)")
    r = client.get("/js/tabs/matrix_lab.js")
    expect("startMorph" in r.text and "sp-overlay" in r.text,
           "matrix_lab in-place 2D→3D transmute morph")
    r = client.get("/js/tabs/search_studio.js")
    expect("run-viz" in r.text, "search_studio unified run panel")
    expect("export function deactivate" in r.text, "search_studio closes the job socket")
    r = client.get("/js/tabs/micromag_sim.js")
    expect("sim-layer-select" in r.text and "data-layer" in r.text,
           "micromag layer select")
    expect("sim-wave-select" in r.text and "data-series" in r.text
           and "waveChart" in r.text, "micromag unified waveform + series select")
    r = client.get("/js/viz/stripchart.js")
    expect("setVisible" in r.text, "stripchart series visibility")
    r = client.get("/js/tabs/terrain.js")
    expect("ter-layer-select" in r.text and "showTerLayer" in r.text,
           "terrain layer view")
    expect("data-oct" in r.text and "layers_f32" in r.text
           and "recombHeight" in r.text, "terrain octave mute/solo recombination")
    # Item 3: 3D viewport and layer view are equal-size 1:1 panels, layer
    # view LEFT; controls moved to the sidebar; renderer display size is
    # CSS-owned (setSize updateStyle=false) so no inline-style drift
    expect("ter-views" in r.text, "terrain equal-size viewport row")
    expect("setSize(w, h, false)" in r.text and '"Layers"' in r.text,
           "terrain CSS-owned canvas size + sidebar layer controls")
    expect("ter-survey-body" in r.text and "ter-mode-select" in r.text
           and "closeSurveyPopup" not in r.text,
           "terrain GENERATE/SURVEY sidebar toggle (popup leftover gone)")
    expect("drawPathProfile" in r.text and "buildSurveyScene" in r.text
           and "elevToY" in r.text,
           "terrain survey path-profile + metric 3-D link")
    expect("rawNum" in r.text and "RX blank" in r.text
           and "imagery_png_b64" in r.text and "sampleHm" in r.text
           and "PATH PROFILE" in r.text
           and "new THREE.Texture" in r.text
           and "ter-s-back" not in r.text and 'id: "ter-survey"' not in r.text,
           "terrain survey: blank RX, Texture imagery, mesh-sampled link")
    r = client.get("/css/app.css")
    expect(".ter-views > .panel" in r.text and ".ter-views canvas" in r.text,
           "app.css ter-views flex + 1:1 canvas rules")
    r = client.get("/js/tabs/orbitals.js")
    expect("orb-layer-select" in r.text and "selectOrbLayer" in r.text,
           "orbitals unified 3D/XZ viewport")
    expect("splatXZ" in r.text and "d.proj_png_b64" not in r.text,
           "orbitals client-side XZ splat (server proj unused)")
    # Item 4: [3D][XZ][BOTH] — BOTH overlays a transparent-cleared cloud on
    # the dimmed splat; Item 2: QUANTUM_FRAG clipped inside the viewport
    # (overflow:hidden), driven by the cloud's radial density profile, and
    # fed into the 3D scene as scene.background via CanvasTexture
    expect('"both"' in r.text and "alpha: true" in r.text
           and "initQuantum(viewportEl)" in r.text
           and 'orbLayer !== "both"' in r.text,
           "orbitals BOTH overlay + quantum-in-viewport")
    expect("overflow:hidden" in r.text and "scene3d.background" in r.text
           and "CanvasTexture" in r.text,
           "orbitals quantum clipped + scene background")
    expect("radialProfile" in r.text and '"uDensity"' in r.text
           and "uDensityOn" in r.text,
           "orbitals quantum driven by the cloud radial profile")
    # Item 3: the post pipeline must get the ACTIVE ramp (palette packs /
    # subthemes), never the static THEMES entry
    for _tab in ("terrain", "orbitals", "hoa_studio", "matrix_lab"):
        _t = client.get(f"/js/tabs/{_tab}.js").text
        expect("THEMES[currentTheme()].ramp" not in _t
               and "themeRamp()" in _t,
               f"{_tab} post pipeline uses themeRamp()")
    print("PASS UI overhaul markers")

    # ---------------- static integrity (module-graph regression guard) -----
    # A top-level throw in main.js/theme.js/settings.js kills the whole
    # module graph and all tab wiring (the Phase-6 TDZ bug). Guard the two
    # statically checkable halves of that class: every asset the page
    # references must serve, and every named import from /js/theme.js used
    # anywhere in js/ must be a real export of the rewritten theme.js.
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    r = client.get("/")
    expect('type="importmap"' in r.text, "index importmap present")
    assets = set(_re.findall(r'(?:src|href)="(/[^"]*)"', r.text))
    assets -= {"//"}  # protocol-relative would be external; none expected
    for a in sorted(assets):
        ra = client.get(a)
        expect(ra.status_code == 200, f"index asset {a} → {ra.status_code}")
    imp = _re.search(r'type="importmap">(.*?)</script>', r.text, _re.S)
    for name, path in _json.loads(imp.group(1))["imports"].items():
        rp = client.get(path)
        expect(rp.status_code == 200, f"importmap {name} → {path} → {rp.status_code}")
    print(f"PASS static assets ({len(assets)} refs + importmap resolve)")

    theme_src = client.get("/js/theme.js").text
    exported = set(_re.findall(r"export (?:function|const|let) (\w+)", theme_src))
    for block in _re.findall(r"export \{([^}]*)\}", theme_src):
        for n in block.split(","):
            n = n.strip().split(" as ")[-1].strip()
            if n:
                exported.add(n)
    js_root = _Path(__file__).parent / "static" / "js"
    n_imports = 0
    for js in sorted(js_root.rglob("*.js")):
        src = js.read_text()
        for block in _re.findall(
            r'import\s*\{([^}]*)\}\s*from\s*["\']/js/theme\.js["\']', src
        ):
            for n in block.split(","):
                n = n.strip().split(" as ")[0].strip()
                if not n:
                    continue
                n_imports += 1
                expect(n in exported,
                       f"{js.name} imports '{n}' from theme.js but theme.js does not export it")
        # every fetched module must also serve over HTTP
        rel = "/js/" + str(js.relative_to(js_root)).replace("\\", "/")
        expect(client.get(rel).status_code == 200, f"{rel} serves")
    print(f"PASS theme.js exports cover {n_imports} named imports across js/")

    # ---------------- frontend path pinning (Bug 1 regression guard) ------
    # Every URL the tab JS fetches must exist with the method the JS uses;
    # unknown /api/* paths must 404 (never fall through to StaticFiles 405).
    frontend_calls = [
        ("POST", "/api/search", {"engine": "maxdet", "order": 8, "budget_s": 2}),
        ("POST", "/api/sim/micromag", {"order": 8, "budget_s": 2, "start": "sylvester"}),
        ("GET", "/api/sim/flux-tiles?order=16&start=sylvester", None),
        ("POST", "/api/materials/design", {"kind": "cloth", "order": 8, "start": "sylvester"}),
        ("POST", "/api/materials/kicad", {"kind": "cloth", "order": 8, "start": "sylvester"}),
        ("POST", "/api/hoa/speakers", {"preset": "ring8", "order": 3}),
        ("POST", "/api/hoa/scene", {"sources": [{"az": 30, "el": 10, "freq": 440}],
                                    "order": 3, "duration": 0.05, "wav": False}),
        ("POST", "/api/hoa/decode-grid", {"hoa": [1.0] + [0.0] * 15, "n_azi": 6, "n_el": 3}),
        ("POST", "/api/gen/terrain", {"size": 32, "order": 16, "octaves": 2, "seed": 1}),
        ("POST", "/api/gen/orbital", {"n": 2, "l": 1, "m": 0, "samples": 500, "seed": 1}),
        ("POST", "/api/gen/noise-field", {"size": 32, "order": 16, "seed": 1}),
        ("GET", "/api/dag?max=128", None),
        ("GET", "/api/detbounds?max=64", None),
        ("GET", "/api/challenges", None),
        ("GET", "/api/palettes", None),
        ("POST", "/api/construct", {"order": 64, "method": "sylvester"}),
        ("GET", "/api/library/64", None),
        ("POST", "/api/verify", {"matrix": [[1, 1], [1, -1]]}),
        ("POST", "/api/viz/hadamard-space", {"order": 16, "mode": "rows", "kappa": 1}),
        ("POST", "/api/antenna/design", {"f_lo_mhz": 2400, "f_hi_mhz": 2485}),
        ("POST", "/api/antenna/parts", {"f_lo_mhz": 2400, "f_hi_mhz": 2485}),
        ("POST", "/api/antenna/kicad", {"design_type": "patch", "f_mhz": 2450}),
        ("GET", "/api/antenna/kicad/library", None),
        ("POST", "/api/antenna/kicad", {"design_type": "mifa", "f_mhz": 2450}),
        ("POST", "/api/antenna/fields", {"f_mhz": 150, "medium": "air", "n": 16,
                                         "max_steps": 10, "budget_s": 5}),
        ("POST", "/api/antenna/evolve", {"f_mhz": 2450, "max_steps": 10,
                                         "budget_s": 5}),
        ("POST", "/api/antenna/smith", {"f_lo_mhz": 950, "f_hi_mhz": 1050,
                                        "n_points": 3}),
        ("GET", "/api/noise/classes", None),
        ("POST", "/api/noise/analyze", {"path": "/nonexistent.wav"}),
        ("POST", "/api/filter/design", {"kind": "lpf", "n": 3, "f_c_mhz": 1000}),
        ("POST", "/api/filter/kicad", {"kind": "lpf", "n": 3, "f_c_mhz": 2450}),
        ("POST", "/api/filter/evolve", {"kind": "lpf", "n": 3, "f_c_mhz": 1000,
                                        "max_steps": 10, "budget_s": 5}),
        ("GET", f"/api/antenna/kicad/{kicad_pin_token}/{kicad_pin_name}", None),
    ]
    for method, path, body in frontend_calls:
        r = client.request(method, path, json=body) if body is not None else client.request(method, path)
        expect(r.status_code != 405, f"frontend path 405: {method} {path}")
        expect(r.status_code < 500, f"frontend path {r.status_code}: {method} {path}")
    r = client.post("/api/definitely-not-a-route", json={})
    expect(r.status_code == 404 and "no such API endpoint" in r.json()["detail"],
           "api catch-all should 404, got " + str(r.status_code))
    print(f"PASS frontend paths pinned ({len(frontend_calls)} calls non-405, catch-all 404)")

    print("selftest: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
