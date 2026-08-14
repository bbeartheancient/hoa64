"""Antenna-type recommender: rank canonical constructions against a band and site.

Pure physics scoring over the `em_physics.ANTENNA_TYPES` registry — every
score component traces to a computable physical quantity; there are no
tunable "fuzzy" weights.  Given a frequency band [f_lo, f_hi], a medium,
and `SiteConditions`, each registered antenna type is built at the band
center f_c = (f_lo+f_hi)/2 via its textbook builder and scored on five
fit factors, each clamped to (0, 1]:

* **bw_fit** — fractional-bandwidth adequacy.  The required fractional
  bandwidth is

      B_req = (f_hi − f_lo)/f_c;

  the builder reports its VSWR 2:1 fractional bandwidth B_av (cavity-Q
  model for the patch, thin-wire rules for wire antennas, etc.).  The
  factor is min(1, B_av/B_req): full credit when the physics bandwidth
  covers the band, linear-in-shortfall penalty when it does not.  Both
  numbers are reported.

* **size_fit** — largest linear dimension of the construction (dipole
  length, patch width, yagi boom, helix axial length, …) vs
  `site.max_size_m`: factor min(1, L_max/L_ant).  Element *counts*
  (yagi elements, helix turns) are not dimensions and are excluded.

* **gain_fit** — link closure.  When `site.range_m` and
  `site.rx_sensitivity_dbw` are set, `em_physics.link_budget` is run at
  f_lo (the band edge with the highest medium attenuation α — worst
  case) with the candidate's gain.  The link margin

      M = P_r − P_sens        (dB)

  is the physical quantity; the score factor maps it through the
  logistic σ(M/3 dB) = 1/(1+e^(−M/3)) — 0.5 at break-even, →1 with
  margin, →0 when the link cannot close.  With no link requirement the
  factor is 1 (gain unpenalized).

* **medium_fit** — propagation attenuation.  The lossy-dielectric
  attenuation constant α(f_c) (Np/m, from `medium_params`) gives the
  one-wavelength field survival e^(−αλ) ≈ 1 − 8.7·αλ dB; the factor is
  exactly that survival, exp(−α·λ_medium) — 1 in air, ≈0.29 per
  wavelength in seawater at 2.45 GHz.  Because every builder keys its
  dimensions off the medium wavelength λ = 2π/β, shrinking in dense
  media is reported (λ/λ₀), not recomputed here.  Types that require a
  ground plane (`_GROUND_TYPES`) are additionally penalized ×0.7 in
  conductive immersion: seawater (σ ≈ 4 S/m) shorts uninsulated ground
  contacts, which the image-theory models do not capture.

* **pol_fit** — polarization match.  `site.polarization` None ≣ no
  requirement (factor 1); a match is 1; a linear↔circular mismatch is
  exactly the 3 dB polarization-mismatch loss (a linear antenna captures
  half the power of a circular wave): factor 0.5.

**Composite.**  The score is the *product* of the five fit factors — a
multiplicative form so that a single physical disqualifier (a link that
cannot close, a band the resonator cannot cover) dominates the ranking
rather than being averaged away.  Physically impossible mountings are
recorded as `viable = False` with the reason (monopole/slot need a
ground plane; patch/pifa/meander are printed structures needing a
pcb/ground mount) and carry a ×0.1 viability factor — they still
appear, scored, instead of being silently skipped.  Entries are sorted
by score, descending; `explain` renders the full reasoning trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .em_physics import ANTENNA_TYPES, MEDIA, link_budget, medium_params

# Types whose pattern/impedance models assume a ground plane.
_GROUND_TYPES = {"monopole", "slot", "patch", "pifa", "meander"}

# Types that are printed structures — they need a PCB/ground substrate mount.
_PCB_TYPES = {"patch", "pifa", "meander"}

# dimensions_m keys that are element counts, not lengths.
_COUNT_KEYS = {"elements", "turns"}

# dB per Np of field attenuation: 20·log₁₀(e).
_DB_PER_NP = 20.0 * math.log10(math.e)      # ≈ 8.6859


@dataclass
class SiteConditions:
    """Physical constraints of the installation site.

    max_size_m: largest linear dimension the mount allows (None ≣ no
    limit).  ground_plane: metal ground available at the mount.
    range_m / p_tx_dbw / g_rx_dbi / rx_sensitivity_dbw: link-budget
    parameters — with both range_m and rx_sensitivity_dbw set, the link
    must close and its margin becomes a score factor.  polarization:
    "linear" | "circular" | None (either).  mounting: "free" | "pcb" |
    "ground".
    """
    max_size_m: float | None = None
    ground_plane: bool = True
    range_m: float | None = None
    p_tx_dbw: float = 0.0
    g_rx_dbi: float = 2.15
    rx_sensitivity_dbw: float | None = None
    polarization: str | None = None
    mounting: str = "free"


def _largest_dimension(dimensions_m: dict) -> float:
    """Largest linear dimension (m) of a builder's dimensions_m dict."""
    vals = [float(v) for k, v in dimensions_m.items()
            if k not in _COUNT_KEYS and isinstance(v, (int, float))]
    return max(vals) if vals else 0.0


def _link_required(site: SiteConditions) -> bool:
    return site.range_m is not None and site.rx_sensitivity_dbw is not None


def recommend(f_lo_hz: float, f_hi_hz: float, medium: str = "air",
              site: SiteConditions | None = None) -> list[dict]:
    """Rank every type in `ANTENNA_TYPES` against the band and site.

    Builds each antenna at the band center f_c = (f_lo+f_hi)/2 in
    `medium`, computes the five fit factors (see module docstring), and
    returns entries sorted by composite score (product of factors,
    ×0.1 when not viable), best first.  Each entry:

        {type, score, viable, fits: {bw_fit, size_fit, gain_fit,
        medium_fit, pol_fit}, design: <builder dict>, reasons: [str]}

    `reasons` states every factor and disqualifier in plain English —
    the accuracy trace shown in the UI.
    """
    if site is None:
        site = SiteConditions()
    if f_hi_hz < f_lo_hz:
        raise ValueError("f_hi_hz must be ≥ f_lo_hz")
    f_c = 0.5 * (f_lo_hz + f_hi_hz)
    bw_req = (f_hi_hz - f_lo_hz) / f_c
    mp = medium_params(f_c, medium)
    lam, lam0, alpha = mp["wavelength"], mp["lambda0"], mp["alpha"]
    in_water = isinstance(medium, str) and medium != "air"

    entries: list[dict] = []
    for name, builder in ANTENNA_TYPES.items():
        design = builder(f_c, medium)
        reasons: list[str] = []
        fits: dict[str, dict] = {}

        # --- viability: physically impossible mountings ----------------
        viable = True
        if name in {"monopole", "slot"} and not site.ground_plane:
            viable = False
            reasons.append(
                f"{name} needs a metal ground plane (image theory) — "
                "the site has none: NOT VIABLE")
        if name in _PCB_TYPES and site.mounting not in ("pcb", "ground"):
            viable = False
            reasons.append(
                f"{name} is a printed structure needing a pcb/ground "
                f"substrate mount — mounting is {site.mounting!r}: "
                "NOT VIABLE")

        # --- bw_fit ----------------------------------------------------
        if bw_req <= 0.0:
            bw_factor = 1.0
            reasons.append(
                f"fractional BW: single-frequency requirement "
                f"(B_req = 0) — the type's {design['bandwidth_frac']*100:.2f} % "
                "bandwidth trivially fits")
        else:
            bw_av = design["bandwidth_frac"]
            bw_factor = min(1.0, bw_av / bw_req)
            verdict = ("fits" if bw_factor >= 1.0 else
                       f"SHORTFALL ×{bw_req / bw_av:.2f}")
            reasons.append(
                f"fractional BW: required B_req = (f_hi−f_lo)/f_c = "
                f"{bw_req*100:.2f} % vs available {bw_av*100:.2f} % — "
                f"{verdict}")
        fits["bw_fit"] = {"required_frac": bw_req,
                          "available_frac": design["bandwidth_frac"],
                          "factor": bw_factor}

        # --- size_fit --------------------------------------------------
        largest = _largest_dimension(design["dimensions_m"])
        if site.max_size_m is None:
            size_factor = 1.0
            reasons.append(
                f"size: largest dimension {largest*1e2:.2f} cm, no site "
                "size limit")
        else:
            size_factor = min(1.0, site.max_size_m / largest) \
                if largest > 0.0 else 1.0
            verdict = ("fits" if size_factor >= 1.0 else "TOO LARGE")
            reasons.append(
                f"size: largest dimension {largest*1e2:.2f} cm vs site "
                f"limit {site.max_size_m*1e2:.2f} cm — {verdict}")
        fits["size_fit"] = {"largest_dim_m": largest,
                            "max_size_m": site.max_size_m,
                            "factor": size_factor}

        # --- gain_fit --------------------------------------------------
        if _link_required(site):
            lb = link_budget(site.p_tx_dbw, design["gain_dbi"],
                             site.g_rx_dbi, site.range_m, f_lo_hz, medium)
            margin = lb["received_dbw"] - site.rx_sensitivity_dbw
            gain_factor = 1.0 / (1.0 + math.exp(-margin / 3.0))
            closes = margin >= 0.0
            reasons.append(
                f"link budget at f_lo = {f_lo_hz:.6g} Hz (worst-case "
                f"attenuation): P_r = {lb['received_dbw']:.2f} dBW "
                f"(FSPL {lb['fspl_db']:.1f} dB + medium "
                f"{lb['medium_loss_db']:.2f} dB over {site.range_m:.3g} m) "
                f"vs sensitivity {site.rx_sensitivity_dbw:.2f} dBW — "
                f"margin {margin:+.2f} dB, link "
                f"{'CLOSES' if closes else 'FAILS'}")
            fits["gain_fit"] = {"margin_db": margin, "closes": closes,
                                "received_dbw": lb["received_dbw"],
                                "sensitivity_dbw": site.rx_sensitivity_dbw,
                                "link_budget": lb, "factor": gain_factor}
        else:
            gain_factor = 1.0
            reasons.append(
                f"gain: {design['gain_dbi']:.2f} dBi (no link requirement "
                "set — gain not scored)")
            fits["gain_fit"] = {"margin_db": None, "closes": None,
                                "factor": gain_factor}

        # --- medium_fit ------------------------------------------------
        medium_factor = math.exp(-alpha * lam)   # 1-wavelength survival
        med_reasons = [
            f"medium: α(f_c) = {alpha:.4g} Np/m "
            f"({_DB_PER_NP * alpha:.4g} dB/m) in {medium!r}; "
            f"field survival over one λ_medium = {lam:.4g} m is "
            f"e^(−αλ) = {medium_factor:.4f}"]
        if in_water:
            med_reasons.append(
                f"λ_medium/λ₀ = {lam / lam0:.4f} — all dimensions key off "
                f"the short medium wavelength and shrink by that factor")
            if name in _GROUND_TYPES:
                medium_factor *= 0.7
                med_reasons.append(
                    f"{name} needs a ground plane; in water immersion "
                    "(seawater σ ≈ 4 S/m) conductive water shorts "
                    "uninsulated ground contacts — penalized ×0.7")
        reasons.extend(med_reasons)
        fits["medium_fit"] = {
            "alpha_np_per_m": alpha,
            "medium_loss_db_per_m": _DB_PER_NP * alpha,
            "wavelength_m": lam,
            "shrink_vs_air": lam / lam0,
            "factor": medium_factor}

        # --- pol_fit ---------------------------------------------------
        ant_pol = design["polarization"]
        if site.polarization is None:
            pol_factor = 1.0
            reasons.append(
                f"polarization: antenna is {ant_pol}, no requirement set")
        elif site.polarization == ant_pol:
            pol_factor = 1.0
            reasons.append(
                f"polarization: requested {site.polarization} ≣ antenna "
                f"{ant_pol} — match")
        else:
            pol_factor = 0.5
            reasons.append(
                f"polarization: requested {site.polarization} vs antenna "
                f"{ant_pol} — linear↔circular mismatch costs exactly the "
                "3 dB polarization-mismatch loss (factor 0.5)")
        fits["pol_fit"] = {"requested": site.polarization,
                           "antenna": ant_pol, "factor": pol_factor}

        # --- composite --------------------------------------------------
        score = (bw_factor * size_factor * gain_factor
                 * medium_factor * pol_factor)
        if not viable:
            score *= 0.1
        entries.append({
            "type": name,
            "score": score,
            "viable": viable,
            "fits": fits,
            "design": design,
            "reasons": reasons,
        })

    entries.sort(key=lambda e: e["score"], reverse=True)
    return entries


def explain(entry: dict) -> str:
    """Render one `recommend` entry's reasoning trace as text lines."""
    d = entry["design"]
    dims = ", ".join(f"{k} = {v:.4g} m" if isinstance(v, float) else
                     f"{k} = {v}" for k, v in d["dimensions_m"].items())
    head = (f"{entry['type']}  score {entry['score']:.4f}  "
            f"[{'viable' if entry['viable'] else 'NOT VIABLE'}]  "
            f"gain {d['gain_dbi']:.2f} dBi, Z_in = {d['z_in_ohm']}, "
            f"BW {d['bandwidth_frac']*100:.2f} %, {d['polarization']}")
    return "\n".join([head, f"  dims: {dims}"]
                     + [f"  - {r}" for r in entry["reasons"]])


# --- self-check --------------------------------------------------------------

if __name__ == "__main__":
    # --- 2.4–2.5 GHz ISM, air, pcb mount, 4 cm limit → patch/pifa top-2 ---
    site = SiteConditions(max_size_m=0.04, mounting="pcb")
    rec = recommend(2.4e9, 2.5e9, "air", site)
    top2 = {e["type"] for e in rec[:2]}
    assert top2 & {"patch", "pifa"}, top2
    bw = rec[0]["fits"]["bw_fit"]
    assert abs(bw["required_frac"] - 0.040816) < 1e-4, bw
    assert any("fractional BW" in r for e in rec for r in e["reasons"])
    dip = next(e for e in rec if e["type"] == "dipole")
    assert dip["fits"]["bw_fit"]["factor"] == 1.0     # 8 % ≥ 4.08 %, fits
    print(f"PASS  2.4–2.5 GHz pcb/4 cm: top-2 = "
          f"{[e['type'] for e in rec[:2]]}, required fractional BW = "
          f"{bw['required_frac']*100:.2f} % ≈ 4.08 %, dipole (8 %) fits")

    # --- 98–102 MHz, air, free mount (no ground), 1.6 m limit → dipole ---
    site = SiteConditions(max_size_m=1.6, ground_plane=False, mounting="free")
    rec = recommend(98e6, 102e6, "air", site)
    assert rec[0]["type"] in ("dipole", "yagi"), rec[0]["type"]
    assert rec[0]["type"] == "dipole"
    print(f"PASS  100 MHz air free mount: top = {rec[0]['type']} "
          f"(score {rec[0]['score']:.3f})")

    # --- 2.45 GHz fresh water, 0.5 m link must close at −120 dBW ---------
    site = SiteConditions(range_m=0.5, rx_sensitivity_dbw=-120.0)
    rec = recommend(2.45e9, 2.45e9, "water", site)
    assert all(e["fits"]["medium_fit"]["alpha_np_per_m"] > 0.0 for e in rec)
    assert all(e["fits"]["medium_fit"]["medium_loss_db_per_m"] > 0.0
               for e in rec)
    top = rec[0]
    assert top["fits"]["gain_fit"]["closes"], top["type"]
    fresh_loss = top["fits"]["gain_fit"]["link_budget"]["medium_loss_db"]
    rec_sea = recommend(2.45e9, 2.45e9, "water_sea", site)
    sea_loss = next(e for e in rec_sea
                    if e["type"] == top["type"]
                    )["fits"]["gain_fit"]["link_budget"]["medium_loss_db"]
    assert sea_loss > 20.0 * fresh_loss, (fresh_loss, sea_loss)
    print(f"PASS  2.45 GHz water: top = {top['type']}, link closes "
          f"(margin {top['fits']['gain_fit']['margin_db']:+.1f} dB); "
          f"medium loss {fresh_loss:.2f} dB fresh vs {sea_loss:.1f} dB "
          "seawater (0.5 m)")

    # --- monopole without a ground plane → not viable --------------------
    site = SiteConditions(ground_plane=False)
    rec = recommend(2.45e9, 2.45e9, "air", site)
    mono = next(e for e in rec if e["type"] == "monopole")
    assert mono["viable"] is False
    assert any("ground plane" in r for r in mono["reasons"])
    print(f"PASS  monopole, ground_plane=False: viable = False "
          f"(score {mono['score']:.4f}, reason recorded)")

    # --- every score in (0, 1], sorted descending; explain renders -------
    for medium in MEDIA:
        rec = recommend(1e9, 1.1e9, medium, SiteConditions())
        assert all(0.0 < e["score"] <= 1.0 for e in rec)
        assert all(rec[i]["score"] >= rec[i + 1]["score"]
                   for i in range(len(rec) - 1))
    txt = explain(rec[0])
    assert rec[0]["type"] in txt and "score" in txt and "- " in txt
    print(f"PASS  all media: scores in (0,1], sorted; explain() renders "
          f"{len(txt.splitlines())} lines for {rec[0]['type']!r}")
    print("antenna_design selftest: all checks passed")
