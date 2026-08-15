"""DARPA's 23 mathematical challenges — the program frame for this lab.

In 2007 DARPA announced 23 mathematical challenges, stating that the
solution of any one of them "would have the effect of dramatically
revolutionizing mathematics and thereby strengthening the scientific and
technological capabilities" of the DoD.  The list spans the brain,
networks, stochasticity, fluids, quantum information, the Langlands
program, the Riemann hypothesis, and the fundamental laws of biology —
it is a map of where mathematics is expected to matter next.

A Hadamard-matrix solver / HOA / RF-physics workbench is obviously not
going to *settle* any of these.  But the lab's engines — annealers,
max-det descents, MoM/FDTD solvers, hyperbolic embeddings, a running
Riemann-consistency checker — each touch the mathematical territory of
one challenge or another.  Tracking the alignment keeps the roadmap
honest: when a new engine lands, we can ask which challenge it serves.

Honesty rule
------------
The statuses in ``ALIGNMENT`` describe **tooling alignment, not progress
toward solutions**.  ``active`` means a real, running piece of code in
this package works on the challenge's subject matter (today that is
exactly one: the ``rh.py`` |Δₙ| bound consistency checker against the
Riemann hypothesis, used as a gate by the search daemons).  ``partial``
means engines exist that operate in the challenge's mathematical
neighbourhood.  ``latent`` means the machinery would transfer but has
not been aimed there.  ``none`` means nothing in the repo touches it.
No status here claims, or implies, a solution.

Source of the statements: DARPA's 23 challenge questions as transcribed
at compmath.wordpress.com/about/10-the-big-picture-darpas-23-challenge-questions
"""

from __future__ import annotations

import os
import re

CHALLENGES: list[dict] = [
    {"n": 1, "title": "The Mathematics of the Brain",
     "statement": "Develop a mathematical theory to build a functional model of the brain that is mathematically consistent and predictive rather than merely biologically inspired."},
    {"n": 2, "title": "The Dynamics of Networks",
     "statement": "Develop the high-dimensional mathematics needed to accurately model and predict behavior in large-scale distributed networks that evolve over time occurring in communication, biology and the social sciences."},
    {"n": 3, "title": "Capture and Harness Stochasticity in Nature",
     "statement": "Address Mumford's call for new mathematics for the 21st century. Develop methods that capture persistence in stochastic environments."},
    {"n": 4, "title": "21st Century Fluids",
     "statement": "Classical fluid dynamics and the Navier-Stokes Equation were extraordinarily successful in obtaining quantitative understanding of shock waves, turbulence and solitons, but new methods are needed to tackle complex fluids such as foams, suspensions, gels and liquid crystals."},
    {"n": 5, "title": "Biological Quantum Field Theory",
     "statement": "Quantum and statistical methods have had great success modeling virus evolution. Can such techniques be used to model more complex systems such as bacteria? Can these techniques be used to control pathogen evolution?"},
    {"n": 6, "title": "Computational Duality",
     "statement": "Duality in mathematics has been a profound tool for theoretical understanding. Can it be extended to develop principled computational techniques where duality and geometry are the basis for novel algorithms?"},
    {"n": 7, "title": "Occam's Razor in Many Dimensions",
     "statement": "As data collection increases can we \"do more with less\" by finding lower bounds for sensing complexity in systems? This is related to questions about entropy maximization algorithms."},
    {"n": 8, "title": "Beyond Convex Optimization",
     "statement": "Can linear algebra be replaced by algebraic geometry in a systematic way?"},
    {"n": 9, "title": "Physical Consequences of Perelman's Proof of Thurston's Geometrization Theorem",
     "statement": "Can profound theoretical advances in understanding three dimensions be applied to construct and manipulate structures across scales to fabricate novel materials?"},
    {"n": 10, "title": "Algorithmic Origami and Biology",
     "statement": "Build a stronger mathematical theory for isometric and rigid embedding that can give insight into protein folding."},
    {"n": 11, "title": "Optimal Nanostructures",
     "statement": "Develop new mathematics for constructing optimal globally symmetric structures by following simple local rules via the process of nanoscale self-assembly."},
    {"n": 12, "title": "The Mathematics of Quantum Computing, Algorithms, and Entanglement",
     "statement": "In the last century we learned how quantum phenomena shape our world. In the coming century we need to develop the mathematics required to control the quantum world."},
    {"n": 13, "title": "Creating a Game Theory that Scales",
     "statement": "What new scalable mathematics is needed to replace the traditional Partial Differential Equations (PDE) approach to differential games?"},
    {"n": 14, "title": "An Information Theory for Virus Evolution",
     "statement": "Can Shannon's theory shed light on this fundamental area of biology?"},
    {"n": 15, "title": "The Geometry of Genome Space",
     "statement": "What notion of distance is needed to incorporate biological utility?"},
    {"n": 16, "title": "Symmetries and Action Principles for Biology",
     "statement": "Extend our understanding of symmetries and action principles in biology along the lines of classical thermodynamics, to include important biological concepts such as robustness, modularity, evolvability and variability."},
    {"n": 17, "title": "Geometric Langlands and Quantum Physics",
     "statement": "How does the Langlands program, which originated in number theory and representation theory, explain the fundamental symmetries of physics? And vice versa?"},
    {"n": 18, "title": "Arithmetic Langlands, Topology, and Geometry",
     "statement": "What is the role of homotopy theory in the classical, geometric, and quantum Langlands programs?"},
    {"n": 19, "title": "Settle the Riemann Hypothesis",
     "statement": "The Holy Grail of number theory."},
    {"n": 20, "title": "Computation at Scale",
     "statement": "How can we develop asymptotics for a world with massively many degrees of freedom?"},
    {"n": 21, "title": "Settle the Hodge Conjecture",
     "statement": "This conjecture in algebraic geometry is a metaphor for transforming transcendental computations into algebraic ones."},
    {"n": 22, "title": "Settle the Smooth Poincare Conjecture in Dimension 4",
     "statement": "What are the implications for space-time and cosmology? And might the answer unlock the secret of \"dark energy\"?"},
    {"n": 23, "title": "What are the Fundamental Laws of Biology",
     "statement": "This question will remain front and center for the next 100 years. DARPA places this challenge last as finding these laws will undoubtedly require the mathematics developed in answering several of the questions listed above."},
]

STATUSES = ("active", "partial", "latent", "none")

# Alignment of the lab's engines with the challenges — tooling
# alignment, NOT progress (see the module docstring).  engines entries
# are {module, tab}: module = a file in this package, tab = the webapp
# data-tab where that engine is visible, or None if it has no UI home.
ALIGNMENT: list[dict] = [
    {"n": 1, "status": "none", "engines": [],
     "note": "Would need a predictive dynamical-systems model of neural activity; nothing here models the brain."},
    {"n": 2, "status": "partial", "engines": [
        {"module": "mcu.py", "tab": "microcontroller"},
        {"module": "game_of_hadamard.py", "tab": "library"},
     ],
     "note": "ESP-NOW RSSI tomography infers structure from a live wireless mesh; the construction DAG is a evolving dependency network — observation, not prediction theory."},
    {"n": 3, "status": "latent", "engines": [
        {"module": "dit_noise.py", "tab": "noise"},
        {"module": "terrain.py", "tab": "terrain"},
     ],
     "note": "The DiT noise classifier and Hadamard fBm both model persistent stochastic structure, but neither captures/harnesses it in Mumford's sense."},
    {"n": 4, "status": "latent", "engines": [
        {"module": "fdtd.py", "tab": "antenna"},
     ],
     "note": "fdtd.py is a Yee-grid EM solver, not fluids — but staggered-grid time-stepping machinery generalizes to Navier–Stokes / LBM solvers for complex fluids."},
    {"n": 5, "status": "none", "engines": [],
     "note": "Would need stochastic/QFT-style population models of pathogens; no biological modelling here."},
    {"n": 6, "status": "partial", "engines": [
        {"module": "hadamard_space.py", "tab": "matrix_lab"},
        {"module": "williamson.py", "tab": "search_studio"},
     ],
     "note": "ℍ³ transmute embeds matrices dually as hyperbolic geometry; Williamson/GS search minimizes an FFT power spectral density — primal/dual domain algorithms in practice."},
    {"n": 7, "status": "latent", "engines": [
        {"module": "basis.py", "tab": "hoa_studio"},
        {"module": "encode.py", "tab": "hoa_studio"},
        {"module": "noise_data.py", "tab": "noise"},
     ],
     "note": "64-channel HOA is a fixed-basis sparse sampling of the sound field and the mel front-end compresses audio — sensing-complexity lower bounds are not addressed."},
    {"n": 8, "status": "partial", "engines": [
        {"module": "micromag.py", "tab": "micromag_sim"},
        {"module": "rf_filter.py", "tab": "filter"},
        {"module": "hadamard.py", "tab": "search_studio"},
        {"module": "tile_search.py", "tab": "search_studio"},
        {"module": "gerzon.py", "tab": "search_studio"},
        {"module": "holographic.py", "tab": "search_studio"},
        {"module": "crown.py", "tab": "search_studio"},
     ],
     "note": "The whole search stack is non-convex optimization: micromagnetic SA, Hadamard-seeded filter_sa, ILS max-det descent, H2-cell tile SA, Gerzon AB cell SA. Heuristics, not algebraic-geometry systematics."},
    {"n": 9, "status": "none", "engines": [],
     "note": "Would need Ricci-flow / 3-manifold computation; not present."},
    {"n": 10, "status": "none", "engines": [],
     "note": "Would need computational-geometry folding / isometric-embedding models; not present."},
    {"n": 11, "status": "partial", "engines": [
        {"module": "materials.py", "tab": "materials"},
        {"module": "micromag.py", "tab": "micromag_sim"},
     ],
     "note": "H.8 flux-tile spin-ice cells and Walsh lattices are globally symmetric structures from simple local rules — macroscale, not nanoscale self-assembly."},
    {"n": 12, "status": "partial", "engines": [
        {"module": "orbitals.py", "tab": "orbitals"},
        {"module": "hadamard.py", "tab": "matrix_lab"},
     ],
     "note": "Hydrogenic |ψ|² sampling and the Walsh–Hadamard gate set are the mathematics of quantum states and quantum circuits, minus entanglement theory."},
    {"n": 13, "status": "none", "engines": [],
     "note": "Would need scalable differential-games / multi-agent solvers; not present."},
    {"n": 14, "status": "none", "engines": [],
     "note": "Would need information theory over viral sequence evolution; not present."},
    {"n": 15, "status": "none", "engines": [],
     "note": "Would need metric geometry on genomic spaces; not present."},
    {"n": 16, "status": "none", "engines": [],
     "note": "Would need symmetry / variational principles for biological robustness and evolvability; not present."},
    {"n": 17, "status": "none", "engines": [],
     "note": "Would need representation-theoretic Langlands tooling; far outside this lab's scope."},
    {"n": 18, "status": "none", "engines": [],
     "note": "Would need arithmetic-geometry and homotopy-theory computation; far outside this lab's scope."},
    {"n": 19, "status": "active", "engines": [
        {"module": "rh.py", "tab": None},
        {"module": "hadamard.py", "tab": "matrix_lab"},
     ],
     "note": "rh.py is a real running checker: the RH-consistent |Δₙ| bounds gate the search daemons, and hadamard.py's max-det machinery works the same determinant-bound territory."},
    {"n": 20, "status": "partial", "engines": [
        {"module": "evolve.py", "tab": None},
        {"module": "search_daemon.py", "tab": None},
        {"module": "fdtd.py", "tab": "antenna"},
        {"module": "muon.py", "tab": None},
     ],
     "note": "Long-running gap-filling daemons, 3-D FDTD grids, and the Muon optimizer are massively-many-degrees-of-freedom computation — engineering scale, not asymptotic theory."},
    {"n": 21, "status": "none", "engines": [],
     "note": "Would need Hodge-theory / algebraic-cycle computation; not present."},
    {"n": 22, "status": "none", "engines": [],
     "note": "Would need smooth 4-manifold invariant computation; not present."},
    {"n": 23, "status": "none", "engines": [],
     "note": "The program-level question — would require the mathematics of several challenges above first."},
]


def summary() -> dict:
    """Counts by status plus the anchor list (challenges with engines)."""
    counts = {s: 0 for s in STATUSES}
    for a in ALIGNMENT:
        counts[a["status"]] += 1
    return {
        "counts": counts,
        "total": len(CHALLENGES),
        "anchors": [a["n"] for a in ALIGNMENT if a["engines"]],
    }


def _selfcheck() -> None:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    files = set(os.listdir(pkg_dir))

    assert len(CHALLENGES) == 23, f"need 23 challenges, got {len(CHALLENGES)}"
    ns = [c["n"] for c in CHALLENGES]
    assert sorted(ns) == list(range(1, 24)) and len(set(ns)) == 23, "n must be unique 1..23"
    for c in CHALLENGES:
        assert c["title"] and c["statement"], f"challenge {c['n']} missing title/statement"

    assert len(ALIGNMENT) == 23, f"need 23 alignment entries, got {len(ALIGNMENT)}"
    aligned = set()
    for a in ALIGNMENT:
        n = a["n"]
        assert 1 <= n <= 23 and n not in aligned, f"bad or duplicate alignment n={n}"
        aligned.add(n)
        assert a["status"] in STATUSES, f"n={n}: bad status {a['status']!r}"
        assert a["note"], f"n={n}: missing note"
        for e in a["engines"]:
            assert e["module"] in files, f"n={n}: module {e['module']} not in package dir"
    assert aligned == set(ns), "ALIGNMENT must cover every challenge exactly once"

    # Tabs must be real data-tab values from the frontend — keeps the
    # mapping honest against renames.
    index = os.path.join(pkg_dir, "webapp", "static", "index.html")
    with open(index, encoding="utf-8") as f:
        tabs = set(re.findall(r'data-tab="([^"]+)"', f.read()))
    assert tabs, "no data-tab values found in index.html"
    for a in ALIGNMENT:
        for e in a["engines"]:
            assert e["tab"] is None or e["tab"] in tabs, \
                f"n={a['n']}: tab {e['tab']!r} not a data-tab in index.html"

    s = summary()
    assert s["total"] == 23 and sum(s["counts"].values()) == 23
    assert s["counts"]["active"] == 1, "exactly one active anchor (rh.py) expected"
    print(f"darpa_challenges self-check OK — {s['counts']} anchors {s['anchors']}")


if __name__ == "__main__":
    _selfcheck()
