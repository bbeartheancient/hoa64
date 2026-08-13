#!/usr/bin/env python3
"""Rebuild the open_hadamard CSV library from scratch.

Requires SageMath for orders > 2000 (uses hadamard_matrix(n)).
Orders <= 2000 are built from the toolchain (Paley, Miyamoto, CW, etc.).
The 12 historically-significant Alpoge matrices are decompressed from
the matrices/ directory.
"""
import sys, gzip, numpy as np
from pathlib import Path

OUT = Path("open_hadamard")
OUT.mkdir(exist_ok=True)

# Step 1: Decompress the 12 Alpoge matrices
matrices_dir = Path(__file__).parent / "matrices"
for gz in sorted(matrices_dir.glob("*.csv.gz")):
    name = gz.stem.replace(".csv", "")
    dest = OUT / f"{name}.csv"
    if not dest.is_file():
        raw = gzip.decompress(gz.read_bytes())
        dest.write_bytes(raw)
        print(f"  decompressed {name}")

# Step 2: Build all constructible orders via the toolchain
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hoa64.evolve import main as evolve_main
evolve_main()

