#!/usr/bin/env python3
"""
Generate manifold voxel visualization PNGs for the voxel_viz paper gallery.

Produces three gallery figures:
  - synthetic: helix, swiss_roll, torus
  - tabular:   iris, digits
  - highdim:   mnist (keras), cifar10 (keras)

Each dataset renders a multi-scalar 2×2 panel via render_multi(off_screen=True).

Usage
-----
    cd /path/to/waverider
    python benchmarks/canonical_tests/gen_gallery_voxels.py

Output
------
    papers/voxel_viz/figures/{dataset}_voxel.png

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

SCRIPT = _HERE / "manifold_voxel_viz.py"
OUT_DIR = _ROOT / "papers" / "voxel_viz" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# (dataset_name, extra_args)
DATASETS = [
    # --- synthetic ---
    ("helix", ["--dataset", "helix", "--n-points", "600", "--multi-scalar"]),
    ("swiss_roll", ["--dataset", "swiss_roll", "--n-points", "600", "--multi-scalar"]),
    ("torus", ["--dataset", "torus", "--n-points", "600", "--multi-scalar"]),
    # --- tabular ---
    ("iris", ["--dataset", "iris", "--multi-scalar"]),
    ("digits", ["--dataset", "digits", "--n-points", "400", "--multi-scalar"]),
    # --- high-dim ---
    ("mnist", ["--dataset", "mnist", "--n-points", "1500", "--pre-pca", "50", "--multi-scalar"]),
    (
        "cifar10",
        ["--dataset", "cifar10", "--n-points", "1000", "--pre-pca", "40", "--multi-scalar"],
    ),
]


def run_one(name: str, extra: list[str]) -> bool:
    out = OUT_DIR / f"{name}_voxel.png"
    cmd = [PYTHON, str(SCRIPT)] + extra + ["--off-screen", "--out", str(out)]
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
        return False
    if out.exists():
        print(f"  -> {out}")
        return True
    print(f"  WARNING: output file not found: {out}")
    return False


if __name__ == "__main__":
    successes, failures = [], []
    for name, extra in DATASETS:
        if run_one(name, extra):
            successes.append(name)
        else:
            failures.append(name)

    print(f"\n{'=' * 60}")
    print(f"Done.  {len(successes)}/{len(DATASETS)} succeeded.")
    for s in successes:
        print(f"  ✓  {s}")
    for f in failures:
        print(f"  ✗  {f}")
