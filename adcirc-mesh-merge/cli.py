#!/usr/bin/env python3
"""CLI: merge two ADCIRC fort.14 meshes across open boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from merge import merge_fort14


def main():
    p = argparse.ArgumentParser(
        description="Merge two non-overlapping ADCIRC meshes by bridging open boundaries."
    )
    p.add_argument("mesh_a", type=Path, help="First fort.14 / .grd file")
    p.add_argument("mesh_b", type=Path, help="Second fort.14 / .grd file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged.14"),
        help="Output fort.14 path (default: merged.14)",
    )
    p.add_argument("--open-a", type=int, default=0, help="Open-boundary index in mesh A")
    p.add_argument("--open-b", type=int, default=0, help="Open-boundary index in mesh B")
    p.add_argument(
        "--max-growth",
        type=float,
        default=1.35,
        help="Max size ratio per cross-gap layer (default: 1.35)",
    )
    args = p.parse_args()

    merged = merge_fort14(
        args.mesh_a,
        args.mesh_b,
        args.output,
        open_index_a=args.open_a,
        open_index_b=args.open_b,
        max_size_ratio_per_layer=args.max_growth,
    )
    print(
        f"Wrote {args.output} with {merged.n_nodes} nodes and {merged.n_elements} elements"
    )


if __name__ == "__main__":
    main()
