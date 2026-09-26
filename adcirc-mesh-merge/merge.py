"""Merge two non-overlapping ADCIRC meshes across open boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from adcirc_io import (
    AdcircMesh,
    Boundary,
    boundary_edge_lengths,
    orient_boundaries,
    read_fort14,
    write_fort14,
)
from bridge import build_bridge

PathLike = Union[str, Path]


def _offset_boundaries(boundaries, offset: int):
    return [
        Boundary(nodes=b.nodes + offset, ibtype=b.ibtype) for b in boundaries
    ]


def merge_meshes(
    mesh_a: AdcircMesh,
    mesh_b: AdcircMesh,
    open_index_a: int = 0,
    open_index_b: int = 0,
    title: Optional[str] = None,
    max_size_ratio_per_layer: float = 1.35,
) -> AdcircMesh:
    """
    Merge two meshes by filling the gap between one open boundary of each.

    The chosen open boundaries are consumed (replaced by the bridge). All other
    open/land boundaries are preserved with remapped node indices.
    """
    if not mesh_a.open_boundaries:
        raise ValueError("mesh_a has no open boundaries")
    if not mesh_b.open_boundaries:
        raise ValueError("mesh_b has no open boundaries")

    nodes_a = mesh_a.open_boundaries[open_index_a].nodes.copy()
    nodes_b = mesh_b.open_boundaries[open_index_b].nodes.copy()
    nodes_a, nodes_b = orient_boundaries(mesh_a.xy, nodes_a, mesh_b.xy, nodes_b)

    size_a = boundary_edge_lengths(mesh_a.xy, nodes_a)
    size_b = boundary_edge_lengths(mesh_b.xy, nodes_b)

    bridge_xy, bridge_dp, bridge_els, idx_a, idx_b = build_bridge(
        mesh_a.xy,
        mesh_a.depth,
        nodes_a,
        size_a,
        mesh_b.xy,
        mesh_b.depth,
        nodes_b,
        size_b,
        max_size_ratio_per_layer=max_size_ratio_per_layer,
    )

    # Interior bridge nodes only (exclude boundary rows that already exist)
    interior_mask = np.ones(len(bridge_xy), dtype=bool)
    interior_mask[idx_a] = False
    interior_mask[idx_b] = False
    interior_ids = np.where(interior_mask)[0]

    n_a = mesh_a.n_nodes
    n_b = mesh_b.n_nodes
    n_int = len(interior_ids)

    # Global layout: [mesh A nodes | mesh B nodes | bridge interior]
    xy = np.vstack([mesh_a.xy, mesh_b.xy, bridge_xy[interior_ids]])
    depth = np.concatenate([mesh_a.depth, mesh_b.depth, bridge_dp[interior_ids]])

    # Map local bridge index -> global
    local_to_global = np.empty(len(bridge_xy), dtype=int)
    for i, la in enumerate(idx_a):
        local_to_global[la] = int(nodes_a[i])  # still in mesh A numbering
    for i, lb in enumerate(idx_b):
        local_to_global[lb] = int(nodes_b[i]) + n_a
    for k, li in enumerate(interior_ids):
        local_to_global[li] = n_a + n_b + k

    bridge_els_g = local_to_global[bridge_els]

    elems_a = mesh_a.elements.copy()
    elems_b = mesh_b.elements + n_a
    elements = np.vstack([elems_a, elems_b, bridge_els_g])

    # Boundaries: drop the connected open segments; keep the rest
    open_out = []
    for i, b in enumerate(mesh_a.open_boundaries):
        if i == open_index_a:
            continue
        open_out.append(Boundary(nodes=b.nodes.copy(), ibtype=b.ibtype))
    for i, b in enumerate(mesh_b.open_boundaries):
        if i == open_index_b:
            continue
        open_out.append(Boundary(nodes=b.nodes + n_a, ibtype=b.ibtype))

    land_out = _offset_boundaries(mesh_a.land_boundaries, 0)
    land_out.extend(_offset_boundaries(mesh_b.land_boundaries, n_a))

    # Side connectors of the gap become new land (or open) boundaries:
    # endpoints of the two open boundaries form the lateral edges.
    # Add them as land boundaries (ibtype=0) so the merged domain is closed
    # along the stitch sides if those edges are not already interior.
    # Only add if the lateral edges have positive length (always true for gap).
    # The lateral edges are already covered by bridge triangles, so they are
    # interior to the merged mesh — do NOT add as boundaries.

    merged = AdcircMesh(
        title=title or f"Merged: {mesh_a.title} + {mesh_b.title}",
        xy=xy,
        depth=depth,
        elements=elements,
        open_boundaries=open_out,
        land_boundaries=land_out,
    )
    return merged


def merge_fort14(
    path_a: PathLike,
    path_b: PathLike,
    path_out: PathLike,
    open_index_a: int = 0,
    open_index_b: int = 0,
    **kwargs,
) -> AdcircMesh:
    """Convenience: read two fort.14 files, merge, write result."""
    mesh_a = read_fort14(path_a)
    mesh_b = read_fort14(path_b)
    merged = merge_meshes(
        mesh_a, mesh_b, open_index_a=open_index_a, open_index_b=open_index_b, **kwargs
    )
    write_fort14(merged, path_out)
    return merged
