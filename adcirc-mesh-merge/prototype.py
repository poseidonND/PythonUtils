#!/usr/bin/env python3
"""Prototype: synthesize two meshes with different resolution, merge, plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from adcirc_io import AdcircMesh, Boundary, write_fort14
from merge import merge_meshes


def _structured_rect_mesh(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    nx: int,
    ny: int,
    depth0: float,
    depth1: float,
    title: str,
    open_side: str = "right",
) -> AdcircMesh:
    """
    Build a rectangular triangulated mesh with one open boundary on `open_side`.

    Element size is roughly uniform within the patch (set by nx, ny).
    """
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    # Depth varies left-right for visual interest
    t = (xy[:, 0] - x0) / max(x1 - x0, 1e-12)
    depth = depth0 * (1 - t) + depth1 * t

    def vid(i, j):
        return j * nx + i

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = vid(i, j)
            n10 = vid(i + 1, j)
            n01 = vid(i, j + 1)
            n11 = vid(i + 1, j + 1)
            # Alternate diagonal for slightly less structured look
            if (i + j) % 2 == 0:
                elems.append([n00, n10, n11])
                elems.append([n00, n11, n01])
            else:
                elems.append([n00, n10, n01])
                elems.append([n10, n11, n01])
    elements = np.asarray(elems, dtype=int)

    # Boundaries
    bottom = np.array([vid(i, 0) for i in range(nx)], dtype=int)
    top = np.array([vid(i, ny - 1) for i in range(nx)], dtype=int)
    left = np.array([vid(0, j) for j in range(ny)], dtype=int)
    right = np.array([vid(nx - 1, j) for j in range(ny)], dtype=int)

    if open_side == "right":
        open_nodes = right
        land = [
            Boundary(nodes=bottom, ibtype=0),
            Boundary(nodes=np.concatenate([right[-1:], top[::-1], left[::-1][1:]]), ibtype=0),
            # simpler: three land sides
        ]
        # Cleaner land loop excluding open side: bottom + left + top
        land = [
            Boundary(
                nodes=np.concatenate([bottom, left[-2::-1], top]),
                ibtype=0,
            )
        ]
    elif open_side == "left":
        open_nodes = left[::-1]  # top-to-bottom or bottom-to-top; orient later
        # Match right-open mesh which goes bottom->top on right side
        open_nodes = left  # bottom -> top
        land = [
            Boundary(
                nodes=np.concatenate([bottom[::-1], right, top[::-1]]),
                ibtype=0,
            )
        ]
    else:
        raise ValueError(open_side)

    return AdcircMesh(
        title=title,
        xy=xy,
        depth=depth,
        elements=elements,
        open_boundaries=[Boundary(nodes=open_nodes, ibtype=0)],
        land_boundaries=land,
    )


def make_prototype_meshes() -> tuple[AdcircMesh, AdcircMesh]:
    """
    Fine mesh on the left (small elements), coarse mesh on the right (large),
    with a gap between their open boundaries.
    """
    # Fine coastal-like patch
    mesh_a = _structured_rect_mesh(
        x0=0.0,
        x1=4.0,
        y0=0.0,
        y1=6.0,
        nx=25,  # small elements
        ny=37,
        depth0=5.0,
        depth1=20.0,
        title="Fine left mesh",
        open_side="right",
    )
    # Coarse ocean-like patch, separated by a gap
    mesh_b = _structured_rect_mesh(
        x0=7.5,
        x1=16.0,
        y0=-1.0,
        y1=7.0,
        nx=8,  # large elements
        ny=7,
        depth0=25.0,
        depth1=80.0,
        title="Coarse right mesh",
        open_side="left",
    )
    return mesh_a, mesh_b


def plot_mesh(
    ax,
    mesh: AdcircMesh,
    edgecolor: str = "0.25",
    linewidth: float = 0.4,
    facecolor=None,
    alpha: float = 0.85,
    open_color: str = "C3",
    land_color: str = "0.4",
):
    polys = mesh.xy[mesh.elements]
    if facecolor is None:
        # Color by mean element depth
        depths = mesh.depth[mesh.elements].mean(axis=1)
        coll = PolyCollection(
            polys,
            array=depths,
            cmap="Blues",
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(coll)
    else:
        coll = PolyCollection(
            polys,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(coll)

    for b in mesh.open_boundaries:
        p = mesh.xy[b.nodes]
        ax.plot(p[:, 0], p[:, 1], color=open_color, lw=2.0, solid_capstyle="round")
    for b in mesh.land_boundaries:
        p = mesh.xy[b.nodes]
        ax.plot(p[:, 0], p[:, 1], color=land_color, lw=1.2)

    return coll


def plot_size_field(ax, mesh: AdcircMesh):
    """Approximate local size as mean edge length of each triangle."""
    tris = mesh.elements
    xy = mesh.xy
    sizes = []
    for t in tris:
        e = [
            np.linalg.norm(xy[t[0]] - xy[t[1]]),
            np.linalg.norm(xy[t[1]] - xy[t[2]]),
            np.linalg.norm(xy[t[2]] - xy[t[0]]),
        ]
        sizes.append(np.mean(e))
    sizes = np.asarray(sizes)
    coll = PolyCollection(
        xy[tris],
        array=sizes,
        cmap="viridis",
        edgecolors="none",
        linewidths=0,
        alpha=0.95,
    )
    ax.add_collection(coll)
    return coll, sizes


def main():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)

    mesh_a, mesh_b = make_prototype_meshes()
    write_fort14(mesh_a, out / "mesh_a.14")
    write_fort14(mesh_b, out / "mesh_b.14")

    merged = merge_meshes(
        mesh_a,
        mesh_b,
        title="Merged prototype mesh",
        max_size_ratio_per_layer=1.30,
    )
    write_fort14(merged, out / "merged.14")

    print(
        f"Mesh A: {mesh_a.n_nodes} nodes, {mesh_a.n_elements} elements\n"
        f"Mesh B: {mesh_b.n_nodes} nodes, {mesh_b.n_elements} elements\n"
        f"Merged: {merged.n_nodes} nodes, {merged.n_elements} elements\n"
        f"Wrote: {out / 'merged.14'}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)

    # Panel 1: inputs
    ax = axes[0]
    plot_mesh(ax, mesh_a, edgecolor="0.2", linewidth=0.35)
    plot_mesh(ax, mesh_b, edgecolor="0.2", linewidth=0.35)
    ax.set_title("Input meshes (open BCs in red)")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Panel 2: merged wireframe
    ax = axes[1]
    plot_mesh(ax, merged, edgecolor="0.15", linewidth=0.3)
    ax.set_title("Merged mesh with size-aware bridge")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x")

    # Panel 3: element size
    ax = axes[2]
    coll, sizes = plot_size_field(ax, merged)
    # Overlay edges lightly
    segs = []
    for t in merged.elements:
        pts = merged.xy[list(t) + [t[0]]]
        segs.append(pts)
    ax.add_collection(
        LineCollection(segs, colors="0.2", linewidths=0.15, alpha=0.35)
    )
    cb = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean edge length")
    ax.set_title("Local element size")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x")

    fig.suptitle(
        "ADCIRC open-boundary merge — smooth small→large transition",
        fontsize=12,
    )
    fig_path = out / "prototype_merge.png"
    fig.savefig(fig_path, dpi=160)
    print(f"Plot: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
