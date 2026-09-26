"""Read and write ADCIRC fort.14-style unstructured meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class Boundary:
    """One ADCIRC boundary segment (open or land)."""

    nodes: np.ndarray  # 0-based node indices along the boundary
    ibtype: int = 0


@dataclass
class AdcircMesh:
    """Minimal ADCIRC mesh: nodes, triangles, open/land boundaries."""

    title: str
    xy: np.ndarray  # (NP, 2)
    depth: np.ndarray  # (NP,)
    elements: np.ndarray  # (NE, 3), 0-based
    open_boundaries: List[Boundary] = field(default_factory=list)
    land_boundaries: List[Boundary] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return int(self.xy.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    def copy(self) -> "AdcircMesh":
        return AdcircMesh(
            title=self.title,
            xy=self.xy.copy(),
            depth=self.depth.copy(),
            elements=self.elements.copy(),
            open_boundaries=[
                Boundary(nodes=b.nodes.copy(), ibtype=b.ibtype) for b in self.open_boundaries
            ],
            land_boundaries=[
                Boundary(nodes=b.nodes.copy(), ibtype=b.ibtype) for b in self.land_boundaries
            ],
        )


def _parse_ints(line: str) -> List[int]:
    return [int(float(tok)) for tok in line.split()]


def read_fort14(path: Union[str, Path]) -> AdcircMesh:
    """Parse an ADCIRC fort.14 / .grd file."""
    path = Path(path)
    lines = path.read_text().splitlines()
    i = 0
    title = lines[i].strip()
    i += 1
    ne, np_ = _parse_ints(lines[i])[:2]
    i += 1

    xy = np.zeros((np_, 2), dtype=float)
    depth = np.zeros(np_, dtype=float)
    for _ in range(np_):
        parts = lines[i].split()
        j = int(float(parts[0])) - 1
        xy[j, 0] = float(parts[1])
        xy[j, 1] = float(parts[2])
        depth[j] = float(parts[3]) if len(parts) > 3 else 0.0
        i += 1

    elements = np.zeros((ne, 3), dtype=int)
    for _ in range(ne):
        parts = _parse_ints(lines[i])
        j = parts[0] - 1
        # parts[1] is NHM (usually 3)
        elements[j] = np.array(parts[2:5], dtype=int) - 1
        i += 1

    open_boundaries: List[Boundary] = []
    land_boundaries: List[Boundary] = []

    if i < len(lines) and lines[i].strip():
        nope = _parse_ints(lines[i])[0]
        i += 1
        _neta = _parse_ints(lines[i])[0]
        i += 1
        for _ in range(nope):
            header = _parse_ints(lines[i])
            nvdll = header[0]
            ibtype = header[1] if len(header) > 1 else 0
            i += 1
            nodes = []
            for _n in range(nvdll):
                nodes.append(_parse_ints(lines[i])[0] - 1)
                i += 1
            open_boundaries.append(Boundary(nodes=np.asarray(nodes, dtype=int), ibtype=ibtype))

        if i < len(lines) and lines[i].strip():
            nbou = _parse_ints(lines[i])[0]
            i += 1
            _nvel = _parse_ints(lines[i])[0]
            i += 1
            for _ in range(nbou):
                header = _parse_ints(lines[i])
                nvell = header[0]
                ibtype = header[1] if len(header) > 1 else 0
                i += 1
                nodes = []
                for _n in range(nvell):
                    nodes.append(_parse_ints(lines[i])[0] - 1)
                    i += 1
                land_boundaries.append(
                    Boundary(nodes=np.asarray(nodes, dtype=int), ibtype=ibtype)
                )

    return AdcircMesh(
        title=title,
        xy=xy,
        depth=depth,
        elements=elements,
        open_boundaries=open_boundaries,
        land_boundaries=land_boundaries,
    )


def write_fort14(mesh: AdcircMesh, path: Union[str, Path]) -> None:
    """Write an ADCIRC fort.14-style file."""
    path = Path(path)
    lines: List[str] = []
    lines.append(mesh.title)
    lines.append(f"{mesh.n_elements} {mesh.n_nodes}")

    for j in range(mesh.n_nodes):
        x, y = mesh.xy[j]
        lines.append(f"{j + 1:8d} {x:16.8f} {y:16.8f} {mesh.depth[j]:16.8f}")

    for j in range(mesh.n_elements):
        n1, n2, n3 = mesh.elements[j] + 1
        lines.append(f"{j + 1:8d} 3 {n1:8d} {n2:8d} {n3:8d}")

    nope = len(mesh.open_boundaries)
    neta = int(sum(len(b.nodes) for b in mesh.open_boundaries))
    lines.append(f"{nope}")
    lines.append(f"{neta}")
    for b in mesh.open_boundaries:
        lines.append(f"{len(b.nodes)} {b.ibtype}")
        for n in b.nodes:
            lines.append(f"{int(n) + 1}")

    nbou = len(mesh.land_boundaries)
    nvel = int(sum(len(b.nodes) for b in mesh.land_boundaries))
    lines.append(f"{nbou}")
    lines.append(f"{nvel}")
    for b in mesh.land_boundaries:
        lines.append(f"{len(b.nodes)} {b.ibtype}")
        for n in b.nodes:
            lines.append(f"{int(n) + 1}")

    path.write_text("\n".join(lines) + "\n")


def boundary_edge_lengths(xy: np.ndarray, nodes: Sequence[int]) -> np.ndarray:
    """Per-node average adjacent boundary edge length (local size proxy)."""
    nodes = np.asarray(nodes, dtype=int)
    n = len(nodes)
    sizes = np.zeros(n, dtype=float)
    if n == 1:
        sizes[0] = 1.0
        return sizes
    for i in range(n):
        lengths = []
        if i > 0:
            lengths.append(np.linalg.norm(xy[nodes[i]] - xy[nodes[i - 1]]))
        if i < n - 1:
            lengths.append(np.linalg.norm(xy[nodes[i]] - xy[nodes[i + 1]]))
        sizes[i] = float(np.mean(lengths)) if lengths else 1.0
    return sizes


def orient_boundaries(
    xy_a: np.ndarray,
    nodes_a: np.ndarray,
    xy_b: np.ndarray,
    nodes_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip boundary B if needed so both polylines run in a compatible direction.

    Uses endpoint pairing: choose the orientation of B that minimizes the sum of
    squared distances between corresponding endpoints.
    """
    a0, a1 = xy_a[nodes_a[0]], xy_a[nodes_a[-1]]
    b0, b1 = xy_b[nodes_b[0]], xy_b[nodes_b[-1]]
    cost_same = np.sum((a0 - b0) ** 2) + np.sum((a1 - b1) ** 2)
    cost_flip = np.sum((a0 - b1) ** 2) + np.sum((a1 - b0) ** 2)
    if cost_flip < cost_same:
        return nodes_a, nodes_b[::-1].copy()
    return nodes_a, nodes_b
