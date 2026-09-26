"""Size-aware bridging mesh between two open-boundary polylines."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d


def _arc_length_param(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc-length parameter normalized to [0, 1]."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total <= 0:
        return np.linspace(0.0, 1.0, len(xy))
    return s / total


def _polyline_interpolators(xy: np.ndarray, values: Optional[np.ndarray] = None):
    """Build 1D interpolators along a polyline vs normalized arc length."""
    u = _arc_length_param(xy)
    # Enforce strictly increasing u for interp1d (duplicate points get tiny eps)
    for i in range(1, len(u)):
        if u[i] <= u[i - 1]:
            u[i] = u[i - 1] + 1e-12
    u = u / u[-1]
    fx = interp1d(u, xy[:, 0], kind="linear")
    fy = interp1d(u, xy[:, 1], kind="linear")
    fd = None
    if values is not None:
        fd = interp1d(u, values, kind="linear")
    return fx, fy, fd, u


def geometric_node_count(length: float, h0: float, h1: float) -> Tuple[int, float]:
    """
    Number of segments and growth rate for a geometric size transition.

    Solves for integer n >= 1 such that the geometric series of edge lengths
    with start h0 and end ~h1 spans approximately `length`.
    """
    h0 = max(float(h0), 1e-12)
    h1 = max(float(h1), 1e-12)
    length = max(float(length), h0)

    if abs(np.log(h1 / h0)) < 1e-8:
        n = max(1, int(np.ceil(length / h0)))
        return n, 1.0

    # Ideal continuous n from L = h0 (r^n - 1)/(r - 1), r = (h1/h0)^(1/(n-1))
    # Search small integer n.
    best_n = 1
    best_err = abs(length - h0)
    best_r = 1.0
    max_n = max(2, int(np.ceil(length / min(h0, h1))) + 5)
    max_n = min(max_n, 500)
    for n in range(1, max_n + 1):
        if n == 1:
            r = 1.0
            span = h0
        else:
            r = (h1 / h0) ** (1.0 / (n - 1))
            if abs(r - 1.0) < 1e-12:
                span = n * h0
            else:
                span = h0 * (r**n - 1.0) / (r - 1.0)
        err = abs(span - length)
        # Prefer slight overshoot so we don't stretch last edges too much
        if span >= 0.85 * length and err < best_err:
            best_err = err
            best_n = n
            best_r = r
    return best_n, best_r


def geometric_stations(length: float, h0: float, h1: float) -> np.ndarray:
    """
    Normalized stations t in [0, 1] along a connector of length `length`
    with geometric grading from h0 to h1.
    """
    n, r = geometric_node_count(length, h0, h1)
    if n <= 1:
        return np.array([0.0, 1.0])
    h0 = max(float(h0), 1e-12)
    edges = h0 * r ** np.arange(n)
    # Rescale so sum(edges) == length exactly
    edges = edges * (length / edges.sum())
    t = np.concatenate([[0.0], np.cumsum(edges)])
    t = t / t[-1]
    return t


def blend_size(h_a: float, h_b: float, t: float) -> float:
    """Geometric (multiplicative) size blend — smooth across large ratios."""
    h_a = max(h_a, 1e-12)
    h_b = max(h_b, 1e-12)
    t = float(np.clip(t, 0.0, 1.0))
    return float(h_a ** (1.0 - t) * h_b**t)


def _cross_stations_for_gap(
    length: float,
    h_a: float,
    h_b: float,
    global_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cross-gap stations. If global_t is given, reuse that topology (same #layers)
    but still grade locally via mapping through a geometric cumulative curve.
    """
    local = geometric_stations(length, h_a, h_b)
    if global_t is None:
        return local
    # Map uniform layer index through local geometric cumulative distribution
    # so every column shares the same number of layers.
    # Build CDF of local edge fractions vs index.
    n = len(global_t) - 1
    if n <= 0:
        return np.array([0.0, 1.0])
    # Resample local stations onto n segments
    # Use local geometric spacing renormalized to n segments
    n_loc, r = geometric_node_count(length, h_a, h_b)
    h0 = max(h_a, 1e-12)
    if n == 1:
        return np.array([0.0, 1.0])
    if abs(r - 1.0) < 1e-12 or n_loc <= 1:
        edges = np.ones(n)
    else:
        # Match end ratio over n segments
        r_n = (max(h_b, 1e-12) / h0) ** (1.0 / n)
        edges = h0 * r_n ** np.arange(n)
    edges = edges / edges.sum()
    return np.concatenate([[0.0], np.cumsum(edges)])


def _tri_aspect(xy: np.ndarray, i: int, j: int, k: int) -> float:
    p = xy[[i, j, k]]
    e = [
        np.linalg.norm(p[0] - p[1]),
        np.linalg.norm(p[1] - p[2]),
        np.linalg.norm(p[2] - p[0]),
    ]
    return max(e) / max(min(e), 1e-15)


def triangulate_strip_xy(
    xy: np.ndarray, row_a: Sequence[int], row_b: Sequence[int]
) -> np.ndarray:
    """Greedy aspect-aware triangulation between two node rows."""
    a = list(row_a)
    b = list(row_b)
    i = j = 0
    tris: List[List[int]] = []
    while i < len(a) - 1 or j < len(b) - 1:
        if i == len(a) - 1:
            tris.append([a[i], b[j], b[j + 1]])
            j += 1
        elif j == len(b) - 1:
            tris.append([a[i], a[i + 1], b[j]])
            i += 1
        else:
            # Candidate A: advance on row a  -> triangle (a[i], a[i+1], b[j])
            # Candidate B: advance on row b  -> triangle (a[i], b[j], b[j+1])
            asp_a = _tri_aspect(xy, a[i], a[i + 1], b[j])
            asp_b = _tri_aspect(xy, a[i], b[j], b[j + 1])
            # Also peek at the leftover triangle after each choice
            rem_a = _tri_aspect(xy, a[i + 1], b[j], b[j + 1]) if j < len(b) - 1 else asp_a
            rem_b = _tri_aspect(xy, a[i], a[i + 1], b[j + 1]) if i < len(a) - 1 else asp_b
            score_a = max(asp_a, rem_a)
            score_b = max(asp_b, rem_b)
            if score_a <= score_b:
                tris.append([a[i], a[i + 1], b[j]])
                i += 1
            else:
                tris.append([a[i], b[j], b[j + 1]])
                j += 1
    if not tris:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(tris, dtype=int)


def ensure_ccw(xy: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Flip triangle winding so area is positive (CCW in xy plane)."""
    out = tris.copy()
    for k, (i, j, m) in enumerate(out):
        area = 0.5 * (
            (xy[j, 0] - xy[i, 0]) * (xy[m, 1] - xy[i, 1])
            - (xy[m, 0] - xy[i, 0]) * (xy[j, 1] - xy[i, 1])
        )
        if area < 0:
            out[k] = [i, m, j]
    return out


def build_bridge(
    xy_a: np.ndarray,
    depth_a: np.ndarray,
    nodes_a: np.ndarray,
    size_a: np.ndarray,
    xy_b: np.ndarray,
    depth_b: np.ndarray,
    nodes_b: np.ndarray,
    size_b: np.ndarray,
    max_size_ratio_per_layer: float = 1.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a transition mesh between two open boundaries.

    Parameters
    ----------
    xy_a, depth_a : full mesh A coordinates/depths
    nodes_a : ordered open-boundary node indices into mesh A
    size_a : local size at each boundary node of A
    (same for B)
    max_size_ratio_per_layer : soft cap on geometric growth per cross-gap layer

    Returns
    -------
    bridge_xy : (Nb, 2) new + boundary nodes for the bridge (boundary nodes first
                as copies of A then interiors then copies of B — actually we return
                ONLY new interior nodes plus connectivity referencing a combined
                index space). Simpler: return all bridge region nodes and elements
                in a local 0-based index space, plus maps of which local nodes are
                the A-boundary and B-boundary (for stitching).

    Actually returns:
      all_xy, all_depth, elements, idx_a, idx_b
    where idx_a / idx_b are local indices of the open-boundary nodes matching
    nodes_a / nodes_b order (for later replacement with global mesh node ids).
    """
    poly_a = xy_a[nodes_a]
    poly_b = xy_b[nodes_b]
    dep_a = depth_a[nodes_a]
    dep_b = depth_b[nodes_b]

    # Interpolators on both boundaries
    fx_a, fy_a, fs_a, _ = _polyline_interpolators(poly_a, size_a)
    _, _, fd_a, _ = _polyline_interpolators(poly_a, dep_a)
    fx_b, fy_b, fs_b, _ = _polyline_interpolators(poly_b, size_b)
    _, _, fd_b, _ = _polyline_interpolators(poly_b, dep_b)

    # Representative gap length and sizes for global layer count
    n_probe = 41
    u_probe = np.linspace(0.0, 1.0, n_probe)
    gap_lens = []
    h0s, h1s = [], []
    for u in u_probe:
        pa = np.array([float(fx_a(u)), float(fy_a(u))])
        pb = np.array([float(fx_b(u)), float(fy_b(u))])
        gap_lens.append(np.linalg.norm(pb - pa))
        h0s.append(float(fs_a(u)))
        h1s.append(float(fs_b(u)))
    L_med = float(np.median(gap_lens))
    h0_med = float(np.median(h0s))
    h1_med = float(np.median(h1s))

    # Cap growth rate per layer for visual smoothness
    n_layers, r = geometric_node_count(L_med, h0_med, h1_med)
    if r > max_size_ratio_per_layer and n_layers > 1:
        # Increase layers until growth rate is acceptable
        ratio = max(h1_med, h0_med) / min(h1_med, h0_med)
        n_needed = int(np.ceil(np.log(max(ratio, 1.0)) / np.log(max_size_ratio_per_layer))) + 1
        n_layers = max(n_layers, n_needed)
    n_layers = max(n_layers, 1)

    # Global cross-gap stations (will be regraded per column)
    # Use a reference geometric progression from median sizes
    t_ref = geometric_stations(L_med, h0_med, h1_med)
    # Force exactly n_layers segments
    if len(t_ref) - 1 != n_layers:
        # rebuild with forced n
        h0 = h0_med
        h1 = h1_med
        if n_layers == 1:
            t_ref = np.array([0.0, 1.0])
        else:
            r_n = (max(h1, 1e-12) / max(h0, 1e-12)) ** (1.0 / n_layers)
            # also soften if still too aggressive
            r_n = min(r_n, max_size_ratio_per_layer) if h1 >= h0 else max(
                r_n, 1.0 / max_size_ratio_per_layer
            )
            edges = h0 * (r_n ** np.arange(n_layers))
            edges = edges / edges.sum()
            t_ref = np.concatenate([[0.0], np.cumsum(edges)])

    # Build rows: row 0 / last snap to real boundary nodes.
    # Intermediate rows gradually coarsen node count from n_A -> n_B while
    # spacing follows the blended size field (avoids slivers at the ends).
    n_a = len(poly_a)
    n_b = len(poly_b)
    u_a = _arc_length_param(poly_a)
    u_b = _arc_length_param(poly_b)

    rows_xy: List[np.ndarray] = []
    rows_dp: List[np.ndarray] = []

    rows_xy.append(poly_a.copy())
    rows_dp.append(dep_a.copy())

    for k in range(1, n_layers):
        t_nom = float(t_ref[k])
        # Progressive node count (geometric in count space)
        n_k = int(round(n_a ** (1.0 - t_nom) * n_b**t_nom))
        n_k = int(np.clip(n_k, min(n_a, n_b), max(n_a, n_b)))
        n_k = max(n_k, 2)

        # Seed u with both boundary parameters (scaled toward current t) plus
        # a size-aware adaptive walk so local spacing stays consistent.
        u_seed = np.unique(
            np.concatenate(
                [
                    u_a,
                    u_b,
                    _adaptive_u_stations(fs_a, fs_b, t_nom, poly_a, poly_b),
                ]
            )
        )
        # Downsample / redistribute to ~n_k points while keeping endpoints
        if len(u_seed) > n_k:
            # Keep endpoints; pick remaining by arc-length on the blended curve
            fx_a2, fy_a2, _, _ = _polyline_interpolators(poly_a)
            fx_b2, fy_b2, _, _ = _polyline_interpolators(poly_b)
            blend_pts = np.column_stack(
                [
                    (1.0 - t_nom) * fx_a2(u_seed) + t_nom * fx_b2(u_seed),
                    (1.0 - t_nom) * fy_a2(u_seed) + t_nom * fy_b2(u_seed),
                ]
            )
            s = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(blend_pts, axis=0), axis=1))]
            )
            s = s / max(s[-1], 1e-12)
            targets = np.linspace(0.0, 1.0, n_k)
            u_dense = np.interp(targets, s, u_seed)
            u_dense[0], u_dense[-1] = 0.0, 1.0
        else:
            u_dense = u_seed

        pts = []
        dps = []
        for u in u_dense:
            pa = np.array([float(fx_a(u)), float(fy_a(u))])
            pb = np.array([float(fx_b(u)), float(fy_b(u))])
            L = float(np.linalg.norm(pb - pa))
            ha, hb = float(fs_a(u)), float(fs_b(u))
            t_col = _cross_stations_for_gap(L, ha, hb, global_t=t_ref)
            tk = float(t_col[k])
            p = (1.0 - tk) * pa + tk * pb
            d = (1.0 - tk) * float(fd_a(u)) + tk * float(fd_b(u))
            pts.append(p)
            dps.append(d)
        rows_xy.append(np.asarray(pts, dtype=float))
        rows_dp.append(np.asarray(dps, dtype=float))

    rows_xy.append(poly_b.copy())
    rows_dp.append(dep_b.copy())

    # Pack nodes
    all_xy = []
    all_dp = []
    row_ids: List[List[int]] = []
    for rxy, rdp in zip(rows_xy, rows_dp):
        ids = []
        for p, d in zip(rxy, rdp):
            ids.append(len(all_xy))
            all_xy.append(p)
            all_dp.append(d)
        row_ids.append(ids)

    all_xy_arr = np.asarray(all_xy, dtype=float)
    all_dp_arr = np.asarray(all_dp, dtype=float)

    tris = []
    for r in range(len(row_ids) - 1):
        strip = triangulate_strip_xy(all_xy_arr, row_ids[r], row_ids[r + 1])
        if len(strip):
            tris.append(strip)
    elements = np.vstack(tris) if tris else np.zeros((0, 3), dtype=int)
    elements = ensure_ccw(all_xy_arr, elements)

    idx_a = np.asarray(row_ids[0], dtype=int)
    idx_b = np.asarray(row_ids[-1], dtype=int)
    return all_xy_arr, all_dp_arr, elements, idx_a, idx_b


def _adaptive_u_stations(fs_a, fs_b, t: float, poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """
    Parametric stations along a blended row, denser where blended size is small.
    """
    # Estimate total row length via many probes
    n_probe = 80
    u_p = np.linspace(0.0, 1.0, n_probe)
    # Approximate blended curve length using linear blend of boundaries
    fx_a, fy_a, _, _ = _polyline_interpolators(poly_a)
    fx_b, fy_b, _, _ = _polyline_interpolators(poly_b)
    pts = []
    hs = []
    for u in u_p:
        pa = np.array([float(fx_a(u)), float(fy_a(u))])
        pb = np.array([float(fx_b(u)), float(fy_b(u))])
        # Use same t for length estimate (good enough for sampling density)
        pts.append((1.0 - t) * pa + t * pb)
        hs.append(blend_size(float(fs_a(u)), float(fs_b(u)), t))
    pts = np.asarray(pts)
    hs = np.asarray(hs)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # Walk accumulating arc length, place samples every local h
    u_list = [0.0]
    acc = 0.0
    target = hs[0]
    for i, ds in enumerate(seg):
        acc += ds
        h_here = 0.5 * (hs[i] + hs[i + 1])
        while acc >= target and u_list[-1] < 1.0:
            # fraction within this segment
            overshoot = acc - target
            frac = 1.0 - overshoot / max(ds, 1e-12)
            u_new = u_p[i] + frac * (u_p[i + 1] - u_p[i])
            u_new = min(max(u_new, u_list[-1] + 1e-6), 1.0)
            u_list.append(u_new)
            target = h_here  # next spacing
            acc = overshoot
            if u_new >= 1.0:
                break
    if u_list[-1] < 1.0 - 1e-9:
        u_list.append(1.0)
    return np.unique(np.asarray(u_list, dtype=float))
