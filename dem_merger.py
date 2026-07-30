"""
Merge DEM GeoTIFFs of mixed resolutions into one consistent mosaic.

Streams the mosaic chunk-by-chunk instead of building it in memory, so output
size is limited by disk rather than RAM. Every tile is warped onto one shared
grid (CRS + resolution + snapped origin) via WarpedVRT, then chunks are
composited in parallel and written directly to a BigTIFF.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds


SUPPORTED_EXTS = {".tif", ".tiff", ".geotiff"}

RESAMPLE_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}

METHODS = ("first", "last", "min", "max", "mean")


def find_geotiffs(directory: Path, exclude: set[Path] | None = None) -> list[Path]:
    exclude = {p.resolve() for p in (exclude or set())}
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and p.resolve() not in exclude
        and not p.name.startswith("warped_")
        and not p.name.startswith("merged_")
    )
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files found in: {directory}")
    return files


def inspect_sources(paths: list[Path]) -> list[dict]:
    infos = []
    for path in paths:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise ValueError(f"{path.name} has no bands")
            res_x, res_y = src.res
            infos.append(
                {
                    "path": path,
                    "crs": src.crs,
                    "res_x": abs(float(res_x)),
                    "res_y": abs(float(res_y)),
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                    "bounds": src.bounds,
                    "count": src.count,
                }
            )
    return infos


def choose_target_crs(infos: list[dict], crs_override: str | None):
    if crs_override:
        from rasterio.crs import CRS

        return CRS.from_user_input(crs_override)

    crs_counts: dict[str, int] = {}
    crs_by_key: dict[str, object] = {}
    for info in infos:
        if info["crs"] is None:
            raise ValueError(
                f"{info['path'].name} has no CRS. Assign one or pass --crs."
            )
        epsg = info["crs"].to_epsg()
        key = f"EPSG:{epsg}" if epsg else info["crs"].to_wkt()
        crs_counts[key] = crs_counts.get(key, 0) + 1
        crs_by_key[key] = info["crs"]

    best = max(crs_counts, key=crs_counts.get)
    return crs_by_key[best]


def choose_target_resolution(
    infos: list[dict],
    resolution: float | None,
    prefer: str,
) -> float:
    if resolution is not None:
        if resolution <= 0:
            raise ValueError("--resolution must be > 0")
        return float(resolution)

    res_values = [min(i["res_x"], i["res_y"]) for i in infos]
    if prefer == "finest":
        return float(min(res_values))
    if prefer == "coarsest":
        return float(max(res_values))
    raise ValueError("prefer must be 'finest' or 'coarsest'")


def choose_nodata(infos: list[dict], nodata_override: float | None) -> float:
    if nodata_override is not None:
        return float(nodata_override)

    values = [i["nodata"] for i in infos if i["nodata"] is not None]
    if values:
        first = float(values[0])
        if any(float(v) != first for v in values[1:]):
            print(
                f"Warning: mixed nodata values {values}; using {first}. "
                "Pass --nodata to override.",
                file=sys.stderr,
            )
        return first
    return -9999.0


def union_bounds_in_crs(infos: list[dict], dst_crs):
    left = bottom = right = top = None
    for info in infos:
        b = transform_bounds(info["crs"], dst_crs, *info["bounds"], densify_pts=21)
        if left is None:
            left, bottom, right, top = b
        else:
            left = min(left, b[0])
            bottom = min(bottom, b[1])
            right = max(right, b[2])
            top = max(top, b[3])
    return left, bottom, right, top


def build_output_grid(bounds, resolution: float):
    """Snap mosaic bounds outward to a regular resolution grid."""
    left, bottom, right, top = bounds
    left = math.floor(left / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution

    width = max(int(round((right - left) / resolution)), 1)
    height = max(int(round((top - bottom) / resolution)), 1)
    transform = from_origin(left, top, resolution, resolution)
    return transform, width, height, (left, bottom, right, top)


def source_window(info: dict, dst_crs, transform, width: int, height: int):
    """Footprint of one source as a clamped window on the output grid."""
    b = transform_bounds(info["crs"], dst_crs, *info["bounds"], densify_pts=21)
    win = window_from_bounds(*b, transform=transform)

    col_off = math.floor(win.col_off)
    row_off = math.floor(win.row_off)
    col_end = math.ceil(win.col_off + win.width)
    row_end = math.ceil(win.row_off + win.height)

    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(width, col_end)
    row_end = min(height, row_end)
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def windows_overlap(a: Window, b: Window) -> bool:
    return not (
        a.col_off + a.width <= b.col_off
        or b.col_off + b.width <= a.col_off
        or a.row_off + a.height <= b.row_off
        or b.row_off + b.height <= a.row_off
    )


class SourcePool:
    """Per-thread WarpedVRTs aligned to the output grid.

    GDAL dataset handles are not safe to share across threads, so each worker
    opens its own set on first use.
    """

    def __init__(self, infos, dst_crs, transform, width, height, nodata, resampling):
        self.infos = infos
        self.dst_crs = dst_crs
        self.transform = transform
        self.width = width
        self.height = height
        self.nodata = nodata
        self.resampling = resampling
        self._local = threading.local()
        self._opened = []
        self._lock = threading.Lock()

    def entries(self):
        entries = getattr(self._local, "entries", None)
        if entries is not None:
            return entries

        entries = []
        opened = []
        for info in self.infos:
            if info["window"] is None:
                continue
            src = rasterio.open(info["path"])
            vrt = WarpedVRT(
                src,
                crs=self.dst_crs,
                transform=self.transform,
                width=self.width,
                height=self.height,
                nodata=self.nodata,
                src_nodata=(
                    info["nodata"] if info["nodata"] is not None else self.nodata
                ),
                resampling=self.resampling,
                dtype="float32",
            )
            entries.append((info["window"], vrt))
            opened.append((src, vrt))

        with self._lock:
            self._opened.extend(opened)
        self._local.entries = entries
        return entries

    def close(self):
        with self._lock:
            for src, vrt in self._opened:
                try:
                    vrt.close()
                except Exception:
                    pass
                try:
                    src.close()
                except Exception:
                    pass
            self._opened.clear()


def chunk_windows(width: int, height: int, chunk: int):
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off, row_off, w, h)


def composite_chunk(window: Window, pool: SourcePool, nodata: float, method: str):
    """Read every overlapping source for one window and combine them."""
    shape = (int(window.height), int(window.width))
    out = np.full(shape, nodata, dtype="float32")
    filled = np.zeros(shape, dtype=bool)

    if method == "mean":
        total = np.zeros(shape, dtype="float64")
        count = np.zeros(shape, dtype="int32")
    else:
        total = count = None

    for src_window, vrt in pool.entries():
        if not windows_overlap(src_window, window):
            continue
        if method == "first" and filled.all():
            break

        data = vrt.read(1, window=window)
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= ~np.isclose(data, nodata)
        if not valid.any():
            continue

        if method == "mean":
            total[valid] += data[valid]
            count[valid] += 1
            filled |= valid
        elif method == "first":
            fresh = valid & ~filled
            out[fresh] = data[fresh]
            filled |= fresh
        elif method == "last":
            out[valid] = data[valid]
            filled |= valid
        elif method == "min":
            fresh = valid & ~filled
            out[fresh] = data[fresh]
            both = valid & filled
            if both.any():
                out[both] = np.minimum(out[both], data[both])
            filled |= valid
        elif method == "max":
            fresh = valid & ~filled
            out[fresh] = data[fresh]
            both = valid & filled
            if both.any():
                out[both] = np.maximum(out[both], data[both])
            filled |= valid

    if method == "mean":
        has = count > 0
        out[has] = (total[has] / count[has]).astype("float32")

    n_valid = int(np.count_nonzero(filled))
    if n_valid:
        values = out[filled]
        vmin = float(values.min())
        vmax = float(values.max())
    else:
        vmin = vmax = None
    return out, n_valid, vmin, vmax


def _fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_dems(
    input_dir: Path,
    output_path: Path,
    resolution: float | None = None,
    prefer_resolution: str = "finest",
    crs: str | None = None,
    resampling_name: str = "bilinear",
    method: str = "first",
    nodata: float | None = None,
    chunk_size: int = 2048,
    workers: int | None = None,
    compress: str = "lzw",
    gdal_cache_mb: int = 1024,
    overviews: bool = False,
) -> Path:
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    workers = workers or min(8, (os.cpu_count() or 4))
    block = 512
    if chunk_size % block:
        chunk_size = max(block, int(round(chunk_size / block)) * block)

    # GDAL tuning; read at use time, so plain env vars are enough.
    os.environ.setdefault("GDAL_CACHEMAX", str(gdal_cache_mb))
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    paths = find_geotiffs(input_dir, exclude={output_path})
    infos = inspect_sources(paths)

    print(f"Input DEMs ({len(infos)}):")
    for info in infos:
        crs_label = info["crs"].to_string() if info["crs"] else "None"
        print(
            f"  - {info['path'].name}: CRS={crs_label}, "
            f"res≈{info['res_x']:.6g} x {info['res_y']:.6g}, "
            f"dtype={info['dtype']}, nodata={info['nodata']}"
        )

    dst_crs = choose_target_crs(infos, crs)
    dst_res = choose_target_resolution(infos, resolution, prefer_resolution)
    dst_nodata = choose_nodata(infos, nodata)
    resampling = RESAMPLE_MAP[resampling_name]

    bounds = union_bounds_in_crs(infos, dst_crs)
    transform, width, height, snapped = build_output_grid(bounds, dst_res)

    for info in infos:
        info["window"] = source_window(info, dst_crs, transform, width, height)
    usable = [i for i in infos if i["window"] is not None]
    if not usable:
        raise ValueError("No input footprints fall inside the output grid.")

    total_px = width * height
    print(f"\nTarget CRS: {dst_crs.to_string()}")
    print(f"Target resolution: {dst_res}")
    print(f"Target nodata: {dst_nodata}")
    print(f"Resampling: {resampling_name}")
    print(f"Overlap method: {method}")
    print(f"Snapped bounds: {snapped}")
    print(f"Output grid: {width} x {height} pixels ({total_px / 1e9:.2f} G px)")
    print(
        f"Streaming in {chunk_size}px chunks, {workers} workers, "
        f"compress={compress}, GDAL cache={gdal_cache_mb} MB"
    )
    print(f"Uncompressed float32 size on disk: ~{total_px * 4 / 1024**3:.1f} GB\n")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": transform,
        "nodata": dst_nodata,
        "tiled": True,
        "blockxsize": block,
        "blockysize": block,
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }
    if compress != "none":
        profile["compress"] = compress
        profile["predictor"] = 3  # float-aware predictor

    windows = list(chunk_windows(width, height, chunk_size))
    n_chunks = len(windows)
    write_lock = threading.Lock()
    pool = SourcePool(
        usable, dst_crs, transform, width, height, dst_nodata, resampling
    )

    done = 0
    valid_px = 0
    vmin = vmax = None
    started = time.monotonic()
    last_print = 0.0
    interactive = sys.stdout.isatty()

    def report(force: bool = False) -> None:
        nonlocal last_print
        now = time.monotonic()
        if not force and now - last_print < 0.5:
            return
        last_print = now
        elapsed = now - started
        frac = done / n_chunks
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_chunks - done) / rate if rate > 0 else 0.0
        line = (
            f"[{done:>7,}/{n_chunks:,}] {frac * 100:5.1f}%  "
            f"elapsed {_fmt_time(elapsed)}  eta {_fmt_time(eta)}  "
            f"{rate:5.1f} chunk/s  valid {valid_px / total_px * 100:5.1f}%"
        )
        if interactive:
            sys.stdout.write("\r" + line + " " * 4)
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    try:
        with rasterio.open(output_path, "w", **profile) as dst:

            def work(window: Window):
                data, n_valid, cmin, cmax = composite_chunk(
                    window, pool, dst_nodata, method
                )
                with write_lock:
                    dst.write(data, 1, window=window)
                return n_valid, cmin, cmax

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(work, w): w for w in windows}
                for future in as_completed(futures):
                    n_valid, cmin, cmax = future.result()
                    done += 1
                    valid_px += n_valid
                    if cmin is not None:
                        vmin = cmin if vmin is None else min(vmin, cmin)
                        vmax = cmax if vmax is None else max(vmax, cmax)
                    report()

            report(force=True)
            if interactive:
                sys.stdout.write("\n")

            if overviews:
                print("Building overviews ...")
                dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
    finally:
        pool.close()

    elapsed = time.monotonic() - started
    print(f"\nWrote mosaic in {_fmt_time(elapsed)}: {output_path}")
    print(f"  size on disk: {output_path.stat().st_size / 1024**3:.2f} GB")
    with rasterio.open(output_path) as out:
        print(
            f"  CRS={out.crs.to_string()}, res={out.res}, "
            f"shape={out.width}x{out.height}, nodata={out.nodata}"
        )
    if valid_px:
        print(
            f"  valid pixels: {valid_px:,} ({valid_px / total_px * 100:.2f}%), "
            f"elev range {vmin:.3f} .. {vmax:.3f}"
        )
    else:
        print("Warning: output has no valid pixels.", file=sys.stderr)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge DEM GeoTIFFs with consistent CRS and resolution."
    )
    p.add_argument("input_dir", type=Path, help="Directory containing DEM GeoTIFF files")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged_dem.tif"),
        help="Output mosaic path (default: merged_dem.tif)",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Target pixel size in CRS units. Default: finest among inputs.",
    )
    p.add_argument(
        "--prefer-resolution",
        choices=("finest", "coarsest"),
        default="finest",
        help="If --resolution omitted, pick finest or coarsest input (default: finest).",
    )
    p.add_argument(
        "--crs",
        type=str,
        default=None,
        help="Target CRS (e.g. EPSG:32633). Default: most common input CRS.",
    )
    p.add_argument(
        "--resampling",
        choices=tuple(RESAMPLE_MAP),
        default="bilinear",
        help="Resampling for continuous DEMs (default: bilinear).",
    )
    p.add_argument(
        "--method",
        choices=METHODS,
        default="first",
        help="How to resolve overlapping pixels (default: first).",
    )
    p.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Nodata value for output (default: from inputs or -9999).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk edge in pixels, rounded to a multiple of 512 (default: 2048).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel chunk workers (default: min(8, CPU count)).",
    )
    p.add_argument(
        "--compress",
        choices=("lzw", "deflate", "zstd", "none"),
        default="lzw",
        help="Output compression (default: lzw).",
    )
    p.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=1024,
        help="GDAL block cache in MB (default: 1024).",
    )
    p.add_argument(
        "--overviews",
        action="store_true",
        help="Build overviews after writing (slower, faster to view in GIS).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        merge_dems(
            input_dir=args.input_dir,
            output_path=args.output,
            resolution=args.resolution,
            prefer_resolution=args.prefer_resolution,
            crs=args.crs,
            resampling_name=args.resampling,
            method=args.method,
            nodata=args.nodata,
            chunk_size=args.chunk_size,
            workers=args.workers,
            compress=args.compress,
            gdal_cache_mb=args.gdal_cache_mb,
            overviews=args.overviews,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

