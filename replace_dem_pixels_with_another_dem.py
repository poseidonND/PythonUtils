"""
Replace pixels in a base DEM with values from a replacement DEM (e.g. the
mosaic produced by merge_dems.py).

Where the replacement has valid (non-nodata) data, those elevations overwrite
the base. Everything is aligned to the base DEM's CRS, resolution, and grid.
Specific replacement values can be discarded with --ignore-value, leaving the
base elevation in place.

Streams chunk-by-chunk like merge_dems.py, so memory stays bounded no matter
how large the base DEM is. Chunks outside the replacement footprint are copied
straight through, and --copy-then-patch skips them entirely.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds


RESAMPLE_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}

DTYPE_CHOICES = ("base", "float32", "float64")


def _fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    """True where pixels are nodata / invalid."""
    if nodata is None:
        if np.issubdtype(arr.dtype, np.floating):
            return ~np.isfinite(arr)
        return np.zeros(arr.shape, dtype=bool)

    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(nodata):
            return ~np.isfinite(arr)
        return np.isclose(arr, nodata) | ~np.isfinite(arr)

    return arr == nodata


def _ignore_mask(
    arr: np.ndarray,
    ignore_values: tuple[float, ...],
    tolerance: float,
) -> np.ndarray:
    """True where the replacement holds a value the caller wants discarded."""
    mask = np.zeros(arr.shape, dtype=bool)
    for value in ignore_values:
        if math.isnan(value):
            mask |= np.isnan(arr)
        elif tolerance > 0:
            mask |= np.abs(arr - value) <= tolerance
        else:
            mask |= arr == value
    return mask


def windows_overlap(a: Window, b: Window) -> bool:
    return not (
        a.col_off + a.width <= b.col_off
        or b.col_off + b.width <= a.col_off
        or a.row_off + a.height <= b.row_off
        or b.row_off + b.height <= a.row_off
    )


def chunk_windows(width: int, height: int, chunk: int):
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off, row_off, w, h)


def footprint_window(src_crs, src_bounds, dst_crs, transform, width, height):
    """Footprint of a raster as a clamped window on the destination grid."""
    b = transform_bounds(src_crs, dst_crs, *src_bounds, densify_pts=21)
    win = window_from_bounds(*b, transform=transform)

    col_off = max(0, math.floor(win.col_off))
    row_off = max(0, math.floor(win.row_off))
    col_end = min(width, math.ceil(win.col_off + win.width))
    row_end = min(height, math.ceil(win.row_off + win.height))
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def choose_work_dtype(base_dtype: str) -> str:
    """float32 keeps DEM precision unless the base needs a wider type."""
    dt = np.dtype(base_dtype)
    if dt == np.float64 or (dt.itemsize >= 4 and np.issubdtype(dt, np.integer)):
        return "float64"
    return "float32"


class GridPool:
    """Per-thread base readers and a replacement WarpedVRT on the base grid.

    GDAL dataset handles are not safe to share across threads, so each worker
    opens its own set on first use.
    """

    def __init__(
        self,
        base_path: Path,
        repl_path: Path,
        base_crs,
        transform,
        width: int,
        height: int,
        src_nodata,
        vrt_nodata,
        resampling,
        work_dtype: str,
    ):
        self.base_path = base_path
        self.repl_path = repl_path
        self.base_crs = base_crs
        self.transform = transform
        self.width = width
        self.height = height
        self.src_nodata = src_nodata
        self.vrt_nodata = vrt_nodata
        self.resampling = resampling
        self.work_dtype = work_dtype
        self._local = threading.local()
        self._opened = []
        self._lock = threading.Lock()

    def entries(self):
        entry = getattr(self._local, "entry", None)
        if entry is not None:
            return entry

        base = rasterio.open(self.base_path)
        repl = rasterio.open(self.repl_path)
        vrt = WarpedVRT(
            repl,
            crs=self.base_crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            nodata=self.vrt_nodata,
            src_nodata=self.src_nodata,
            resampling=self.resampling,
            dtype=self.work_dtype,
        )
        entry = (base, vrt)

        with self._lock:
            self._opened.append((base, repl, vrt))
        self._local.entry = entry
        return entry

    def close(self):
        with self._lock:
            for base, repl, vrt in self._opened:
                for handle in (vrt, repl, base):
                    try:
                        handle.close()
                    except Exception:
                        pass
            self._opened.clear()


def patch_chunk(
    window: Window,
    pool: GridPool,
    repl_window: Window | None,
    base_band: int,
    repl_band: int,
    base_nodata,
    out_nodata,
    out_dtype: np.dtype,
    ignore_values: tuple[float, ...] = (),
    ignore_tolerance: float = 0.0,
):
    """Composite one window: base elevations with the replacement burned in."""
    base_src, repl_vrt = pool.entries()

    base_arr = base_src.read(base_band, window=window)
    base_valid = ~_nodata_mask(base_arr, base_nodata)
    out = base_arr.astype(pool.work_dtype)

    n_replaced = 0
    n_ignored = 0
    if repl_window is not None and windows_overlap(repl_window, window):
        warped = repl_vrt.read(repl_band, window=window)
        repl_valid = ~_nodata_mask(warped, pool.vrt_nodata)
        if ignore_values and repl_valid.any():
            ignored = repl_valid & _ignore_mask(
                warped, ignore_values, ignore_tolerance
            )
            n_ignored = int(np.count_nonzero(ignored))
            if n_ignored:
                repl_valid &= ~ignored
        if repl_valid.any():
            out[repl_valid] = warped[repl_valid]
            n_replaced = int(np.count_nonzero(repl_valid))
            valid = base_valid | repl_valid
        else:
            valid = base_valid
    else:
        valid = base_valid

    if out_nodata is not None:
        out[~valid] = out_nodata

    if np.issubdtype(out_dtype, np.integer):
        fill = out_nodata if out_nodata is not None else 0
        write_arr = np.full(out.shape, fill, dtype=out_dtype)
        finite = np.isfinite(out)
        write_arr[finite] = np.rint(out[finite]).astype(out_dtype)
    else:
        write_arr = out.astype(out_dtype)

    n_valid = int(np.count_nonzero(valid))
    if n_valid:
        values = out[valid]
        vmin = float(values.min())
        vmax = float(values.max())
    else:
        vmin = vmax = None
    return write_arr, n_replaced, n_ignored, n_valid, vmin, vmax


def replace_dem_pixels(
    base_path: Path,
    replacement_path: Path,
    output_path: Path,
    resampling_name: str = "bilinear",
    band: int = 1,
    replacement_band: int | None = None,
    out_dtype_name: str = "base",
    nodata: float | None = None,
    ignore_values: list[float] | None = None,
    ignore_tolerance: float = 0.0,
    chunk_size: int = 2048,
    workers: int | None = None,
    compress: str = "lzw",
    gdal_cache_mb: int = 1024,
    overviews: bool = False,
    copy_then_patch: bool = False,
) -> Path:
    base_path = base_path.resolve()
    replacement_path = replacement_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path in (base_path, replacement_path):
        raise ValueError("Output path must differ from the input rasters.")

    workers = workers or min(8, (os.cpu_count() or 4))
    resampling = RESAMPLE_MAP[resampling_name]
    repl_band = replacement_band or band
    ignore = tuple(float(v) for v in (ignore_values or ()))
    if ignore_tolerance < 0:
        raise ValueError("--ignore-tolerance must be >= 0")

    os.environ.setdefault("GDAL_CACHEMAX", str(gdal_cache_mb))
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    with rasterio.open(base_path) as base, rasterio.open(replacement_path) as repl:
        if base.crs is None:
            raise ValueError(f"Base DEM has no CRS: {base_path}")
        if repl.crs is None:
            raise ValueError(f"Replacement DEM has no CRS: {replacement_path}")
        if band < 1 or band > base.count:
            raise ValueError(f"Base band {band} out of range (1..{base.count})")
        if repl_band < 1 or repl_band > repl.count:
            raise ValueError(
                f"Replacement band {repl_band} out of range (1..{repl.count})"
            )

        base_crs = base.crs
        transform = base.transform
        width, height = base.width, base.height
        base_nodata = base.nodata
        base_dtype = base.dtypes[band - 1]
        base_block = base.block_shapes[band - 1]
        base_profile = base.profile.copy()

        repl_nodata = repl.nodata
        repl_window = footprint_window(
            repl.crs, repl.bounds, base_crs, transform, width, height
        )

        print(f"Base:        {base_path.name}")
        print(
            f"  CRS={base_crs.to_string()}, res={base.res}, "
            f"shape={width}x{height}, dtype={base_dtype}, nodata={base_nodata}"
        )
        print(f"Replacement: {replacement_path.name}")
        print(
            f"  CRS={repl.crs.to_string()}, res={repl.res}, "
            f"shape={repl.width}x{repl.height}, dtype={repl.dtypes[repl_band - 1]}, "
            f"nodata={repl_nodata}"
        )

    if repl_window is None:
        print(
            "Warning: replacement footprint does not intersect the base DEM; "
            "output will be a copy of the base.",
            file=sys.stderr,
        )

    work_dtype = choose_work_dtype(base_dtype)
    vrt_nodata = float(repl_nodata) if repl_nodata is not None else float("nan")

    # With one ignored value and no declared nodata, hand it to the warper as
    # src_nodata so resampling never blends it into neighbouring pixels.
    src_nodata = repl_nodata
    if (
        src_nodata is None
        and len(ignore) == 1
        and not math.isnan(ignore[0])
    ):
        src_nodata = ignore[0]

    if copy_then_patch:
        if nodata is not None:
            print(
                "Warning: --nodata ignored with --copy-then-patch "
                "(base nodata is preserved).",
                file=sys.stderr,
            )
        if out_dtype_name != "base":
            print(
                "Warning: --dtype ignored with --copy-then-patch "
                "(base dtype is preserved).",
                file=sys.stderr,
            )
        out_nodata = base_nodata
        out_dtype = np.dtype(base_dtype)
        block = int(min(base_block))
    else:
        out_nodata = (
            float(nodata)
            if nodata is not None
            else (base_nodata if base_nodata is not None else repl_nodata)
        )
        out_dtype = np.dtype(base_dtype if out_dtype_name == "base" else out_dtype_name)
        block = 512

    if chunk_size % block:
        chunk_size = max(block, int(round(chunk_size / block)) * block)

    total_px = width * height
    if repl_window is not None:
        footprint_px = int(repl_window.width) * int(repl_window.height)
    else:
        footprint_px = 0

    print(f"\nResampling: {resampling_name}")
    print(f"Output dtype: {out_dtype.name}, nodata={out_nodata}")
    print(f"Grid: {width} x {height} pixels ({total_px / 1e9:.2f} G px)")
    print(
        f"Replacement footprint covers {footprint_px / total_px * 100:.2f}% "
        "of the base grid"
    )
    if ignore:
        shown = ", ".join(f"{v:g}" for v in ignore)
        tol = f" +/- {ignore_tolerance:g}" if ignore_tolerance else ""
        print(f"Ignoring replacement value(s): {shown}{tol} (base value kept)")
        if src_nodata in ignore:
            print("  excluded from resampling as src_nodata")
        elif resampling_name != "nearest":
            print(
                "  note: --resampling "
                f"{resampling_name} blends neighbours, so ignored values may "
                "survive as slightly different numbers. Use --resampling "
                "nearest or --ignore-tolerance to catch them.",
                file=sys.stderr,
            )
    print(
        f"Streaming in {chunk_size}px chunks, {workers} workers, "
        f"GDAL cache={gdal_cache_mb} MB"
    )

    windows = list(chunk_windows(width, height, chunk_size))
    if copy_then_patch:
        if repl_window is not None:
            windows = [w for w in windows if windows_overlap(repl_window, w)]
        else:
            windows = []
        print(
            f"Copying base to output, then patching {len(windows):,} chunks "
            "inside the footprint"
        )
    else:
        print(f"Writing a fresh GeoTIFF, compress={compress}")

    started = time.monotonic()

    if copy_then_patch:
        shutil.copy2(base_path, output_path)
        print(f"Copied base in {_fmt_time(time.monotonic() - started)}")
        open_args = (output_path, "r+")
        open_kwargs = {}
    else:
        profile = base_profile
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": out_dtype.name,
                "tiled": True,
                "blockxsize": block,
                "blockysize": block,
                "BIGTIFF": "YES",
                "NUM_THREADS": "ALL_CPUS",
            }
        )
        profile["nodata"] = out_nodata
        profile.pop("predictor", None)
        if compress != "none":
            profile["compress"] = compress
            profile["predictor"] = 3 if np.issubdtype(out_dtype, np.floating) else 2
        else:
            profile.pop("compress", None)
        open_args = (output_path, "w")
        open_kwargs = profile

    n_chunks = len(windows)
    # A fresh output is single-band; a copied one keeps the base's band layout.
    out_band = band if copy_then_patch else 1
    pool = GridPool(
        base_path,
        replacement_path,
        base_crs,
        transform,
        width,
        height,
        src_nodata,
        vrt_nodata,
        resampling,
        work_dtype,
    )
    write_lock = threading.Lock()

    done = 0
    processed_px = 0
    replaced_px = 0
    ignored_px = 0
    valid_px = 0
    vmin = vmax = None
    last_print = 0.0
    interactive = sys.stdout.isatty()

    def report(force: bool = False) -> None:
        nonlocal last_print
        now = time.monotonic()
        if not force and now - last_print < 0.5:
            return
        last_print = now
        elapsed = now - started
        frac = done / n_chunks if n_chunks else 1.0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_chunks - done) / rate if rate > 0 else 0.0
        replaced_pct = replaced_px / processed_px * 100 if processed_px else 0.0
        line = (
            f"[{done:>7,}/{n_chunks:,}] {frac * 100:5.1f}%  "
            f"elapsed {_fmt_time(elapsed)}  eta {_fmt_time(eta)}  "
            f"{rate:5.1f} chunk/s  replaced {replaced_pct:5.1f}%"
        )
        if interactive:
            sys.stdout.write("\r" + line + " " * 4)
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    try:
        with rasterio.open(*open_args, **open_kwargs) as dst:

            def work(window: Window):
                data, n_replaced, n_ignored, n_valid, cmin, cmax = patch_chunk(
                    window,
                    pool,
                    repl_window,
                    band,
                    repl_band,
                    base_nodata,
                    out_nodata,
                    out_dtype,
                    ignore,
                    ignore_tolerance,
                )
                with write_lock:
                    dst.write(data, out_band, window=window)
                return window, n_replaced, n_ignored, n_valid, cmin, cmax

            if n_chunks:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(work, w) for w in windows]
                    for future in as_completed(futures):
                        (
                            window,
                            n_replaced,
                            n_ignored,
                            n_valid,
                            cmin,
                            cmax,
                        ) = future.result()
                        done += 1
                        processed_px += int(window.width) * int(window.height)
                        replaced_px += n_replaced
                        ignored_px += n_ignored
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
    print(f"\nWrote output in {_fmt_time(elapsed)}: {output_path}")
    print(f"  size on disk: {output_path.stat().st_size / 1024**3:.2f} GB")
    with rasterio.open(output_path) as out:
        print(
            f"  CRS={out.crs.to_string()}, res={out.res}, "
            f"shape={out.width}x{out.height}, dtype={out.dtypes[0]}, "
            f"nodata={out.nodata}"
        )
    print(
        f"  replaced {replaced_px:,} pixels "
        f"({replaced_px / total_px * 100:.2f}% of the base grid)"
    )
    if ignore:
        print(f"  kept base at {ignored_px:,} ignored replacement pixels")
    if vmin is not None:
        scope = "patched area" if copy_then_patch else "output"
        print(f"  {scope} elev range {vmin:.3f} .. {vmax:.3f}")
    if not replaced_px:
        print(
            "Warning: no pixels were replaced. Check CRS overlap and nodata values.",
            file=sys.stderr,
        )
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replace base DEM pixels with values from a replacement DEM "
            "(e.g. merged mosaic), aligned to the base grid."
        )
    )
    p.add_argument(
        "base",
        type=Path,
        help="Base DEM GeoTIFF whose pixels will be updated",
    )
    p.add_argument(
        "replacement",
        type=Path,
        help="Replacement DEM GeoTIFF (e.g. merged_dem.tif)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dem_updated.tif"),
        help="Output path (default: dem_updated.tif)",
    )
    p.add_argument(
        "--resampling",
        choices=tuple(RESAMPLE_MAP),
        default="bilinear",
        help="How to resample replacement onto the base grid (default: bilinear).",
    )
    p.add_argument(
        "--band",
        type=int,
        default=1,
        help="Base band to update (default: 1).",
    )
    p.add_argument(
        "--replacement-band",
        type=int,
        default=None,
        help="Replacement band to read (default: same as --band).",
    )
    p.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="base",
        help="Output dtype (default: base, i.e. keep the base DEM's dtype).",
    )
    p.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Output nodata value (default: base nodata, else replacement's).",
    )
    p.add_argument(
        "--ignore-value",
        type=float,
        action="append",
        dest="ignore_values",
        metavar="VALUE",
        help="Treat this replacement value as invalid and keep the base DEM "
        "value there. Repeat the flag for several values.",
    )
    p.add_argument(
        "--ignore-tolerance",
        type=float,
        default=0.0,
        help="Match --ignore-value within +/- this amount (default: 0, exact).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk edge in pixels, rounded to a multiple of the block size "
        "(default: 2048).",
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
    p.add_argument(
        "--copy-then-patch",
        action="store_true",
        help="Copy the base file, then only rewrite chunks inside the "
        "replacement footprint. Much faster for small footprints; keeps the "
        "base dtype, nodata, and compression.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        replace_dem_pixels(
            base_path=args.base,
            replacement_path=args.replacement,
            output_path=args.output,
            resampling_name=args.resampling,
            band=args.band,
            replacement_band=args.replacement_band,
            out_dtype_name=args.dtype,
            nodata=args.nodata,
            ignore_values=args.ignore_values,
            ignore_tolerance=args.ignore_tolerance,
            chunk_size=args.chunk_size,
            workers=args.workers,
            compress=args.compress,
            gdal_cache_mb=args.gdal_cache_mb,
            overviews=args.overviews,
            copy_then_patch=args.copy_then_patch,
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


