#!/usr/bin/env python3
"""
alignment.py -- Two-pass LiDAR registration and full-resolution fusion for
scooter-mounted mobile mapping (TwinPass3D, step 1 of the processing chain).

Pipeline
--------
  S0  discover segments, read trajectories, build a local metric frame
  S1  streamed range-gated / corridor-cropped / voxel-decimated load (alignment math only)
  S2  1-D along-track lock  (kills the ICP "sliding down the street" failure mode)
  S3  global coarse point-to-plane ICP (removes the inter-pass tilt)
  S4  piecewise point-to-plane ICP on overlapping blocks + SLERP blending
  S5  full-resolution streamed merge -> fused .las   (NO downsampling)

The fused cloud ("GT" in file names) is the exact union of both passes:
points_out == points_A + points_B.  Objects that differ between the passes
are handled later, during annotation and by vehicle_persistence.py.

Expected input layout (--data)
------------------------------
  <data>/Zone <z>/<segment>/1/agg_transformed_refined.las   pass 1 (reference A)
  <data>/Zone <z>/<segment>/1/trj_G_Real.las                pass 1 trajectory
  <data>/Zone <z>/<segment>/2/agg_transformed_refined.las   pass 2 (moving B)
  <data>/Zone <z>/<segment>/2/trj_G_Real.las                pass 2 trajectory
The scans are the georeferenced SLAM outputs (UTM, 1 mm scale) with the extra
dimensions distance_m, laser_id and timestamp; the trajectories are position
series.  Rename SCAN_NAME / TRJ_NAME below if your files are named differently.

Outputs (--out), per segment <Zone>_seg<n>/
  <slug>_GT.las          fused point cloud (pass 1 rows first, then pass 2), with a
                         pass_id extra dimension (1 = pass 1, 2 = pass 2)
  <slug>_align.npz       frame, centreline and block transforms (read by split_tiles.py)
  <slug>_report.json     parameters, along-track lock, ICP diagnostics, residuals

Example
-------
  python alignment.py --data Data --out out --list
  python alignment.py --data Data --out out --segment "Zone 1/10"
  python alignment.py --data Data --out out --all

Requirements: numpy, scipy, laspy, open3d, tqdm (optional, progress bars).

Pass 1 is the fixed reference and is written through bit-exact; pass 2 carries
the piecewise warp.  Every stage that touches the big files streams in chunks;
peak RSS stays a few GB regardless of segment size.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import laspy
except ImportError:  # pragma: no cover
    sys.exit("laspy is required:  conda env create -f environment.yml")

from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Every tunable in one place; serialised into each segment's report."""

    # ---- streaming / memory -------------------------------------------------
    chunk_size: int = 4_000_000        # points per laspy chunk (~150 MB working set)
    apply_batch: int = 2_000_000       # points per warp-application sub-batch

    # ---- decimated working cloud (ALIGNMENT MATH ONLY) ----------------------
    range_gate: float = 30.0           # m, per-point sensor range cut
    corridor_half_width: float = 25.0  # m, |cross-track| about the road centreline
    voxel_align: float = 0.10          # m, working-cloud voxel size
    height_band: tuple = (-3.0, 8.0)   # m about the local centreline elevation

    # ---- S2 along-track lock ------------------------------------------------
    lock_search: float = 5.0           # m, +/- along-track search half-range
    lock_bin: float = 0.25             # m, density-profile bin
    lock_detrend: float = 8.0          # m, high-pass window on the profile
    lock_refine: float = 0.75          # m, +/- KD-tree refinement half-range
    lock_refine_step: float = 0.05     # m
    lock_sample: int = 80_000          # points used in the refinement

    # ---- S3 global ICP ------------------------------------------------------
    normal_radius: float = 0.60        # m
    normal_max_nn: int = 30
    global_icp_scales: tuple = (2.0, 1.0, 0.5, 0.25)   # correspondence distances
    global_icp_iters: int = 50

    # ---- S4 piecewise ICP ---------------------------------------------------
    block_len: float = 60.0            # m, block length
    block_stride: float = 30.0         # m, block centre spacing (50 % overlap)
    block_icp_scales: tuple = (1.0, 0.5, 0.25)
    block_icp_iters: int = 40
    block_min_points: int = 4_000      # per side, else block is invalid
    block_min_fitness: float = 0.25
    block_max_rmse: float = 0.40       # m
    block_max_trans: float = 2.50      # m, sanity cap on a block correction
    block_max_rot_deg: float = 3.0     # deg, sanity cap on a block correction
    block_smooth: int = 3              # median-filter window over the block sequence
    tukey_k: float = 0.30              # robust kernel scale (m)

    # ---- output -------------------------------------------------------------
    add_pass_id: bool = True           # append a uint8 `pass_id` extra dimension
    drop_extra_dims: bool = False      # write only core dims (~45 % smaller GT)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    return o


def voxel_decimate(xyz: np.ndarray, voxel: float,
                   origin: np.ndarray | None = None) -> np.ndarray:
    """Keep one representative point per occupied voxel.

    A *fixed* origin keeps the lattice identical across chunks, so decimating
    chunk-by-chunk and then decimating the concatenation is equivalent to
    decimating the whole cloud at once -- that is what keeps this streamable.
    """
    if xyz.shape[0] == 0:
        return xyz
    if origin is None:
        origin = xyz.min(axis=0)
    k = np.floor((xyz - origin) / voxel).astype(np.int64)
    k -= k.min(axis=0)
    span = k.max(axis=0).astype(np.int64) + 1
    key = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
    _, idx = np.unique(key, return_index=True)
    return xyz[np.sort(idx)]


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (N,3) by unit quaternions q (N,4) in scipy's [x,y,z,w] order.

    Uses the Rodrigues form so no (N,3,3) matrix stack is ever materialised --
    that alone saves ~290 MB per 4 M points.
    """
    u = q[:, :3]
    w = q[:, 3:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def smoothstep(w: np.ndarray) -> np.ndarray:
    """C1 blending weight: w'(0)=w'(1)=0, so blended transforms meet smoothly."""
    return w * w * (3.0 - 2.0 * w)


def rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R[:3, :3]) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
#  Segment discovery
# --------------------------------------------------------------------------- #
SCAN_NAME = "agg_transformed_refined.las"
TRJ_NAME = "trj_G_Real.las"


@dataclass
class Segment:
    zone: str
    seg: str
    root: Path

    @property
    def slug(self) -> str:
        z = self.zone.replace(" ", "")
        return f"{z}_seg{self.seg}"

    def scan(self, p: int) -> Path:
        return self.root / str(p) / SCAN_NAME

    def trj(self, p: int) -> Path:
        return self.root / str(p) / TRJ_NAME

    def ok(self) -> bool:
        return all(self.scan(p).exists() and self.trj(p).exists() for p in (1, 2))


def discover_segments(data_root: Path) -> list[Segment]:
    segs: list[Segment] = []
    for zone_dir in sorted(data_root.glob("Zone *")):
        if not zone_dir.is_dir():
            continue
        for seg_dir in sorted(zone_dir.iterdir()):
            if not seg_dir.is_dir() or seg_dir.name.startswith("."):
                continue
            s = Segment(zone_dir.name, seg_dir.name, seg_dir)
            if s.ok():
                segs.append(s)

    def key(s: Segment):
        zn = int("".join(c for c in s.zone if c.isdigit()) or 0)
        try:
            sn = int(s.seg)
        except ValueError:
            sn = 10**6
        return (zn, sn, s.seg)

    return sorted(segs, key=key)


def parse_segment_spec(spec: str, data_root: Path) -> list[Segment]:
    """Accept 'Zone 1/10', 'Zone1_seg10', 'Zone 1' (whole zone) or 'all'."""
    allsegs = discover_segments(data_root)
    if spec.lower() == "all":
        return allsegs
    norm = spec.replace("\\", "/").strip().strip("/")
    hits = [s for s in allsegs
            if norm in (f"{s.zone}/{s.seg}", s.slug, f"{s.zone}_{s.seg}")]
    if hits:
        return hits
    hits = [s for s in allsegs if s.zone == norm or s.zone.replace(" ", "") == norm]
    if hits:
        return hits
    raise SystemExit(
        f"no segment matches {spec!r}. Known: "
        + ", ".join(f"{s.zone}/{s.seg}" for s in allsegs))


# --------------------------------------------------------------------------- #
#  S0 -- local metric frame and road centreline
# --------------------------------------------------------------------------- #
@dataclass
class LocalFrame:
    """Right-handed (along-track, cross-track, up) frame anchored at `origin`.

    Working in this frame lets every downstream array be float32: coordinates
    stay within a few hundred metres of the origin, where float32 resolves
    ~3e-5 m -- two orders below the 1 mm LAS quantisation.  UTM float64 is used
    only at the read/write boundary.
    """
    origin: np.ndarray      # (3,) float64, UTM
    R: np.ndarray           # (3,3) float64, columns = e_along, e_cross, e_up

    def to_local(self, xyz: np.ndarray, dtype=np.float32) -> np.ndarray:
        return ((xyz - self.origin) @ self.R).astype(dtype, copy=False)

    def to_global(self, loc: np.ndarray) -> np.ndarray:
        return loc.astype(np.float64) @ self.R.T + self.origin


def build_frame(trj_a: np.ndarray, trj_b: np.ndarray) -> LocalFrame:
    """Heading recovery: trajectories are position-only, so the along-track
    direction is the principal axis of the pooled path, levelled to horizontal."""
    pooled = np.vstack([trj_a, trj_b])
    origin = pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(pooled - origin, full_matrices=False)
    e = vt[0].copy()
    # Canonical sign: point along the dominant horizontal component's + direction
    if e[int(np.argmax(np.abs(e[:2])))] < 0:
        e = -e
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(up, e)
    n = np.linalg.norm(cross)
    if n < 1e-9:                                    # degenerate (vertical) path
        cross = np.array([0.0, 1.0, 0.0])
    else:
        cross /= n
    e = np.cross(cross, up)                          # re-level the along axis
    e /= np.linalg.norm(e)
    R = np.column_stack([e, cross, up])
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1.0
    return LocalFrame(origin=origin, R=R)


@dataclass
class Centreline:
    """Road centreline resampled on a regular along-track grid.

    Both trajectories are pooled before binning, so this is the *street* centre
    (the two passes run in opposite lanes ~15 m apart) rather than either lane.
    """
    s: np.ndarray          # (M,) increasing
    cross: np.ndarray      # (M,)
    up: np.ndarray         # (M,)

    def eval(self, s_query: np.ndarray) -> np.ndarray:
        sq = np.asarray(s_query, dtype=np.float64)
        out = np.empty((sq.size, 3), dtype=np.float64)
        out[:, 0] = sq
        out[:, 1] = np.interp(sq, self.s, self.cross)
        out[:, 2] = np.interp(sq, self.s, self.up)
        return out

    def cross_at(self, s_query: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s_query, np.float64), self.s, self.cross)

    def up_at(self, s_query: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s_query, np.float64), self.s, self.up)


def build_centreline(trj_a_loc: np.ndarray, trj_b_loc: np.ndarray,
                     bin_m: float = 2.0, pad: float = 60.0) -> Centreline:
    pooled = np.vstack([trj_a_loc, trj_b_loc]).astype(np.float64)
    lo, hi = pooled[:, 0].min() - pad, pooled[:, 0].max() + pad
    nb = max(4, int(np.ceil((hi - lo) / bin_m)) + 1)
    idx = np.clip(((pooled[:, 0] - lo) / bin_m).astype(np.int64), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    sy = np.bincount(idx, weights=pooled[:, 1], minlength=nb)
    sz = np.bincount(idx, weights=pooled[:, 2], minlength=nb)
    grid = lo + (np.arange(nb) + 0.5) * bin_m
    have = cnt > 0
    if have.sum() < 2:
        raise RuntimeError("degenerate trajectory: cannot build a centreline")
    cross = np.interp(grid, grid[have], sy[have] / cnt[have])
    up = np.interp(grid, grid[have], sz[have] / cnt[have])
    # light smoothing so the anchor curve is free of trajectory jitter
    k = max(1, int(round(6.0 / bin_m)) | 1)
    cross = uniform_filter1d(cross, k, mode="nearest")
    up = uniform_filter1d(up, k, mode="nearest")
    return Centreline(s=grid, cross=cross, up=up)


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(t.x), np.asarray(t.y), np.asarray(t.z)])
    if "gps_time" in t.point_format.dimension_names:
        gt = np.asarray(t.gps_time, dtype=np.float64)
    else:
        gt = np.arange(len(xyz), dtype=np.float64)
    o = np.argsort(gt, kind="stable")
    return xyz[o], gt[o]


# --------------------------------------------------------------------------- #
#  S1 -- streamed, gated, decimated working cloud  (+ on-disk cache)
# --------------------------------------------------------------------------- #
CACHE_VERSION = "v3"


def _cache_key(path: Path, cfg: Config, tag: str, frame: "LocalFrame") -> str:
    import hashlib
    st = path.stat()
    # The cached cloud is expressed in the local frame, which is derived from the
    # trajectories -- so the frame itself has to take part in the key, or an
    # edited trajectory would silently be served a stale cloud.
    fh = hashlib.sha1(np.ascontiguousarray(
        np.concatenate([frame.origin.ravel(), frame.R.ravel()]),
        dtype=np.float64).tobytes()).hexdigest()[:10]
    return (f"{CACHE_VERSION}_{tag}_r{cfg.range_gate:g}_c{cfg.corridor_half_width:g}"
            f"_v{cfg.voxel_align:g}_f{fh}_s{st.st_size}_m{int(st.st_mtime)}")


def stream_working_cloud(scan: Path, frame: LocalFrame, centre: Centreline,
                         cfg: Config, cache_dir: Path | None,
                         tag: str, quiet: bool = False) -> np.ndarray:
    """Range-gated + corridor-cropped + voxel-decimated cloud, local float32.

    This cloud is used for the registration mathematics ONLY.  The final GT is
    always rebuilt from the untouched originals in S5.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{_cache_key(scan, cfg, tag, frame)}.npy"
        if cache_file.exists():
            try:
                arr = np.load(cache_file)
                log(f"    cache hit {cache_file.name} -> {len(arr):,} pts", quiet)
                return arr
            except Exception:
                cache_file.unlink(missing_ok=True)

    kept: list[np.ndarray] = []
    origin_grid = None
    n_read = 0
    with laspy.open(str(scan)) as f:
        hdr = f.header
        dims = set(hdr.point_format.dimension_names)
        total_chunks = max(1, math.ceil(hdr.point_count / cfg.chunk_size))
        it = f.chunk_iterator(cfg.chunk_size)
        for ch in tqdm(it, total=total_chunks, desc=f"    read {tag}",
                       disable=quiet, leave=False, unit="chunk"):
            n_read += len(ch)
            # --- range gate (free: the scanner range rides along per point) ---
            if "distance_m" in dims and cfg.range_gate > 0:
                m = np.asarray(ch["distance_m"]) <= cfg.range_gate
                if not m.any():
                    continue
                xyz = np.column_stack([np.asarray(ch.x)[m],
                                       np.asarray(ch.y)[m],
                                       np.asarray(ch.z)[m]])
            else:
                xyz = np.column_stack([np.asarray(ch.x), np.asarray(ch.y),
                                       np.asarray(ch.z)])
            loc = frame.to_local(xyz, dtype=np.float32)
            del xyz
            # --- corridor crop + height band about the centreline -------------
            s = loc[:, 0].astype(np.float64)
            m = np.abs(loc[:, 1] - centre.cross_at(s)) <= cfg.corridor_half_width
            dz = loc[:, 2].astype(np.float64) - centre.up_at(s)
            m &= (dz >= cfg.height_band[0]) & (dz <= cfg.height_band[1])
            if not m.any():
                continue
            loc = loc[m]
            if origin_grid is None:
                origin_grid = np.floor(loc.min(axis=0) / cfg.voxel_align) * cfg.voxel_align
            kept.append(voxel_decimate(loc, cfg.voxel_align, origin_grid))

    if not kept:
        raise RuntimeError(f"{scan}: no points survived gating")
    out = voxel_decimate(np.concatenate(kept), cfg.voxel_align, origin_grid)
    del kept
    log(f"    {tag}: {n_read:,} raw -> {len(out):,} working pts "
        f"({100.0*len(out)/max(1,n_read):.2f} %)", quiet)
    if cache_file is not None:
        try:
            np.save(cache_file, out)
        except Exception as e:      # disk pressure must never kill a run
            log(f"    (cache write skipped: {e})", quiet)
    return out


# --------------------------------------------------------------------------- #
#  Open3D glue
# --------------------------------------------------------------------------- #
def _o3d():
    import open3d as o3d
    return o3d


def make_pcd(xyz: np.ndarray, normals: np.ndarray | None = None):
    o3d = _o3d()
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    if normals is not None:
        p.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
    return p


def estimate_normals(xyz: np.ndarray, cfg: Config) -> np.ndarray:
    o3d = _o3d()
    pcd = make_pcd(xyz)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg.normal_radius, max_nn=cfg.normal_max_nn))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    return np.asarray(pcd.normals)


def icp_multiscale(src_xyz: np.ndarray, tgt_xyz: np.ndarray,
                   tgt_normals: np.ndarray, scales: Sequence[float],
                   iters: int, init: np.ndarray, cfg: Config):
    """Point-to-plane ICP annealed over a coarse->fine correspondence radius.

    A Tukey kernel is used throughout: objects that differ between the passes
    (e.g. parked vehicles) and one-sided facades would otherwise bias the fit.
    """
    o3d = _o3d()
    src = make_pcd(src_xyz)
    tgt = make_pcd(tgt_xyz, tgt_normals)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=cfg.tukey_k))
    T = np.asarray(init, dtype=np.float64).copy()
    res = None
    for d in scales:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, float(d), T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)))
        T = res.transformation.copy()
    return T, (float(res.fitness) if res else 0.0), (float(res.inlier_rmse) if res else np.inf)


# --------------------------------------------------------------------------- #
#  S2 -- 1-D along-track lock
# --------------------------------------------------------------------------- #
def along_track_lock(a_loc: np.ndarray, b_loc: np.ndarray, centre: Centreline,
                     cfg: Config, quiet: bool = False) -> tuple[float, dict]:
    """Estimate the along-track offset delta such that B + delta*e_along ~ A.

    A street corridor is nearly translation-degenerate along its own axis:
    facades, kerbs and the road surface all slide onto themselves, so ICP has
    no gradient to recover this one DOF and happily converges metres off.  We
    therefore solve it separately and *first*, from the along-track density
    profile of above-ground structure (poles, trees, parked cars, gateways),
    which is strongly peaked and carries the missing information.
    """
    def profile(P):
        s = P[:, 0].astype(np.float64)
        h = P[:, 2].astype(np.float64) - centre.up_at(s)
        m = (h > 0.8) & (h < 6.0)                    # above-ground structure only
        return s[m]

    sa, sb = profile(a_loc), profile(b_loc)
    if sa.size < 1000 or sb.size < 1000:
        return 0.0, {"method": "skipped", "n_a": int(sa.size), "n_b": int(sb.size)}

    lo = min(sa.min(), sb.min()) - cfg.lock_search
    hi = max(sa.max(), sb.max()) + cfg.lock_search
    nb = int(np.ceil((hi - lo) / cfg.lock_bin)) + 1
    ha = np.bincount(np.clip(((sa - lo) / cfg.lock_bin).astype(np.int64), 0, nb - 1),
                     minlength=nb).astype(np.float64)
    hb = np.bincount(np.clip(((sb - lo) / cfg.lock_bin).astype(np.int64), 0, nb - 1),
                     minlength=nb).astype(np.float64)

    # High-pass: the raw profile is dominated by the sensor's density envelope,
    # which is identical for both passes and carries no registration signal.
    k = max(3, int(round(cfg.lock_detrend / cfg.lock_bin)) | 1)
    ha -= uniform_filter1d(ha, k, mode="nearest")
    hb -= uniform_filter1d(hb, k, mode="nearest")
    ha /= (np.linalg.norm(ha) + 1e-12)
    hb /= (np.linalg.norm(hb) + 1e-12)

    maxlag = int(round(cfg.lock_search / cfg.lock_bin))
    lags = np.arange(-maxlag, maxlag + 1)
    scores = np.empty(lags.size)
    for i, L in enumerate(lags):
        if L >= 0:
            scores[i] = float(np.dot(ha[L:], hb[:nb - L])) if L < nb else -np.inf
        else:
            scores[i] = float(np.dot(ha[:nb + L], hb[-L:]))
    best = int(np.argmax(scores))
    delta = float(lags[best] * cfg.lock_bin)
    info = {"method": "profile-xcorr", "coarse_delta_m": delta,
            "coarse_score": float(scores[best])}

    # --- KD-tree refinement on the true 3-D residual --------------------------
    rng = np.random.default_rng(0)
    bs = b_loc if len(b_loc) <= cfg.lock_sample else \
        b_loc[rng.choice(len(b_loc), cfg.lock_sample, replace=False)]
    tree = cKDTree(a_loc.astype(np.float64))
    cand = np.arange(delta - cfg.lock_refine, delta + cfg.lock_refine + 1e-9,
                     cfg.lock_refine_step)
    cost = np.empty(cand.size)
    probe = bs.astype(np.float64).copy()
    base_s = probe[:, 0].copy()
    for i, d in enumerate(cand):
        probe[:, 0] = base_s + d
        dist, _ = tree.query(probe, k=1, distance_upper_bound=1.0)
        dist[~np.isfinite(dist)] = 1.0
        cost[i] = float(np.mean(dist))
    j = int(np.argmin(cost))
    delta = float(cand[j])
    info.update(refined_delta_m=delta, refined_cost=float(cost[j]),
                cost_at_zero=float(cost[int(np.argmin(np.abs(cand)))]))
    log(f"    along-track lock: {delta:+.3f} m "
        f"(profile {info['coarse_delta_m']:+.2f} m)", quiet)
    return delta, info


# --------------------------------------------------------------------------- #
#  S4 -- piecewise transform with SLERP blending
# --------------------------------------------------------------------------- #
@dataclass
class PiecewiseTransform:
    """A spatially-varying rigid warp  T(s) : local(B) -> local(A).

    Composition:  p |-> blend_s( T_global(p) ).

    Blending (the important bit).  Each block ICP yields a global rigid map
    T_k(x) = R_k x + u_k.  Naively LERP-ing u_k between blocks is wrong: the
    translation part of a rigid map is meaningless without the rotation centre
    it is expressed about, so two *identical* transforms stored against
    different block centres would blend into something that is neither.

    Instead we interpolate the *displacement of a shared anchor point*.  With
    a = a(s) the road centreline at the query point's own along-track station,

        R(s) = Slerp(R_k, R_{k+1}; w)                         w = smoothstep(.)
        d_k  = R_k a + u_k - a          (block k's displacement of a)
        d(s) = (1-w) d_k + w d_{k+1}
        T(s) : p |-> R(s) (p - a) + a + d(s)

    This is rigid for every w, is C1 in s because smoothstep has vanishing
    endpoint derivatives, and is *exact* in the degenerate case: if
    T_k == T_{k+1} == T then d_k = d_{k+1} = Ta - a and the formula collapses
    to T(p) identically, for any anchor.  Blocks are indexed by the point's own
    along-track coordinate rather than its acquisition time, which makes T a
    smooth spatial deformation field and guarantees no seam can open between
    points that sit next to each other but were seen seconds apart.
    """
    centres: np.ndarray        # (K,) block centre stations, increasing
    quats: np.ndarray          # (K,4) scipy [x,y,z,w]
    trans: np.ndarray          # (K,3) u_k
    T_global: np.ndarray       # (4,4) coarse pre-transform
    centre_line: Centreline
    valid: np.ndarray          # (K,) bool, diagnostics only
    fitness: np.ndarray
    rmse: np.ndarray

    def __post_init__(self):
        self._slerp = None
        if len(self.centres) >= 2:
            self._slerp = Slerp(self.centres, Rotation.from_quat(self.quats))

    # -- diagnostics ------------------------------------------------------
    def block_table(self) -> list[dict]:
        out = []
        for i, c in enumerate(self.centres):
            R = Rotation.from_quat(self.quats[i]).as_matrix()
            a = self.centre_line.eval(np.array([c]))[0]
            d = R @ a + self.trans[i] - a
            out.append(dict(centre_s=float(c), valid=bool(self.valid[i]),
                            fitness=float(self.fitness[i]), rmse=float(self.rmse[i]),
                            rot_deg=rot_angle_deg(R),
                            disp_along=float(d[0]), disp_cross=float(d[1]),
                            disp_up=float(d[2]), disp_norm=float(np.linalg.norm(d))))
        return out

    # -- application ------------------------------------------------------
    def apply(self, pts_local: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
        """Warp local-frame points of pass B into pass A's frame."""
        if pts_local.shape[0] == 0:
            return pts_local
        out = np.empty_like(pts_local, dtype=np.float32)
        for i0 in range(0, pts_local.shape[0], batch):
            out[i0:i0 + batch] = self._apply_batch(pts_local[i0:i0 + batch])
        return out

    def _apply_batch(self, P: np.ndarray) -> np.ndarray:
        Rg = self.T_global[:3, :3]
        tg = self.T_global[:3, 3]
        X = P.astype(np.float64) @ Rg.T + tg          # global coarse first

        if self._slerp is None:                       # single block -> rigid
            R = Rotation.from_quat(self.quats[0]).as_matrix()
            return (X @ R.T + self.trans[0]).astype(np.float32)

        s_true = X[:, 0]
        s_c = np.clip(s_true, self.centres[0], self.centres[-1])
        idx = np.clip(np.searchsorted(self.centres, s_c, side="right") - 1,
                      0, len(self.centres) - 2)
        c0 = self.centres[idx]
        c1 = self.centres[idx + 1]
        w = smoothstep(np.clip((s_c - c0) / (c1 - c0), 0.0, 1.0))

        # Slerp with the smoothstep weight == Slerp queried at a warped station
        q = self._slerp(c0 + w * (c1 - c0)).as_quat()

        a = self.centre_line.eval(s_true)             # anchor at the point's own s
        q0 = self.quats[idx]
        q1 = self.quats[idx + 1]
        d0 = qrot(q0, a) + self.trans[idx] - a
        d1 = qrot(q1, a) + self.trans[idx + 1] - a
        d = d0 + w[:, None] * (d1 - d0)
        return (qrot(q, X - a) + a + d).astype(np.float32)


def fit_piecewise(a_loc: np.ndarray, b_loc: np.ndarray, a_normals: np.ndarray,
                  T_global: np.ndarray, centre: Centreline, cfg: Config,
                  quiet: bool = False) -> PiecewiseTransform:
    """Overlapping-block ICP.

    A single global 4x4 cannot absorb trajectory drift: over one segment the
    two passes diverge as a slow function of along-track station (measured on
    this dataset as several metres of relative tilt end-to-end).  Blocks of
    `block_len` at 50 % overlap localise the estimate; SLERP blending across
    the overlaps removes the seams a hard block partition would produce.
    """
    Rg, tg = T_global[:3, :3], T_global[:3, 3]
    b_g = (b_loc.astype(np.float64) @ Rg.T + tg)      # coarse-aligned B

    ao = np.argsort(a_loc[:, 0], kind="stable")
    a_sorted = a_loc[ao].astype(np.float64)
    an_sorted = a_normals[ao]
    a_s = a_sorted[:, 0]
    bo = np.argsort(b_g[:, 0], kind="stable")
    b_sorted = b_g[bo]
    b_s = b_sorted[:, 0]

    s_lo = max(a_s[0], b_s[0])
    s_hi = min(a_s[-1], b_s[-1])
    if not np.isfinite(s_lo) or s_hi - s_lo < cfg.block_len:
        centres = np.array([0.5 * (s_lo + s_hi)])
    else:
        n = max(2, int(np.floor((s_hi - s_lo - cfg.block_len) / cfg.block_stride)) + 1)
        centres = s_lo + cfg.block_len / 2.0 + np.arange(n) * cfg.block_stride
        # stretch the last centre so the blocks span the full overlap
        centres = np.append(centres, s_hi - cfg.block_len / 2.0)
        centres = np.unique(np.round(centres, 3))
        # drop near-duplicate centres: c_{k+1}-c_k appears in the blend weight's
        # denominator, so a sub-metre gap would make the field switch abruptly
        keep = [0]
        for j in range(1, centres.size):
            if centres[j] - centres[keep[-1]] >= 1.0:
                keep.append(j)
        centres = centres[keep]
        if centres.size < 2:
            centres = np.array([0.5 * (s_lo + s_hi)])

    K = centres.size
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (K, 1))
    trans = np.zeros((K, 3))
    valid = np.zeros(K, dtype=bool)
    fitness = np.zeros(K)
    rmse = np.full(K, np.inf)
    half = cfg.block_len / 2.0
    margin = 0.25 * cfg.block_len                     # target gets extra context

    for k, c in enumerate(tqdm(centres, desc="    block ICP", disable=quiet,
                              leave=False, unit="blk")):
        ia0, ia1 = np.searchsorted(a_s, [c - half - margin, c + half + margin])
        ib0, ib1 = np.searchsorted(b_s, [c - half, c + half])
        na, nb = ia1 - ia0, ib1 - ib0
        if na < cfg.block_min_points or nb < cfg.block_min_points:
            continue
        try:
            T, fit, rms = icp_multiscale(
                b_sorted[ib0:ib1], a_sorted[ia0:ia1], an_sorted[ia0:ia1],
                cfg.block_icp_scales, cfg.block_icp_iters, np.eye(4), cfg)
        except Exception as e:                        # never let one block abort
            log(f"      block {c:+.0f} m ICP failed: {e}", quiet)
            continue

        anchor = centre.eval(np.array([c]))[0]
        disp = T[:3, :3] @ anchor + T[:3, 3] - anchor
        ang = rot_angle_deg(T)
        if (fit < cfg.block_min_fitness or rms > cfg.block_max_rmse
                or np.linalg.norm(disp) > cfg.block_max_trans
                or ang > cfg.block_max_rot_deg):
            continue
        quats[k] = Rotation.from_matrix(T[:3, :3]).as_quat()
        trans[k] = T[:3, 3]
        valid[k] = True
        fitness[k] = fit
        rmse[k] = rms

    n_valid = int(valid.sum())
    if n_valid == 0:
        log("    WARNING: no block converged -- falling back to the global fit", quiet)
        return PiecewiseTransform(centres, quats, trans, T_global, centre,
                                  valid, fitness, rmse)

    # --- fill invalid blocks from their valid neighbours ---------------------
    if n_valid < K:
        rv = Rotation.from_quat(quats[valid])
        if n_valid == 1:
            quats[~valid] = quats[valid][0]
            trans[~valid] = trans[valid][0]
        else:
            sl = Slerp(centres[valid], rv)
            q_at = np.clip(centres[~valid], centres[valid][0], centres[valid][-1])
            quats[~valid] = sl(q_at).as_quat()
            for j in range(3):
                trans[~valid, j] = np.interp(centres[~valid], centres[valid],
                                             trans[valid, j])

    # --- regularise the block sequence --------------------------------------
    # One block that latched onto a bus or a hedge would otherwise punch a local
    # bump into the warp field.  A short median filter on the rotation-vector /
    # translation sequence suppresses exactly that without smearing real drift.
    if K >= 3 and cfg.block_smooth >= 3:
        w = min(cfg.block_smooth | 1, K if K % 2 else K - 1)
        rv = Rotation.from_quat(quats).as_rotvec()
        rv = median_filter(rv, size=(w, 1), mode="nearest")
        tr = median_filter(trans, size=(w, 1), mode="nearest")
        quats = Rotation.from_rotvec(rv).as_quat()
        trans = tr

    log(f"    piecewise: {n_valid}/{K} blocks converged "
        f"(median fitness {np.median(fitness[valid]):.3f}, "
        f"median rmse {np.median(rmse[valid]):.3f} m)", quiet)
    return PiecewiseTransform(centres, quats, trans, T_global, centre,
                              valid, fitness, rmse)


# --------------------------------------------------------------------------- #
#  S5 -- full-resolution streamed merge
# --------------------------------------------------------------------------- #
def build_output_header(hdr_a, cfg: Config):
    """Output header inherits pass A's scales/offsets/point format verbatim, so
    pass A round-trips bit-exactly and UTM precision is never renegotiated."""
    from laspy.vlrs.known import ExtraBytesVlr

    def _clip(s: str) -> str:
        # LAS spec: the extra-bytes description field is 32 bytes, no more.
        b = (s or "").encode("utf-8", "ignore")[:32]
        return b.decode("utf-8", "ignore")

    out = laspy.LasHeader(version=str(hdr_a.version),
                          point_format=laspy.PointFormat(hdr_a.point_format.id))
    if not cfg.drop_extra_dims:
        for ed in hdr_a.point_format.extra_dimensions:
            out.add_extra_dim(laspy.ExtraBytesParams(
                name=_clip(ed.name), type=ed.dtype,
                description=_clip(ed.description)))
    if cfg.add_pass_id:
        out.add_extra_dim(laspy.ExtraBytesParams(
            name="pass_id", type=np.uint8,
            description=_clip("1=passA(ref) 2=passB(warp)")))
    out.scales = np.asarray(hdr_a.scales, dtype=np.float64)
    out.offsets = np.asarray(hdr_a.offsets, dtype=np.float64)
    for v in hdr_a.vlrs:                      # carry CRS / GeoKeys through
        if not isinstance(v, ExtraBytesVlr):
            out.vlrs.append(v)
    return out


def _write_pass(writer, scan: Path, pass_id: int, pw: PiecewiseTransform | None,
                frame: LocalFrame, cfg: Config, out_hdr, quiet: bool) -> int:
    """Stream one pass into the open writer.  Pass 1 (pw=None) is copied
    through unchanged; pass 2 is warped point by point.  Returns the number of
    points written (always every point of the input file)."""
    out_dims = set(out_hdr.point_format.dimension_names)
    kept = 0
    with laspy.open(str(scan)) as f:
        hdr = f.header
        raw_ok = (pw is None
                  and np.allclose(hdr.scales, out_hdr.scales)
                  and np.allclose(hdr.offsets, out_hdr.offsets))
        n_chunks = max(1, math.ceil(hdr.point_count / cfg.chunk_size))
        copy_dims = [d for d in hdr.point_format.dimension_names
                     if d in out_dims and (raw_ok or d not in ("X", "Y", "Z"))]
        for ch in tqdm(f.chunk_iterator(cfg.chunk_size), total=n_chunks,
                       desc=f"    merge pass {pass_id}", disable=quiet,
                       leave=False, unit="chunk"):
            nk = len(ch)
            g = None
            if pw is not None:
                # world -> local frame -> blended piecewise warp -> world.  laspy
                # rounds the result to the 1 mm grid of the output header.
                xyz = np.column_stack([np.asarray(ch.x), np.asarray(ch.y),
                                       np.asarray(ch.z)])
                loc = frame.to_local(xyz, dtype=np.float32)
                del xyz
                g = frame.to_global(pw.apply(loc, cfg.apply_batch))
                del loc
            kept += nk
            if nk == 0:
                continue
            sel = slice(None)
            rec = laspy.ScaleAwarePointRecord.zeros(
                nk, point_format=out_hdr.point_format,
                scales=out_hdr.scales, offsets=out_hdr.offsets)
            for d in copy_dims:
                rec[d] = np.asarray(ch[d])[sel]
            if not raw_ok:
                if g is None:                       # pass A, different header grid
                    g = np.column_stack([np.asarray(ch.x)[sel],
                                         np.asarray(ch.y)[sel],
                                         np.asarray(ch.z)[sel]])
                rec.x = g[:, 0]
                rec.y = g[:, 1]
                rec.z = g[:, 2]
            if cfg.add_pass_id:
                rec["pass_id"][:] = pass_id
            writer.write_points(rec)
            del rec, g
    return kept


def merge_full_resolution(seg: Segment, frame: LocalFrame,
                          pw: PiecewiseTransform, cfg: Config, out_path: Path,
                          quiet: bool, force: bool = False) -> dict:
    """Stream both originals, warp B, write the union (|G| = |A| + |B|).

    Nothing here is decimated: the working cloud from S1 is used only to *fit*
    the transform, never to build the product.
    """
    with laspy.open(str(seg.scan(1))) as fa:
        hdr_a = fa.header
        n_a = int(hdr_a.point_count)
    with laspy.open(str(seg.scan(2))) as fb:
        n_b = int(fb.header.point_count)
    out_hdr = build_output_header(hdr_a, cfg)

    est = (n_a + n_b) * out_hdr.point_format.size
    free = shutil.disk_usage(out_path.parent).free
    log(f"    merge: {n_a:,} + {n_b:,} pts, ~{est/2**30:.2f} GiB "
        f"({out_hdr.point_format.size} B/pt), {free/2**30:.1f} GiB free", quiet)
    if est > free * 0.97 and not force:
        raise RuntimeError(
            f"need ~{est/2**30:.1f} GiB but only {free/2**30:.1f} GiB free. "
            f"Use --drop-extra-dims (~45 % smaller), free space, or --force.")

    tmp = out_path.with_suffix(".partial.las")
    t0 = time.time()
    with laspy.open(str(tmp), mode="w", header=out_hdr) as w:
        # Pass 1 rows first, then pass 2, each in original file order: the
        # label transfer (transfer_labels.py) relies on this row order.
        ka = _write_pass(w, seg.scan(1), 1, None, frame, cfg, out_hdr, quiet)
        kb = _write_pass(w, seg.scan(2), 2, pw, frame, cfg, out_hdr, quiet)
    tmp.replace(out_path)
    dt = time.time() - t0
    res = dict(points_a_in=n_a, points_b_in=n_b,
               points_a_out=ka, points_b_out=kb,
               points_total=ka + kb, expected_union=n_a + n_b,
               bytes_per_point=int(out_hdr.point_format.size),
               file_bytes=int(out_path.stat().st_size), seconds=dt)
    log(f"    GT written: {ka+kb:,} pts "
        f"(= {n_a:,} + {n_b:,}) in {dt/60:.1f} min", quiet)
    return res


# --------------------------------------------------------------------------- #
#  Quality metrics
# --------------------------------------------------------------------------- #
def residual_metrics(a_loc: np.ndarray, b_before: np.ndarray, b_after: np.ndarray,
                     sample: int = 300_000, cap: float = 1.0) -> dict:
    rng = np.random.default_rng(12345)
    n = b_before.shape[0]
    sel = np.arange(n) if n <= sample else rng.choice(n, sample, replace=False)
    tree = cKDTree(a_loc.astype(np.float64))

    def stat(P):
        d, _ = tree.query(P[sel].astype(np.float64), k=1, distance_upper_bound=cap)
        fin = np.isfinite(d)
        if not fin.any():
            return dict(median=float("nan"), mean=float("nan"), p90=float("nan"),
                        frac_below_10cm=0.0, matched=0.0)
        dv = d[fin]
        return dict(median=float(np.median(dv)), mean=float(dv.mean()),
                    p90=float(np.percentile(dv, 90)),
                    frac_below_10cm=float((dv < 0.10).mean()),
                    matched=float(fin.mean()))

    before, after = stat(b_before), stat(b_after)
    return dict(before=before, after=after,
                median_improvement_m=before["median"] - after["median"],
                cap_m=cap, n_sampled=int(len(sel)))


def window_metrics(a_loc: np.ndarray, b_before: np.ndarray, b_after: np.ndarray,
                   step: float = 20.0, cap_before: float = 3.0,
                   cap_after: float = 1.0, min_pairs: int = 1500,
                   sample: int = 40_000) -> list[dict]:
    """Per-along-track-window registration quality, before and after the warp.

    Both readouts come from *matched nearest-neighbour pairs*, so a window in
    which one pass barely has coverage reports `null` instead of a spurious
    number.  An earlier version compared an independent low percentile of each
    pass's ground points and produced multi-metre phantom offsets wherever the
    two passes did not see the same surface -- exactly the ends of a segment
    and the far side of intersections.
    """
    tree = cKDTree(a_loc.astype(np.float64))
    az = a_loc[:, 2].astype(np.float64)
    rng = np.random.default_rng(4242)
    s_after = b_after[:, 0]
    lo = max(float(a_loc[:, 0].min()), float(s_after.min()))
    hi = min(float(a_loc[:, 0].max()), float(s_after.max()))
    out = []
    for s in np.arange(lo, hi - step + 1e-9, step):
        idx = np.flatnonzero((s_after >= s) & (s_after < s + step))
        n_a = int(((a_loc[:, 0] >= s) & (a_loc[:, 0] < s + step)).sum())
        if idx.size > sample:
            idx = rng.choice(idx, sample, replace=False)
        rec = dict(s=float(s + step / 2), n_a=n_a, n_b=int(idx.size))
        for tag, P, cap in (("before", b_before, cap_before),
                            ("after", b_after, cap_after)):
            if idx.size < 200 or n_a < 200:
                rec[tag] = None
                continue
            d, j = tree.query(P[idx].astype(np.float64), k=1,
                              distance_upper_bound=cap)
            fin = np.isfinite(d)
            if int(fin.sum()) < min_pairs:
                rec[tag] = None
                continue
            dz = P[idx][fin, 2].astype(np.float64) - az[j[fin]]
            rec[tag] = dict(nn_median=float(np.median(d[fin])),
                            frac_below_10cm=float((d[fin] < 0.10).mean()),
                            dz_median=float(np.median(dz)),
                            n_pairs=int(fin.sum()),
                            matched=float(fin.mean()))
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
#  Per-segment state (consumed by split_tiles.py)
# --------------------------------------------------------------------------- #
def save_state(path: Path, frame: LocalFrame, centre: Centreline,
               pw: PiecewiseTransform, seg: Segment) -> None:
    """Everything needed to re-apply the registration later without refitting."""
    np.savez(path,
             origin=frame.origin, R=frame.R,
             cl_s=centre.s, cl_cross=centre.cross, cl_up=centre.up,
             centres=pw.centres, quats=pw.quats, trans=pw.trans,
             T_global=pw.T_global, valid=pw.valid,
             fitness=pw.fitness, rmse=pw.rmse,
             zone=seg.zone, seg=seg.seg)


def load_state(path: Path):
    """Return (frame, centre, pw) from a <slug>_align.npz written by save_state.
    State files from older runs may carry extra arrays; they are ignored."""
    d = np.load(path, allow_pickle=False)
    frame = LocalFrame(origin=d["origin"], R=d["R"])
    centre = Centreline(s=d["cl_s"], cross=d["cl_cross"], up=d["cl_up"])
    pw = PiecewiseTransform(centres=d["centres"], quats=d["quats"], trans=d["trans"],
                            T_global=d["T_global"], centre_line=centre,
                            valid=d["valid"], fitness=d["fitness"], rmse=d["rmse"])
    return frame, centre, pw


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def process_segment(seg: Segment, cfg: Config, out_root: Path,
                    cache_dir: Path | None, do_merge: bool,
                    quiet: bool = False, force: bool = False,
                    overwrite: bool = False) -> dict:
    t_start = time.time()
    out_dir = out_root / seg.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / f"{seg.slug}_GT.las"
    state_path = out_dir / f"{seg.slug}_align.npz"
    rep_path = out_dir / f"{seg.slug}_report.json"

    log(f"=== {seg.zone} / segment {seg.seg}  ({seg.slug}) ===", quiet)
    rep: dict = dict(zone=seg.zone, segment=seg.seg, slug=seg.slug,
                     scan_a=str(seg.scan(1)), scan_b=str(seg.scan(2)),
                     config=cfg.as_dict())

    # ---- S0 frame + centreline ---------------------------------------------
    trj_a, _ = read_trajectory(seg.trj(1))
    trj_b, _ = read_trajectory(seg.trj(2))
    frame = build_frame(trj_a, trj_b)
    ta_l = frame.to_local(trj_a, np.float64)
    tb_l = frame.to_local(trj_b, np.float64)
    centre = build_centreline(ta_l, tb_l)
    rep["frame"] = dict(origin=frame.origin, R=frame.R,
                        e_along=frame.R[:, 0], e_cross=frame.R[:, 1],
                        traj_len_a=float(np.abs(np.diff(ta_l[:, 0])).sum()),
                        traj_len_b=float(np.abs(np.diff(tb_l[:, 0])).sum()),
                        lane_offset_cross=float(tb_l[:, 1].mean() - ta_l[:, 1].mean()))
    log(f"    frame: along={np.round(frame.R[:,0],4)}  "
        f"lane offset {rep['frame']['lane_offset_cross']:+.1f} m", quiet)

    # ---- S1 working clouds --------------------------------------------------
    A = stream_working_cloud(seg.scan(1), frame, centre, cfg, cache_dir, "p1", quiet)
    B = stream_working_cloud(seg.scan(2), frame, centre, cfg, cache_dir, "p2", quiet)
    rep["working_cloud"] = dict(n_a=int(len(A)), n_b=int(len(B)),
                                voxel=cfg.voxel_align, range_gate=cfg.range_gate)

    # ---- S2 along-track lock ------------------------------------------------
    delta, lock_info = along_track_lock(A, B, centre, cfg, quiet)
    rep["along_track_lock"] = lock_info

    # ---- S3 global coarse ICP ----------------------------------------------
    log("    estimating target normals ...", quiet)
    a_norm = estimate_normals(A, cfg)
    T_glob, g_fit, g_rmse = icp_multiscale(
        B, A, a_norm, cfg.global_icp_scales, cfg.global_icp_iters,
        rigid(np.eye(3), np.array([delta, 0.0, 0.0])), cfg)
    rep["global_icp"] = dict(T=T_glob, fitness=g_fit, rmse=g_rmse,
                             rot_deg=rot_angle_deg(T_glob),
                             translation=T_glob[:3, 3])
    log(f"    global ICP: fitness {g_fit:.3f}  rmse {g_rmse:.3f} m  "
        f"rot {rot_angle_deg(T_glob):.3f} deg", quiet)

    # ---- S4 piecewise ICP ---------------------------------------------------
    pw = fit_piecewise(A, B, a_norm, T_glob, centre, cfg, quiet)
    del a_norm
    B_w = pw.apply(B, cfg.apply_batch)
    rep["piecewise"] = dict(n_blocks=int(len(pw.centres)),
                            n_valid=int(pw.valid.sum()),
                            block_len=cfg.block_len, block_stride=cfg.block_stride,
                            blocks=pw.block_table())
    rep["residuals"] = residual_metrics(A, B, B_w)
    rep["windows"] = window_metrics(A, B, B_w)
    r = rep["residuals"]
    log(f"    residual B->A: median {r['before']['median']:.3f} -> "
        f"{r['after']['median']:.3f} m | <10 cm "
        f"{100*r['before']['frac_below_10cm']:.1f} -> "
        f"{100*r['after']['frac_below_10cm']:.1f} %", quiet)

    save_state(state_path, frame, centre, pw, seg)
    del A, B, B_w

    # ---- S5 full-resolution merge ------------------------------------------
    if do_merge:
        if gt_path.exists() and not overwrite:
            log(f"    GT exists, skipping merge ({gt_path.name}); "
                f"use --overwrite to rebuild", quiet)
            rep["merge"] = dict(skipped=True, path=str(gt_path),
                                file_bytes=int(gt_path.stat().st_size))
        else:
            rep["merge"] = merge_full_resolution(seg, frame, pw, cfg,
                                                 gt_path, quiet, force)
            rep["merge"]["path"] = str(gt_path)
    else:
        rep["merge"] = dict(skipped=True, reason="--no-merge")

    rep["seconds"] = time.time() - t_start
    rep_path.write_text(json.dumps(_jsonable(rep), indent=2))

    log(f"    done in {rep['seconds']/60:.1f} min -> {out_dir}", quiet)
    return rep


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-pass LiDAR alignment + full-resolution GT generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="Data", type=Path, help="dataset root")
    p.add_argument("--out", default="out", type=Path, help="output root")
    p.add_argument("--segment", action="append", default=None,
                   help="'Zone 1/10', 'Zone1_seg10', 'Zone 2' or 'all' (repeatable)")
    p.add_argument("--all", action="store_true", help="process every segment")
    p.add_argument("--list", action="store_true", help="list segments and exit")
    p.add_argument("--limit", type=int, default=0, help="process at most N segments")

    p.add_argument("--no-merge", action="store_true",
                   help="fit + report only, do not write the fused .las")
    p.add_argument("--overwrite", action="store_true", help="rebuild an existing GT")
    p.add_argument("--force", action="store_true", help="ignore the disk-space guard")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--cache", default=".alignment_cache", type=Path)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--clean-cache", action="store_true",
                   help="delete cache entries from older pipeline versions and exit")

    p.add_argument("--range-gate", type=float, default=None)
    p.add_argument("--voxel", type=float, default=None, help="working-cloud voxel (m)")
    p.add_argument("--block-len", type=float, default=None)
    p.add_argument("--block-stride", type=float, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--drop-extra-dims", action="store_true",
                   help="omit distance_m/laser_id/timestamp from the GT (~45%% smaller)")
    p.add_argument("--no-pass-id", action="store_true",
                   help="do not append the pass_id extra dimension")
    p.add_argument("--seed", type=int, default=20260831)
    return p


def clean_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        print(f"{cache_dir} does not exist")
        return
    stale = [f for f in cache_dir.iterdir()
             if f.is_file() and not f.name.startswith(CACHE_VERSION + "_")]
    size = sum(f.stat().st_size for f in stale)
    print(f"{len(stale)} stale cache files ({size/2**30:.2f} GiB) in {cache_dir}")
    if not stale:
        return
    if input("delete them? [y/N] ").strip().lower() == "y":
        for f in stale:
            f.unlink()
        print("deleted")
    else:
        print("kept")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))

    if args.clean_cache:
        clean_cache(args.cache)
        return 0

    segs_all = discover_segments(args.data)
    if args.list:
        print(f"{len(segs_all)} segments under {args.data}:")
        for s in segs_all:
            gb = sum(s.scan(p).stat().st_size for p in (1, 2)) / 2**30
            print(f"  {s.zone:8s} / {s.seg:3s}  slug={s.slug:16s} {gb:5.1f} GiB")
        return 0

    if args.all or not args.segment:
        segs = segs_all
    else:
        seen, segs = set(), []
        for spec in args.segment:
            for s in parse_segment_spec(spec, args.data):
                if s.slug not in seen:
                    seen.add(s.slug)
                    segs.append(s)
    if args.limit:
        segs = segs[:args.limit]
    if not segs:
        sys.exit("no segments to process")

    cfg = Config()
    for attr, val in (("range_gate", args.range_gate), ("voxel_align", args.voxel),
                      ("block_len", args.block_len), ("block_stride", args.block_stride),
                      ("chunk_size", args.chunk_size)):
        if val is not None:
            setattr(cfg, attr, val)
    if args.drop_extra_dims:
        cfg.drop_extra_dims = True
    if args.no_pass_id:
        cfg.add_pass_id = False

    cache_dir = None if args.no_cache else args.cache
    args.out.mkdir(parents=True, exist_ok=True)

    log(f"{len(segs)} segment(s); out={args.out}; "
        f"merge={'off' if args.no_merge else 'on'}", args.quiet)

    summary, failures = [], []
    for i, s in enumerate(segs, 1):
        log(f"--- [{i}/{len(segs)}] {s.zone}/{s.seg} ---", args.quiet)
        try:
            rep = process_segment(s, cfg, args.out, cache_dir,
                                  do_merge=not args.no_merge, quiet=args.quiet,
                                  force=args.force, overwrite=args.overwrite)
            summary.append(dict(
                slug=s.slug, zone=s.zone, segment=s.seg,
                residual_before=rep["residuals"]["before"]["median"],
                residual_after=rep["residuals"]["after"]["median"],
                frac10_before=rep["residuals"]["before"]["frac_below_10cm"],
                frac10_after=rep["residuals"]["after"]["frac_below_10cm"],
                blocks=rep["piecewise"]["n_valid"],
                points_out=rep["merge"].get("points_total"),
                seconds=rep["seconds"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append(dict(slug=s.slug, error=str(e)))
            log(f"    FAILED: {e}", args.quiet)

    (args.out / "summary.json").write_text(
        json.dumps(_jsonable(dict(segments=summary, failures=failures,
                                  config=cfg.as_dict())), indent=2))
    print("\n" + "=" * 78)
    print(f"{'segment':16s} {'res before':>11s} {'res after':>10s} "
          f"{'<10cm b/a':>13s} {'blk':>4s} {'points out':>13s}")
    print("-" * 78)
    for r in summary:
        print(f"{r['slug']:16s} {r['residual_before']:>10.3f}m {r['residual_after']:>9.3f}m "
              f"{100*r['frac10_before']:>5.1f}/{100*r['frac10_after']:<7.1f} "
              f"{r['blocks']:>4d} "
              f"{(r['points_out'] or 0):>13,d}")
    if failures:
        print(f"\n{len(failures)} failure(s): "
              + ", ".join(f["slug"] for f in failures))
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
