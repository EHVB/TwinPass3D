#!/usr/bin/env python3
"""
split_tiles.py -- Cut every annotated segment into fixed-length (default 25 m)
along-street tiles for model training, for the ground truth and for the
labelled single-pass scans, with identical tile boundaries in every source
(TwinPass3D, step 4 of the processing chain, after transfer_labels.py).

Tiling
------
  * Axis = the pass-1 trajectory (Data/Zone z/s/1/trj_G_Real.las), which is
    already in the GT frame because pass 1 is the alignment reference.
      straight (default): one line fitted to the trajectory (robust: points
        > 3 m off the first fit are ignored, so an end hook cannot tilt it).
        Station = projection onto that line, so every cut is perpendicular to
        it and all tiles are parallel rectangles of equal length.
      curved (--curved, default Z1S8): the trajectory resampled every 1 m and
        Gaussian-smoothed (sigma 5 m); station = arc length to the projection
        on the nearest vertex's tangent, so tiles follow the turn.
    Station 0 is at the start of pass 1's drive.  Tiles are L long along the
    axis and keep the full cross-street width.
  * Extent = the station range of the GT's road-class points (0.1-99.9
    percentile, so a stray mislabelled road point cannot stretch it).  Only
    whole tiles are made: n = floor(extent / L), centred in the road extent,
    the < L remainder split evenly between the two ends and discarded.  No tile
    reaches beyond the road.
  * Points of the EXCLUDED classes (not_classified) are dropped from every
    source.  MERGE folds shared_pathway into sidewalk and signal into pole.
    The written classification uses the released TwinPass3D codes 0-10
    (RELEASE_CLASSES below): 0 Road, 1 Other Road, 2 Sidewalk, 3 Walkway,
    4 Building, 5 Fence, 6 Pole, 7 Sign, 8 Vehicle, 9 Low Vegetation,
    10 High Vegetation.
  * Boundaries depend only on the GT road points and the pass-1 trajectory, so
    tile k covers the same stretch of street in gt / pass1 / pass2.  Pass 2's
    raw coordinates are off the GT frame by its alignment warp (up to ~2 m);
    its stations are computed after applying that warp.  Its written
    coordinates stay raw unless --pass2-aligned is given.
    NOTE: --pass2-aligned re-applies the warp to the raw pass-2 coordinates.
    Rounding to the 1 mm grid can then differ from the GT by about 1 mm for a
    fraction of points.  For tiles whose pass-2 coordinates must be bit-exact
    copies of the GT points (as in the released dataset), snap each pass-2
    point to its counterpart in the GT tile after tiling.

Sources (folder layout set in transfer_labels.py)
-------
  gt     ZxSy-annotated/<class>.las  (classification set from the file's class)
  pass1  out/ZxSy/ZxSy_pass1_labeled.las   (from transfer_labels.py)
  pass2  out/ZxSy/ZxSy_pass2_labeled.las

Output
------
  <out>/<source>/Zone<z>/ZxSy/ZxSy_<kkk>.las     all original fields kept,
                                                classification = released code 0-10
  <out>/<source>/Zone<z>/ZxSy/tiles.json         boundaries, counts, class map
  Tile ids match across sources; an empty tile writes no file (listed in json).

    python split_tiles.py --source gt --all
    python split_tiles.py --source pass1 pass2 --all
    python split_tiles.py --source gt pass1 pass2 --segment Z1S10 --tile-len 25
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import laspy
from scipy.spatial import cKDTree

import transfer_labels as T
from alignment import load_state  # alignment.py must sit next to this script

DEFAULT_OUT = T.HERE / "Final_Dataset"
TRJ_STEP = 1.0          # m, curved-mode resampling
TRJ_SIGMA = 5.0         # m, curved-mode smoothing
LINE_INLIER = 3.0       # m, straight-mode robust refit threshold
DEFAULT_CURVED = ("Z1S8",)  # TwinPass3D's only curved segment; change for other data
EXCLUDED = ("not_classified",)          # never written to any tile
EXCLUDED_IDS = np.array([T.CLASS_MAP[n] for n in EXCLUDED], np.uint8)
MERGE = {"shared_pathway": "sidewalk", "signal": "pole"}
# Released class codes: contiguous 0-10, in this order (paper, Table 3).
RELEASE_CLASSES = ("road", "other_road", "sidewalk", "walkway", "building", "fence",
                   "pole", "sign", "vehicle", "low_vegetation", "high_vegetation")
RELEASE_CODE = {n: i for i, n in enumerate(RELEASE_CLASSES)}
# annotation id (transfer_labels.CLASS_MAP) -> released code; 255 = never written
OUT_LUT = np.full(256, 255, dtype=np.uint8)
for _name, _id in T.CLASS_MAP.items():
    if _name not in EXCLUDED:
        OUT_LUT[_id] = RELEASE_CODE[MERGE.get(_name, _name)]
ROAD_PCT = (0.1, 99.9)  # robust road extent


def log(msg: str) -> None:
    print(msg, flush=True)


def _rel(p: Path) -> str:
    """Path relative to the data folder for the manifest, absolute otherwise."""
    try:
        return str(p.relative_to(T.HERE))
    except ValueError:
        return str(p)


def zone_seg(slug: str) -> tuple[str, str]:
    z, s = re.fullmatch(r"Z(\d+)S(\d+)", slug).groups()
    return z, s


# --------------------------------------------------------------------------- #
#  Station along the road centreline
# --------------------------------------------------------------------------- #
class Stationer:
    """Along-street station from the pass-1 trajectory (straight or curved)."""

    def __init__(self, trj: Path, npz: Path, curved: bool):
        frame, _, pw = load_state(npz)
        self.frame, self.pw = frame, pw                    # pass-2 warp only
        t = laspy.read(str(trj))
        order = np.argsort(np.asarray(t.gps_time), kind="stable") \
            if "gps_time" in t.point_format.dimension_names else slice(None)
        xy = np.column_stack([np.asarray(t.x), np.asarray(t.y)])[order]
        self.mode = "curved" if curved else "straight"

        # robust straight line (also reported for curved mode)
        c = xy.mean(axis=0)
        a = self._axis(xy - c)
        for _ in range(3):
            r = np.abs((xy - c) @ np.array([-a[1], a[0]]))
            inl = r <= LINE_INLIER
            c = xy[inl].mean(axis=0)
            a = self._axis(xy[inl] - c)
        if (xy[-1] - xy[0]) @ a < 0:                       # point along the drive
            a = -a
        r = (xy - c) @ np.array([-a[1], a[0]])
        self.info = dict(axis_mode=self.mode, trajectory=_rel(trj),
                         trajectory_points=int(len(xy)),
                         line_origin=c.tolist(), line_direction=a.tolist(),
                         trajectory_max_offset_from_line_m=float(np.abs(r).max()),
                         trajectory_inlier_fraction=float(inl.mean()))
        if not curved:
            self.c, self.a = c, a
            self.s0 = float(((xy - c) @ a).min())
            return

        # curved: arc-length resample + Gaussian smoothing of the trajectory
        from scipy.ndimage import gaussian_filter1d
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 0]
        arc, xy = arc[keep], xy[keep]
        q = np.arange(0.0, arc[-1], TRJ_STEP)
        pts = np.column_stack([np.interp(q, arc, xy[:, 0]), np.interp(q, arc, xy[:, 1])])
        pts = gaussian_filter1d(pts, TRJ_SIGMA / TRJ_STEP, axis=0, mode="nearest")
        seg = np.diff(pts, axis=0)
        L = np.linalg.norm(seg, axis=1)
        self.xy = pts
        self.station = np.r_[0.0, np.cumsum(L)]
        tg = np.vstack([seg, seg[-1:]])
        self.tan = tg / np.linalg.norm(tg, axis=1, keepdims=True)
        self.tree = cKDTree(pts)

    @staticmethod
    def _axis(d: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(np.cov(d.T))
        return v[:, np.argmax(w)]

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        if self.mode == "straight":
            return (xy - self.c) @ self.a - self.s0
        _, i = self.tree.query(xy, workers=-1)
        return self.station[i] + np.einsum("ij,ij->i", xy - self.xy[i], self.tan[i])

    def warp_pass2(self, xyz: np.ndarray) -> np.ndarray:
        """Raw pass-2 UTM -> GT (pass-1) frame, as alignment.py's merge does."""
        loc = self.frame.to_local(xyz, dtype=np.float32)
        return self.frame.to_global(self.pw.apply(loc))


def tile_edges(road_station: np.ndarray, L: float) -> tuple[np.ndarray, dict]:
    lo, hi = np.percentile(road_station, ROAD_PCT)
    n = int(np.floor((hi - lo) / L))
    pad = ((hi - lo) - n * L) / 2
    edges = lo + pad + L * np.arange(n + 1)
    info = dict(road_station_min=float(road_station.min()),
                road_station_max=float(road_station.max()),
                road_extent=[float(lo), float(hi)], road_length=float(hi - lo),
                n_tiles=n, remainder_each_end=float(pad))
    return edges, info


def gt_road_stations(ann_dir: Path, st: Stationer) -> np.ndarray:
    road = [p for p in ann_dir.glob("*.las") if T.class_name(p) == "road"]
    if not road:
        raise SystemExit(f"{ann_dir.name}: no road class file")
    las = laspy.read(str(road[0]))
    return st(np.column_stack([np.asarray(las.x), np.asarray(las.y)]))


# --------------------------------------------------------------------------- #
#  Tile writers
# --------------------------------------------------------------------------- #
class TileWriters:
    """One lazily-opened LAS writer per tile, sharing one header."""

    def __init__(self, out_dir: Path, slug: str, header, n: int):
        self.out_dir, self.slug, self.header = out_dir, slug, header
        self.w: dict[int, laspy.LasWriter] = {}
        self.counts = np.zeros(n, np.int64)
        self.per_class = np.zeros((n, 32), np.int64)

    def path(self, k: int) -> Path:
        return self.out_dir / f"{self.slug}_{k:03d}.las"

    def write(self, k: int, rec) -> None:
        if k not in self.w:
            self.w[k] = laspy.open(str(self.path(k)), mode="w", header=self.header)
        self.w[k].write_points(rec)
        self.counts[k] += len(rec)
        self.per_class[k] += np.bincount(np.asarray(rec.classification),
                                         minlength=32)[:32]

    def close(self) -> None:
        for w in self.w.values():
            w.close()


def new_header(src_hdr, off_mm: np.ndarray):
    h = laspy.LasHeader(point_format=src_hdr.point_format, version=src_hdr.version)
    h.scales = np.array([0.001, 0.001, 0.001])
    h.offsets = off_mm * 0.001
    return h


def route(writers: TileWriters, rec, station: np.ndarray, edges: np.ndarray) -> int:
    """Send each point to its tile; returns how many fell outside all tiles."""
    k = np.searchsorted(edges, station, side="right") - 1
    inside = (k >= 0) & (k < len(edges) - 1)
    if not inside.any():
        return len(k)
    order = np.flatnonzero(inside)
    order = order[np.argsort(k[order], kind="stable")]
    ks = k[order]
    cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    for a, b in zip(cuts, np.r_[cuts[1:], len(order)]):
        writers.write(int(ks[a]), rec[order[a:b]])
    return int((~inside).sum())


def split_gt(seg: dict, st: Stationer, edges: np.ndarray, out_dir: Path) -> TileWriters:
    files = sorted(p for p in seg["ann_dir"].glob("*.las")
                   if T.class_name(p) not in EXCLUDED)
    with laspy.open(str(files[0])) as f:
        hdr0 = f.header
    offs = []
    for p in files:
        with laspy.open(str(p)) as f:
            offs.append(T.offset_mm(f.header))
    off_out = np.min(offs, axis=0)
    hdr = new_header(hdr0, off_out)
    tw = TileWriters(out_dir, seg["slug"], hdr, len(edges) - 1)
    tw.outside = 0
    tw.excluded = 0
    for p in seg["ann_dir"].glob("*.las"):
        if T.class_name(p) in EXCLUDED:
            with laspy.open(str(p)) as f:
                tw.excluded += f.header.point_count
    for p in files:
        cid = OUT_LUT[T.CLASS_MAP[T.class_name(p)]]
        if cid == 255:
            raise SystemExit(f"{p}: class has no released code")
        with laspy.open(str(p)) as f:
            if f.header.point_format.id != hdr.point_format.id:
                raise SystemExit(f"{p}: point format differs from {files[0].name}")
            d_off = T.offset_mm(f.header) - off_out
            for ch in f.chunk_iterator(T.CHUNK):
                station = st(np.column_stack([np.asarray(ch.x), np.asarray(ch.y)]))
                rec = laspy.ScaleAwarePointRecord(ch.array.copy(), ch.point_format,
                                                  hdr.scales, hdr.offsets)
                rec["X"] = np.asarray(ch.X, np.int64) + d_off[0]   # exact mm shift
                rec["Y"] = np.asarray(ch.Y, np.int64) + d_off[1]
                rec["Z"] = np.asarray(ch.Z, np.int64) + d_off[2]
                rec.classification = np.full(len(ch), cid, np.uint8)
                tw.outside += route(tw, rec, station, edges)
    return tw


def split_pass(path: Path, slug: str, p: int, st: Stationer, edges: np.ndarray,
               out_dir: Path, aligned: bool) -> TileWriters:
    with laspy.open(str(path)) as f:
        hdr = new_header(f.header, T.offset_mm(f.header))
        tw = TileWriters(out_dir, slug, hdr, len(edges) - 1)
        tw.outside = 0
        tw.excluded = 0
        for ch in f.chunk_iterator(T.CHUNK):
            drop = np.isin(np.asarray(ch.classification), EXCLUDED_IDS)
            if drop.any():
                tw.excluded += int(drop.sum())
                ch = ch[~drop]
                if len(ch) == 0:
                    continue
            xyz = np.column_stack([np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z)])
            if p == 2:
                g = st.warp_pass2(xyz)
                station = st(g[:, :2])
            else:
                station = st(xyz[:, :2])
            rec = laspy.ScaleAwarePointRecord(ch.array.copy(), ch.point_format,
                                              hdr.scales, hdr.offsets)
            cls = OUT_LUT[np.asarray(ch.classification)]
            if (cls == 255).any():
                raise SystemExit(f"{path}: unknown class ids "
                                 f"{np.unique(np.asarray(ch.classification)[cls == 255])}")
            rec.classification = cls
            if p == 2 and aligned:
                off = T.offset_mm(hdr)
                q = np.round(g * 1000).astype(np.int64) - off
                rec["X"], rec["Y"], rec["Z"] = q[:, 0], q[:, 1], q[:, 2]
            tw.outside += route(tw, rec, station, edges)
    return tw


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def process(seg: dict, sources: list[str], a) -> None:
    t0 = time.time()
    slug = seg["slug"]
    z, s = zone_seg(slug)
    npz = T.GT_ROOT / f"Zone{z}_seg{s}" / f"Zone{z}_seg{s}_align.npz"
    log(f"\n== {slug} ==")
    curved = slug.upper() in {x.upper() for x in a.curved}
    st = Stationer(seg["pass1"].parent / "trj_G_Real.las", npz, curved)
    edges, info = tile_edges(gt_road_stations(seg["ann_dir"], st), a.tile_len)
    info.update(st.info)
    log(f"  axis: {st.mode} (pass-1 trajectory, max {st.info['trajectory_max_offset_from_line_m']:.1f} m "
        f"off a straight line)")
    log(f"  road {info['road_length']:.1f} m -> {info['n_tiles']} x {a.tile_len:g} m tiles "
        f"({info['remainder_each_end']:.1f} m left off each end)")
    if info["n_tiles"] == 0:
        log("  road shorter than one tile, skipped")
        return

    for src in sources:
        out_dir = a.out / src / f"Zone{z}" / slug
        if out_dir.exists():
            if not a.force:
                log(f"  {src}: {out_dir} exists, skipped (use --force)")
                continue
            shutil.rmtree(out_dir)
        if src == "gt":
            tmp = out_dir.with_name(out_dir.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            tw = split_gt(seg, st, edges, tmp)
        else:
            p = int(src[-1])
            path = T.OUT_ROOT / slug / f"{slug}_pass{p}_labeled.las"
            rep = T.OUT_ROOT / slug / f"{slug}_transfer_report.json"
            if not path.exists() or not rep.exists():
                log(f"  {src}: {path.name} missing or transfer unfinished, skipped")
                continue
            tmp = out_dir.with_name(out_dir.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            tw = split_pass(path, slug, p, st, edges, tmp, a.pass2_aligned)
        tw.close()

        inv = {v: k for k, v in RELEASE_CODE.items()}
        tiles = []
        for k in range(len(edges) - 1):
            tiles.append(dict(
                id=k, file=tw.path(k).name if tw.counts[k] else None,
                station=[float(edges[k]), float(edges[k + 1])],
                points=int(tw.counts[k]),
                per_class={inv[c]: int(tw.per_class[k, c])
                           for c in range(32) if tw.per_class[k, c] and c in inv}))
        manifest = dict(
            segment=slug, zone=int(z), source=src, tile_len_m=a.tile_len,
            coordinates=("aligned to GT frame" if src == "pass2" and a.pass2_aligned
                         else "GT frame" if src in ("gt", "pass1") else "raw pass 2"),
            pass2_warp=_rel(npz),
            class_map=dict(RELEASE_CODE),
            merged_classes=MERGE, excluded_classes=list(EXCLUDED), **info,
            points_in_tiles=int(tw.counts.sum()), points_outside_tiles=int(tw.outside),
            points_excluded_classes=int(tw.excluded),
            empty_tiles=[t["id"] for t in tiles if not t["points"]], tiles=tiles)
        (tmp / "tiles.json").write_text(json.dumps(manifest, indent=2))
        tmp.rename(out_dir)
        log(f"  {src:<5}: {tw.counts.sum():>12,} pts in {int((tw.counts > 0).sum())} tiles, "
            f"{tw.outside:,} outside the road extent, {tw.excluded:,} not_classified "
            f"dropped  -> {out_dir}")
    log(f"  [{time.time() - t0:.0f} s]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--segment", nargs="+", help="e.g. Z1S10 Z2S3")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--source", nargs="+", required=True,
                    choices=("gt", "pass1", "pass2"))
    ap.add_argument("--tile-len", type=float, default=25.0, help="metres along the road")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pass2-aligned", action="store_true",
                    help="write pass-2 tiles in the GT frame instead of raw coordinates")
    ap.add_argument("--curved", nargs="*", default=list(DEFAULT_CURVED),
                    help="segments whose tiles follow the curved trajectory instead of "
                         "a straight line (default: Z1S8; pass --curved alone for none)")
    ap.add_argument("--force", action="store_true", help="overwrite existing tile folders")
    a = ap.parse_args(argv)

    segs = T.discover()
    if a.segment:
        want = {x.upper() for x in a.segment}
        segs = [s for s in segs if s["slug"].upper() in want]
        if len(segs) != len(want):
            sys.exit("unknown segment(s); see: python transfer_labels.py --list")
    for s in segs:
        process(s, a.source, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
