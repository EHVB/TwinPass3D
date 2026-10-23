#!/usr/bin/env python3
"""
vehicle_persistence.py -- keep vehicles seen in BOTH passes, drop the rest
from pass 1, pass 2 and the GT (TwinPass3D, step 5 of the processing chain,
after split_tiles.py).

Inputs are read-only: the tiles written by split_tiles.py,
  <DATA>/{gt,pass1,pass2}/Zone<z>/ZxSy/ZxSy_kkk.las      (DATA set below)
Review output goes under this folder (review/, apply_reports/); the cleaned
dataset is written to <FINAL> (set below) in the release layout.

Requirements: numpy, scipy, laspy, shapely (>= 2.0), matplotlib, Pillow.

Frames
------
  pass1  GT frame (pass 1 is the alignment reference; its points appear in the
         GT bit-exactly).
  gt     pass 1  U  warped pass 2, shuffled, no pass_id.  The points of a GT
         tile that do not coincide (to the mm) with a pass-1 point are pass 2
         in the GT frame -- called "GT-B" below.
  pass2  RAW pass-2 coordinates (tiles.json: "coordinates": "raw pass 2"),
         off the GT frame by the alignment warp (up to a few m).  The
         comparison therefore does not use the alignment warp: it is done on
         GT-B instead, and removals are carried back to raw pass 2 by
         registering each raw tile onto GT-B (same points, so the
         registration residual is ~mm and is reported).

Persistence test (all in the GT frame, bird's-eye view)
------------------------------------------------------
  1. Vehicle points (class 8) of pass 1 and of GT-B are clustered per segment
     (voxel connected components, so a car cut by a tile boundary is one
     object).
  2. Each cluster gets a minimum-area oriented 2-D bounding box.
  3. A cluster is persistent if EITHER
       (a) coverage: >= thr of its box lies inside the union of the other
           pass's vehicle boxes (a wheel or panel split off a car still sits
           inside that car's box in the other pass), OR
       (b) it pairs with an other-pass box with overlap = area(A n B) /
           min(area A, area B) >= thr (intersection over the smaller box, so
           a car seen from one side in one pass and fully in the other still
           matches).  A pair only counts when the smaller box is >= 25 % of
           the larger (a stray fragment cannot vouch for a whole car); (b) is
           symmetric, so a matched pair is kept in both passes, OR
       (c) grid coverage: >= thr of its 0.2 m bird's-eye cells lie within
           0.4 m of the other pass's vehicle cells (no boxes involved, so it
           is immune to the min-area box flipping orientation on an
           L-shaped partial view; a car that moved > ~1 m fails it).
     Only vehicle-sized clusters are judged, drawn and used as support: box
     >= 1.5 m long, >= 1.0 m², points spanning >= 0.5 m in height, >= 15 pts.
     A smaller cluster within 1 m of a vehicle of its own pass is a piece of it
     (wheel, mirror, edge) and follows its decision ("fragment_of"); any other
     small cluster is left untouched ("not_vehicle").  Support only
     counts from other-pass vehicles that are themselves kept; decisions are
     iterated until nothing changes.
     best_overlap = max of (a), (b), (c).
  4. A cluster whose best overlap is below --thr (default 0.7) is a candidate
     for removal in its pass and in the GT.  0.7 sits in the gap observed over
     all 21 segments: every genuine one-pass vehicle scored <= 0.57, while
     sparse, distant cars present in both passes scored 0.78-0.90 (0.9 would
     remove them, and then -- through the consistency rule -- their twin).
     decisions.json keeps initial_best_overlap (vs every other-pass vehicle)
     next to best_overlap (vs kept ones only, after convergence).

Workflow
--------
  python vehicle_persistence.py --list
  python vehicle_persistence.py --segment Z1S10            # review: figures + decisions, writes no LAS
  python vehicle_persistence.py --all                      # review every segment
  #   inspect review/<seg>/  ;  edit review/<seg>/decisions.json ("remove": true/false) to override
  python vehicle_persistence.py --segment Z1S10 --apply --dry-run   # counts + pass-2 mapping check, no LAS
  python vehicle_persistence.py --all --apply                        # write <FINAL>/

Review output   review/<seg>/
  overview_<k>.png            plan view strips, both passes, every box (green kept / red removed)
  pass1_c<id>.png, pass2_c<id>.png
                              one figure per removal candidate: plan view + side slab,
                              candidate pass next to the other pass, same window
  decisions.json              every cluster, score, partner, decision (editable)
  clusters.npz                mm keys of every cluster's points (used by --apply)

Apply output    <FINAL>/Zone <z>/{TwinPass,Pass 1,Pass 2}/<seg>/<seg>_kkk.las  (tiles only)
                apply_reports/summary.json ; apply_reports/<seg>.json (per tile)
  A complete dataset: tiles that lose points are rewritten, every other tile is
  copied (as a copy-on-write clone on macOS/APFS, a plain copy elsewhere).
  tiles.json is not copied.
  Pass 2 keeps raw pass-2 coordinates, as in the source tiles.
  Only Vehicle (class 8) points are ever removed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import laspy
from scipy.sparse import coo_matrix
from scipy.ndimage import binary_dilation
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

# ---- folder layout: edit these if your data live elsewhere --------------------
HERE = Path(__file__).resolve().parent
DATA = HERE / "Final_Dataset"   # split_tiles.py --out (default), read-only here
SOURCES = ("gt", "pass1", "pass2")
VEHICLE = 8              # released class code of Vehicle (split_tiles.RELEASE_CLASSES)

# clustering
VOXEL = 0.15            # m, voxel size for connected components
LINK = 0.45             # m, voxel centres closer than this are connected
MIN_PTS = 15            # fewer points than this: never a vehicle
MIN_VEH_LEN = 1.5       # m, a vehicle's box is at least this long ...
MIN_VEH_AREA = 1.0      # m², ... covers at least this area ...
MIN_VEH_HEIGHT = 0.5    # m, ... and its points span at least this height
MIN_AREA_RATIO = 0.25   # a matching pair's smaller box must be >= this fraction of the larger
GRID_CELL = 0.2         # m, BEV occupancy cell
GRID_DILATE = 0.4       # m, tolerance around the other pass's vehicle cells
FRAG_DIST = 1.0         # m, a non-vehicle cluster this close to a vehicle is a piece of it
# figures
CONTEXT_FRAC = 0.12     # fraction of non-vehicle points kept for figure context
WIN_PAD = 7.0           # m, plan-view window margin around the candidate box
SLAB_PAD = 1.0          # m, side-view slab half-width beyond the box (cross-street)
OVERVIEW_LEN = 125.0    # m of street per overview figure
# pass-2 mapping
MATCH_TOL = 0.03        # m, raw-pass-2 point <-> GT-B point identity tolerance


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
#  Segments, tiles, keys
# --------------------------------------------------------------------------- #
def segments() -> dict[str, str]:
    """slug -> 'Zone<z>' for segments present in all three sources."""
    out = {}
    for d in sorted((DATA / "pass1").glob("Zone*/Z*S*")):
        if all((DATA / s / d.parent.name / d.name).is_dir() for s in SOURCES):
            out[d.name] = d.parent.name
    return dict(sorted(out.items(), key=lambda kv: [int(x) for x in re.findall(r"\d+", kv[0])]))


def seg_dir(src: str, slug: str) -> Path:
    return DATA / src / segments()[slug] / slug


def tile_files(slug: str) -> list[str]:
    names = [sorted(p.name for p in seg_dir(s, slug).glob(f"{slug}_*.las")) for s in SOURCES]
    common = sorted(set(names[0]) & set(names[1]) & set(names[2]))
    if len(common) != len(names[1]):
        log(f"  warning: tile sets differ between sources, using {len(common)} common tiles")
    return common


def mm_xyz(las) -> np.ndarray:
    """Exact integer mm coordinates (scale is 0.001 in every source)."""
    if not np.allclose(las.header.scales, 0.001):
        raise SystemExit(f"unexpected LAS scale {las.header.scales}")
    off = np.round(np.asarray(las.header.offsets) / 0.001).astype(np.int64)
    return np.column_stack([np.asarray(las.X, np.int64) + off[0],
                            np.asarray(las.Y, np.int64) + off[1],
                            np.asarray(las.Z, np.int64) + off[2]])


class Keyer:
    """Collision-free int64 key from mm coordinates (21 bits per axis)."""

    def __init__(self, slug: str):
        lo = np.full(3, np.iinfo(np.int64).max)
        hi = np.full(3, np.iinfo(np.int64).min)
        for f in tile_files(slug):
            with laspy.open(str(seg_dir("gt", slug) / f)) as r:
                lo = np.minimum(lo, np.floor(np.asarray(r.header.mins) * 1000).astype(np.int64) - 10)
                hi = np.maximum(hi, np.ceil(np.asarray(r.header.maxs) * 1000).astype(np.int64) + 10)
        if np.any(hi - lo >= 2 ** 21):
            raise SystemExit(f"{slug}: extent {(hi - lo) / 1000} m too large for 21-bit keys")
        self.lo = lo

    def __call__(self, mm: np.ndarray) -> np.ndarray:
        d = mm - self.lo
        if d.min() < 0 or d.max() >= 2 ** 21:
            raise ValueError("point outside key range")
        return (d[:, 0] << 42) | (d[:, 1] << 21) | d[:, 2]


class Street:
    """(along, cross) display frame from the tiling line in tiles.json."""

    def __init__(self, slug: str):
        meta = json.loads((seg_dir("pass1", slug) / "tiles.json").read_text())
        self.o = np.asarray(meta["line_origin"])
        self.a = np.asarray(meta["line_direction"]) / np.linalg.norm(meta["line_direction"])
        self.n = np.array([-self.a[1], self.a[0]])
        self.mode = meta.get("axis_mode", "straight")

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        d = xy - self.o
        return np.column_stack([d @ self.a, d @ self.n])


# --------------------------------------------------------------------------- #
#  Clustering and boxes
# --------------------------------------------------------------------------- #
def cluster(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.zeros(0, np.int64)
    vox = np.floor(xyz / VOXEL).astype(np.int64)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.ravel()
    pairs = cKDTree((uniq + 0.5) * VOXEL).query_pairs(LINK, output_type="ndarray")
    n = len(uniq)
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    # relabel by size, largest first
    lab = comp[inv]
    _, first, counts = np.unique(lab, return_index=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[lab]


def _box_dims(box: Polygon) -> tuple[float, float]:
    c = np.asarray(box.exterior.coords)
    e = np.linalg.norm(np.diff(c[:3], axis=0), axis=1)
    return float(e.max()), float(e.min())


def obox(xy: np.ndarray) -> Polygon:
    g = MultiPoint(xy).convex_hull
    if g.geom_type != "Polygon" or g.area < 1e-4:
        return g.buffer(0.1).minimum_rotated_rectangle
    return g.minimum_rotated_rectangle


# --------------------------------------------------------------------------- #
#  Load one segment
# --------------------------------------------------------------------------- #
class Segment:
    def __init__(self, slug: str):
        self.slug = slug
        self.files = tile_files(slug)
        self.key = Keyer(slug)
        self.street = Street(slug)
        self.origin = self.key.lo / 1000.0               # local float frame origin (m)

    def load(self) -> None:
        """Vehicle points of pass 1 and GT-B (with keys), plus figure context."""
        rng = np.random.default_rng(0)
        veh = {"pass1": [], "pass2": []}
        ctx = {"pass1": [], "pass2": []}
        self.n_gt_b = 0
        for t, f in enumerate(self.files):
            t0 = time.time()
            a = laspy.read(str(seg_dir("pass1", self.slug) / f))
            g = laspy.read(str(seg_dir("gt", self.slug) / f))
            ka = self.key(mm_xyz(a))
            gmm = mm_xyz(g)
            kg = self.key(gmm)
            in_a = np.isin(kg, ka)
            gc = np.asarray(g.classification)
            ac = np.asarray(a.classification)
            amm = mm_xyz(a)
            self.n_gt_b += int((~in_a).sum())
            for name, mm, k, c in (("pass1", amm, ka, ac),
                                   ("pass2", gmm[~in_a], kg[~in_a], gc[~in_a])):
                v = c == VEHICLE
                loc = (mm[v] - self.key.lo) / 1000.0
                veh[name].append((loc.astype(np.float32), k[v], np.full(v.sum(), t, np.int16)))
                nv = np.flatnonzero(~v)
                pick = nv[rng.random(len(nv)) < CONTEXT_FRAC]
                ctx[name].append(np.column_stack([(mm[pick] - self.key.lo) / 1000.0]).astype(np.float32))
            log(f"    {f}: pass1 {len(a.points):,}  GT {len(g.points):,}  "
                f"GT-B {(~in_a).sum():,}  ({time.time() - t0:.1f}s)")
            del a, g
        self.veh = {}
        for name in veh:
            xyz = np.concatenate([x[0] for x in veh[name]])
            keys = np.concatenate([x[1] for x in veh[name]])
            tile = np.concatenate([x[2] for x in veh[name]])
            self.veh[name] = dict(xyz=xyz, key=keys, tile=tile)
        self.ctx = {n: np.concatenate(ctx[n]) for n in ctx}

    # ------------------------------------------------------------------ #
    def decide(self, thr: float) -> list[dict]:
        boxes, recs = {}, {}
        for name in ("pass1", "pass2"):
            v = self.veh[name]
            lab = cluster(v["xyz"])
            v["label"] = lab
            n = int(lab.max()) + 1 if len(lab) else 0
            order = np.argsort(lab, kind="stable")
            bounds = np.searchsorted(lab[order], np.arange(n + 1))
            recs[name], boxes[name] = [], []
            for c in range(n):
                idx = order[bounds[c]:bounds[c + 1]]
                p = v["xyz"][idx].astype(np.float64)
                box = obox(p[:, :2])
                boxes[name].append(box)
                recs[name].append(dict(
                    id=c, pass_=name, n_points=int(len(idx)),
                    tiles=sorted({int(x) for x in v["tile"][idx]}),
                    centroid_local=p.mean(0).round(3).tolist(),
                    z_range=[round(float(p[:, 2].min()), 3), round(float(p[:, 2].max()), 3)],
                    box_local=np.round(np.asarray(box.exterior.coords)[:-1], 3).tolist(),
                    box_area_m2=round(box.area, 3),
                    box_length_m=round(_box_dims(box)[0], 3), box_width_m=round(_box_dims(box)[1], 3),
                    height_m=round(float(p[:, 2].max() - p[:, 2].min()), 3),
                    best_overlap=0.0, best_iou=0.0, partner=None))
        # only vehicle-sized clusters are judged, drawn and used as support
        real = {n: np.array([r["n_points"] >= MIN_PTS and r["box_length_m"] >= MIN_VEH_LEN and
                             r["box_area_m2"] >= MIN_VEH_AREA and r["height_m"] >= MIN_VEH_HEIGHT
                             for r in recs[n]], bool) for n in recs}
        self.boxes = boxes
        self._grid_setup()

        # fragments: a non-vehicle cluster within FRAG_DIST of a vehicle of its
        # own pass is a piece of it (wheel, mirror, edge) and follows its decision;
        # any other non-vehicle cluster is left untouched
        host = {}
        for name in recs:
            big = [i for i in range(len(recs[name])) if real[name][i]]
            bt = STRtree([boxes[name][i] for i in big]) if big else None
            host[name] = np.full(len(recs[name]), -1)
            for i, r in enumerate(recs[name]):
                if bt is None or real[name][i]:
                    continue
                near = bt.query(boxes[name][i], predicate="dwithin", distance=FRAG_DIST)
                if len(near):
                    host[name][i] = big[int(min(near, key=lambda k: boxes[name][big[k]].distance(boxes[name][i])))]

        # support only counts from other-pass vehicles that are themselves kept;
        # removals can only withdraw support, so this converges
        keep = {n: np.ones(len(recs[n]), bool) for n in recs}
        for it in range(20):
            scores = {n: self._scores(n, ("pass2" if n == "pass1" else "pass1"),
                                      keep["pass2" if n == "pass1" else "pass1"]
                                      & real["pass2" if n == "pass1" else "pass1"], real[n])
                      for n in recs}
            if it == 0:
                first = {n: scores[n]["best"].copy() for n in recs}
            new_keep = {}
            for name in recs:
                k = ~real[name] | (scores[name]["best"] >= thr)
                f = host[name] >= 0
                k[f] = k[host[name][f]]
                new_keep[name] = k
            changed = any((new_keep[n] != keep[n]).any() for n in recs)
            keep = new_keep
            if not changed:
                break
        self.iterations = it + 1

        out = []
        for name in ("pass1", "pass2"):
            s = scores[name]
            for i, r in enumerate(recs[name]):
                r.update(coverage=round(float(s["cover"][i]), 4), pair_overlap=round(float(s["pair"][i]), 4),
                         grid_coverage=round(float(s["grid"][i]), 4), best_overlap=round(float(s["best"][i]), 4),
                         best_iou=round(float(s["iou"][i]), 4),
                         partner=None if s["partner"][i] < 0 else int(s["partner"][i]),
                         initial_best_overlap=round(float(first[name][i]), 4))
                r["is_vehicle"] = bool(real[name][i])
                if host[name][i] >= 0:
                    r["fragment_of"] = int(host[name][i])
                if real[name][i] or host[name][i] >= 0:
                    r.update(decision="keep" if keep[name][i] else "remove", remove=not keep[name][i])
                else:
                    r.update(decision="not_vehicle", remove=False)
                r["pass"] = r.pop("pass_")
                out.append(r)
        return out

    # ------------------------------------------------------------------ #
    def _grid_setup(self) -> None:
        names = ("pass1", "pass2")
        self.ij = {n: np.floor(self.veh[n]["xyz"][:, :2] / GRID_CELL).astype(np.int64) for n in names}
        self.shape = tuple(int(max((self.ij[n][:, k].max() if len(self.ij[n]) else 0) for n in names)) + 1
                           for k in (0, 1))
        r_c = int(round(GRID_DILATE / GRID_CELL))
        yy, xx = np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]
        self.disk = xx ** 2 + yy ** 2 <= r_c ** 2

    def _scores(self, name: str, other: str, elig: np.ndarray, real_me: np.ndarray) -> dict:
        """Scores of every `name` cluster against the eligible `other` clusters."""
        B, O = self.boxes[name], self.boxes[other]
        n = len(B)
        cover, pair, grid, iou = (np.zeros(n) for _ in range(4))
        partner = np.full(n, -1)
        pidx = [j for j in range(len(O)) if elig[j]]
        tree = STRtree([O[j] for j in pidx]) if pidx else None
        for i, b in enumerate(B):
            if not real_me[i] or tree is None:
                continue
            hits = [pidx[k] for k in tree.query(b)]
            if not hits:
                continue
            # (a) box cover: share of this box inside the other pass's boxes
            cover[i] = b.intersection(unary_union([O[j] for j in hits])).area / b.area
            partner[i] = max(hits, key=lambda j: b.intersection(O[j]).area)
            # (b) comparable-size pair, intersection over the smaller box
            for j in hits:
                bb = O[j]
                if min(b.area, bb.area) < MIN_AREA_RATIO * max(b.area, bb.area):
                    continue
                inter = b.intersection(bb).area
                ios = inter / max(min(b.area, bb.area), 1e-9)
                if ios > pair[i]:
                    pair[i], iou[i] = ios, inter / max(b.union(bb).area, 1e-9)
                    partner[i] = j
        # (c) BEV occupancy grid, immune to box orientation
        lab_o = self.veh[other]["label"]
        sel = elig[lab_o] if len(lab_o) else np.zeros(0, bool)
        occ = np.zeros(self.shape, bool)
        occ[self.ij[other][sel, 0], self.ij[other][sel, 1]] = True
        occ = binary_dilation(occ, structure=self.disk)
        lab = self.veh[name]["label"]
        if len(lab):
            N = self.shape[0] * self.shape[1]
            u = np.unique(lab.astype(np.int64) * N + self.ij[name][:, 0] * self.shape[1] + self.ij[name][:, 1])
            ul = u // N
            n_cells = np.bincount(ul, minlength=n)
            n_hit = np.bincount(ul, weights=occ.ravel()[u % N], minlength=n)
            grid = np.where(real_me, n_hit / np.maximum(n_cells, 1), 0.0)
        return dict(cover=cover, pair=pair, grid=grid, iou=iou, partner=partner,
                    best=np.maximum(np.maximum(cover, pair), grid))


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


PASS_LABEL = {"pass1": "Pass 1", "pass2": "Pass 2 (aligned, from GT)"}
C_REMOVE, C_KEEP, C_CAND = "#d62728", "#2ca02c", "#d62728"


def _to_street(seg: Segment, xyz_local: np.ndarray) -> np.ndarray:
    return seg.street(xyz_local[:, :2] + seg.origin[:2])


def _box_street(seg: Segment, box_local) -> np.ndarray:
    b = np.asarray(box_local)
    b = np.vstack([b, b[:1]])
    return seg.street(b + seg.origin[:2])


def figure_candidate(seg: Segment, rec: dict, recs: dict, thr: float, path: Path) -> None:
    plt = _plt()
    me, other = rec["pass"], ("pass2" if rec["pass"] == "pass1" else "pass1")
    bx = _box_street(seg, rec["box_local"])
    (a0, c0), (a1, c1) = bx.min(0), bx.max(0)
    ca, cc = (a0 + a1) / 2, (c0 + c1) / 2
    half = max(a1 - a0, c1 - c0) / 2 + WIN_PAD
    win = (ca - half, ca + half, cc - half, cc + half)
    zr = rec["z_range"]
    zlo, zhi = zr[0] - 2.5, zr[1] + 2.0

    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5),
                           gridspec_kw=dict(height_ratios=[1.0, 0.55]))
    for col, name in enumerate((me, other)):
        # ---- context
        cx = seg.ctx[name]
        s = _to_street(seg, cx)
        m = (s[:, 0] > win[0]) & (s[:, 0] < win[1]) & (s[:, 1] > win[2]) & (s[:, 1] < win[3])
        z = cx[m, 2]
        zc = np.clip((z - zr[0] + 0.5) / 4.0, 0, 1)
        ax[0, col].scatter(s[m, 0], s[m, 1], c=zc, cmap="Greys", vmin=-0.3, vmax=1.3,
                           s=0.4, lw=0, rasterized=True)
        slab = m & (s[:, 1] > c0 - SLAB_PAD) & (s[:, 1] < c1 + SLAB_PAD)
        ax[1, col].scatter(s[slab, 0], cx[slab, 2], c="0.6", s=0.6, lw=0, rasterized=True)
        # ---- vehicles
        v = seg.veh[name]
        sv = _to_street(seg, v["xyz"])
        mv = (sv[:, 0] > win[0]) & (sv[:, 0] < win[1]) & (sv[:, 1] > win[2]) & (sv[:, 1] < win[3])
        lab = v["label"]
        rm = np.array([recs[name][c]["remove"] for c in range(len(recs[name]))], bool)
        is_rm = rm[lab] if len(lab) else np.zeros(0, bool)
        mine = [rec["id"]] + [r["id"] for r in recs[me] if r.get("fragment_of") == rec["id"]]
        is_me = np.isin(lab, mine) if name == me else np.zeros(len(lab), bool)
        nv = np.array([r["decision"] == "not_vehicle" for r in recs[name]], bool)
        is_nv = nv[lab] if len(lab) else np.zeros(0, bool)
        for sel, color, lab_txt in ((mv & is_nv, "#8fb3d9", "small non-vehicle cluster - untouched"),
                                    (mv & ~is_rm & ~is_nv, C_KEEP, "vehicle - kept"),
                                    (mv & is_rm & ~is_me, "#ff9f40", "vehicle - other removal"),
                                    (mv & is_me, C_CAND, "vehicle - THIS candidate")):
            if sel.any():
                ax[0, col].scatter(sv[sel, 0], sv[sel, 1], c=color, s=1.2, lw=0,
                                   rasterized=True, label=lab_txt)
                ss = sel & (sv[:, 1] > c0 - SLAB_PAD) & (sv[:, 1] < c1 + SLAB_PAD)
                ax[1, col].scatter(sv[ss, 0], v["xyz"][ss, 2], c=color, s=1.5, lw=0, rasterized=True)
        # ---- boxes of this pass
        for r in recs[name]:
            if not r["is_vehicle"]:
                continue
            b = _box_street(seg, r["box_local"])
            if b[:, 0].max() < win[0] or b[:, 0].min() > win[1] or \
               b[:, 1].max() < win[2] or b[:, 1].min() > win[3]:
                continue
            ax[0, col].plot(b[:, 0], b[:, 1], color=C_REMOVE if r["remove"] else C_KEEP,
                            lw=0.9, alpha=0.8)
        # ---- the candidate box, solid in its pass, dashed in the other
        ls = "-" if name == me else "--"
        ax[0, col].plot(bx[:, 0], bx[:, 1], color=C_CAND, lw=2.2, ls=ls)
        for x in (a0, a1):
            ax[1, col].axvline(x, color=C_CAND, lw=1.2, ls=ls)
        nveh = int((mv & (np.abs(sv[:, 1] - cc) < (c1 - c0) / 2 + 0.3) &
                    (sv[:, 0] > a0) & (sv[:, 0] < a1)).sum())
        ttl = (f"{PASS_LABEL[name]}  -  candidate {me} #{rec['id']}" if name == me else
               f"{PASS_LABEL[name]}  -  same footprint ({nveh:,} vehicle pts inside box)")
        ax[0, col].set_title(ttl, fontsize=11, color=C_CAND if name == me else "k")
        ax[0, col].set_xlim(win[0], win[1])
        ax[0, col].set_ylim(win[2], win[3])
        ax[0, col].set_aspect("equal")
        ax[0, col].set_xlabel("along street (m)")
        ax[0, col].set_ylabel("across street (m)")
        ax[1, col].set_xlim(win[0], win[1])
        ax[1, col].set_ylim(zlo, zhi)
        ax[1, col].set_xlabel("along street (m)")
        ax[1, col].set_ylabel("z (m)")
        ax[1, col].set_title(f"side view, slab {c1 - c0 + 2 * SLAB_PAD:.1f} m wide through the box",
                             fontsize=9)
    h, l = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[0, 1].get_legend_handles_labels()
    seen = dict(zip(l + l2, h + h2))
    fig.legend([seen[k] for k in seen], list(seen), loc="lower center", ncol=4, markerscale=8,
               frameon=False, fontsize=9)
    partner = (f"box cover {rec.get('coverage', 0):.2f}, box pair {rec.get('pair_overlap', 0):.2f}, "
               f"grid {rec.get('grid_coverage', 0):.2f} vs {other}")
    fig.suptitle(f"{seg.slug}  |  {me} vehicle #{rec['id']}  ->  REMOVE  |  {rec['n_points']:,} pts, "
                 f"box {rec['box_area_m2']:.1f} m²  |  {partner}  (threshold {thr:.2f})  |  "
                 f"tiles {', '.join(f'{t:03d}' for t in rec['tiles_files'])}",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_overview(seg: Segment, recs: dict, thr: float, outdir: Path) -> list[Path]:
    plt = _plt()
    s_all = {n: _to_street(seg, seg.ctx[n]) for n in ("pass1", "pass2")}
    v_all = {n: _to_street(seg, seg.veh[n]["xyz"]) for n in ("pass1", "pass2")}
    lo = min(s_all["pass1"][:, 0].min(), s_all["pass2"][:, 0].min())
    hi = max(s_all["pass1"][:, 0].max(), s_all["pass2"][:, 0].max())
    vc = np.concatenate([v_all["pass1"][:, 1], v_all["pass2"][:, 1]])
    clo, chi = (np.percentile(vc, [0.5, 99.5]) if len(vc) else (-15, 15))
    clo, chi = min(clo - 6, -12), max(chi + 6, 12)
    paths = []
    starts = np.arange(lo, hi, OVERVIEW_LEN)
    for k, st in enumerate(starts):
        en = st + OVERVIEW_LEN
        fig, ax = plt.subplots(2, 1, figsize=(18, 2 * (chi - clo) / OVERVIEW_LEN * 16 + 1.8),
                               sharex=True)
        for row, name in enumerate(("pass1", "pass2")):
            s = s_all[name]
            m = (s[:, 0] >= st) & (s[:, 0] < en) & (s[:, 1] > clo) & (s[:, 1] < chi)
            ax[row].scatter(s[m, 0], s[m, 1], c="0.75", s=0.15, lw=0, rasterized=True)
            v = seg.veh[name]
            sv = v_all[name]
            lab = v["label"]
            rm = np.array([r["remove"] for r in recs[name]], bool)
            mv = (sv[:, 0] >= st) & (sv[:, 0] < en)
            if len(lab):
                for sel, col in ((mv & ~rm[lab], C_KEEP), (mv & rm[lab], C_REMOVE)):
                    ax[row].scatter(sv[sel, 0], sv[sel, 1], c=col, s=0.5, lw=0, rasterized=True)
            for r in recs[name]:
                if not r["is_vehicle"]:
                    continue
                b = _box_street(seg, r["box_local"])
                if b[:, 0].max() < st or b[:, 0].min() > en:
                    continue
                col = C_REMOVE if r["remove"] else C_KEEP
                ax[row].plot(b[:, 0], b[:, 1], color=col, lw=0.9)
                ax[row].text(b[:, 0].mean(), b[:, 1].max() + 0.4, f"{r['id']}", fontsize=6.5,
                             color=col, ha="center", va="bottom", clip_on=True)
            nk = sum(1 for r in recs[name] if r["decision"] == "keep")
            nr = sum(1 for r in recs[name] if r["remove"])
            ax[row].set_title(f"{PASS_LABEL[name]}  -  green: kept (overlap >= {thr:.2f}),  "
                              f"red: remove  [segment totals: {nk} kept, {nr} remove]",
                              fontsize=10, loc="left")
            ax[row].set_ylim(clo, chi)
            ax[row].set_aspect("equal")
            ax[row].set_ylabel("across (m)")
        ax[1].set_xlim(st, en)
        ax[1].set_xlabel("along street (m)")
        fig.suptitle(f"{seg.slug}  overview {k + 1}/{len(starts)}  ({st:.0f}-{en:.0f} m)",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        p = outdir / f"overview_{k + 1:02d}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
#  Review
# --------------------------------------------------------------------------- #
def review(slug: str, thr: float, no_figs: bool) -> None:
    t0 = time.time()
    log(f"== {slug}: review (threshold {thr})")
    seg = Segment(slug)
    seg.load()
    rec_list = seg.decide(thr)
    recs = {"pass1": [r for r in rec_list if r["pass"] == "pass1"],
            "pass2": [r for r in rec_list if r["pass"] == "pass2"]}
    for r in rec_list:
        r["tiles_files"] = [int(re.search(r"_(\d+)\.las$", seg.files[t]).group(1)) for t in r["tiles"]]
    outdir = HERE / "review" / slug
    if outdir.exists():
        for p in outdir.glob("*.png"):
            p.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    # point keys per cluster, for --apply
    np.savez_compressed(outdir / "clusters.npz",
                        **{f"{n}_key": seg.veh[n]["key"] for n in ("pass1", "pass2")},
                        **{f"{n}_label": seg.veh[n]["label"] for n in ("pass1", "pass2")},
                        **{f"{n}_tile": seg.veh[n]["tile"] for n in ("pass1", "pass2")},
                        key_lo=seg.key.lo, files=np.array(seg.files))

    if not no_figs:
        log("  figures ...")
        figure_overview(seg, recs, thr, outdir)
        for r in rec_list:
            if r["remove"] and r["is_vehicle"]:
                p = outdir / f"{r['pass']}_c{r['id']:03d}.png"
                figure_candidate(seg, r, recs, thr, p)
                r["figure"] = p.name
        for r in rec_list:                  # a removed fragment is drawn in its vehicle's figure
            if r["remove"] and not r["is_vehicle"]:
                r["figure"] = recs[r["pass"]][r["fragment_of"]].get("figure")

    summary = {}
    for n in ("pass1", "pass2"):
        rr = recs[n]
        summary[n] = dict(
            clusters=len(rr),
            vehicles=sum(r["is_vehicle"] for r in rr),
            keep=sum(r["is_vehicle"] and not r["remove"] for r in rr),
            remove=sum(r["is_vehicle"] and r["remove"] for r in rr),
            fragments_removed=sum(r["remove"] and not r["is_vehicle"] for r in rr),
            not_vehicle=sum(r["decision"] == "not_vehicle" for r in rr),
            vehicle_points=int(len(seg.veh[n]["key"])),
            points_to_remove=int(sum(r["n_points"] for r in rr if r["remove"])))
    doc = dict(segment=slug, threshold=thr,
               params=dict(voxel=VOXEL, link=LINK, min_pts=MIN_PTS, min_veh_len=MIN_VEH_LEN,
                           min_veh_area=MIN_VEH_AREA, min_veh_height=MIN_VEH_HEIGHT,
                           min_area_ratio=MIN_AREA_RATIO, grid_cell=GRID_CELL,
                           grid_dilate=GRID_DILATE, frag_dist=FRAG_DIST),
               frame_note="box_local / centroid_local are metres relative to key_lo_m (GT frame)",
               key_lo_m=(seg.key.lo / 1000.0).tolist(), files=seg.files,
               summary=summary, clusters=rec_list)
    (outdir / "decisions.json").write_text(json.dumps(doc, indent=1))
    for n in ("pass1", "pass2"):
        s = summary[n]
        log(f"  {n}: {s['vehicles']} vehicles: keep {s['keep']}, remove {s['remove']} "
            f"(+{s['fragments_removed']} attached fragments; {s['points_to_remove']:,} of "
            f"{s['vehicle_points']:,} vehicle pts)  |  {s['not_vehicle']} small non-vehicle clusters untouched")
    log(f"  decisions converged in {seg.iterations} iteration(s)")
    log(f"  -> {outdir}  ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- #
#  Apply
# --------------------------------------------------------------------------- #
def _kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def _icp(src, tree, dst, R, t, iters, sample=None, rng=None):
    S = src if sample is None or len(src) <= sample else src[rng.choice(len(src), sample, replace=False)]
    for _ in range(iters):
        X = S @ R.T + t
        d, i = tree.query(X, workers=-1)
        m = d <= max(np.percentile(d, 80), 0.003)
        dR, dt = _kabsch(X[m], dst[i[m]])
        R, t = dR @ R, dR @ t + dt
    return R, t


def map_pass2_removals(raw_mm: np.ndarray, raw_cls: np.ndarray, gtb_mm: np.ndarray,
                       gtb_remove: np.ndarray, lo: np.ndarray, rng) -> tuple[np.ndarray, list]:
    """Boolean mask over raw pass-2 points to remove, plus per-region residuals.

    Registers the raw tile onto GT-B (identical points, warped): a rigid fit
    for the tile, then a local rigid refit around each group of GT-B points to
    remove; a raw vehicle point is removed when its nearest GT-B point (after
    the local fit) is within MATCH_TOL and flagged for removal.
    """
    raw = (raw_mm - lo) / 1000.0
    gtb = (gtb_mm - lo) / 1000.0
    out = np.zeros(len(raw), bool)
    if not gtb_remove.any():
        return out, []
    tree = cKDTree(gtb)
    R, t = np.eye(3), gtb.mean(0) - raw.mean(0)
    R, t = _icp(raw, tree, gtb, R, t, 40, 60000, rng)
    # regions: connected groups of GT-B points to remove
    rem = gtb[gtb_remove]
    lab = cluster(rem)
    report = []
    rv = np.flatnonzero(raw_cls == VEHICLE)
    rawT = raw @ R.T + t
    for c in range(lab.max() + 1):
        p = rem[lab == c]
        lo_b, hi_b = p.min(0) - 4.0, p.max(0) + 4.0
        gm = np.all((gtb >= lo_b) & (gtb <= hi_b), axis=1)
        rm_ = np.all((rawT >= lo_b - 0.5) & (rawT <= hi_b + 0.5), axis=1)
        if gm.sum() < 50 or rm_.sum() < 50:
            continue
        gi = np.flatnonzero(gm)
        lt = cKDTree(gtb[gi])
        Rl, tl = _icp(raw[rm_], lt, gtb[gi], R, t, 15)
        cand = rv[rm_[rv]]
        d, j = lt.query(raw[cand] @ Rl.T + tl, workers=-1)
        hit = (d <= MATCH_TOL) & gtb_remove[gi[j]] & ~out[cand]
        out[cand[hit]] = True
        report.append(dict(region=int(c), gtb_points_to_remove=int(len(p)),
                           raw_points_newly_matched=int(hit.sum()),
                           vehicle_residual_mm_p50=round(float(np.median(d)) * 1000, 2) if len(d) else None))
    return out, report


FINAL = HERE / "TwinPass3D"     # release layout written by --apply; must differ from DATA
FINAL_NAME = {"gt": "TwinPass", "pass1": "Pass 1", "pass2": "Pass 2"}


def final_dir(src: str, slug: str) -> Path:
    """Final Dataset/Zone <z>/<TwinPass|Pass 1|Pass 2>/<seg>/"""
    z = re.fullmatch(r"Z(\d+)S\d+", slug).group(1)
    return FINAL / f"Zone {z}" / FINAL_NAME[src] / slug


def clone(src: Path, dst: Path) -> None:
    """APFS copy-on-write clone (no extra disk until edited); plain copy elsewhere."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def write_like(las, keep: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = laspy.LasData(las.header)
    out.points = las.points[keep]
    out.update_header()
    tmp = dst.with_suffix(".las.tmp")
    out.write(str(tmp))
    tmp.replace(dst)


def apply(slug: str, dry: bool) -> dict:
    t0 = time.time()
    rdir = HERE / "review" / slug
    dec_p, npz_p = rdir / "decisions.json", rdir / "clusters.npz"
    if not dec_p.exists() or not npz_p.exists():
        raise SystemExit(f"{slug}: run the review first (missing {rdir})")
    dec = json.loads(dec_p.read_text())
    z = np.load(npz_p)
    lo = z["key_lo"]
    files = [str(f) for f in z["files"]]
    log(f"== {slug}: apply{' (dry run)' if dry else ''}  threshold {dec['threshold']}")
    rm_ids = {n: np.array(sorted(r["id"] for r in dec["clusters"] if r["pass"] == n and r["remove"]),
                          np.int64) for n in ("pass1", "pass2")}
    sel = {n: np.isin(z[f"{n}_label"], rm_ids[n]) for n in ("pass1", "pass2")}
    rm_keys = {n: z[f"{n}_key"][sel[n]] for n in ("pass1", "pass2")}
    hit_tiles = sorted(set(z["pass1_tile"][sel["pass1"]].tolist()) | set(z["pass2_tile"][sel["pass2"]].tolist()))
    log(f"  removing {len(rm_ids['pass1'])} pass-1 and {len(rm_ids['pass2'])} pass-2 clusters "
        f"(vehicles + attached fragments, from decisions.json); {len(hit_tiles)} of {len(files)} tiles affected")
    keyer = Keyer.__new__(Keyer)
    keyer.lo = lo
    rng = np.random.default_rng(0)
    report = dict(segment=slug, threshold=dec["threshold"], dry_run=dry,
                  removed_vehicles={n: [r["id"] for r in dec["clusters"]
                                        if r["pass"] == n and r["remove"] and r.get("is_vehicle", True)]
                                    for n in ("pass1", "pass2")},
                  removed_clusters={n: rm_ids[n].tolist() for n in rm_ids}, tiles=[])
    tot = dict(pass1=0, pass2=0, gt=0, pass2_expected=0)
    for t, f in enumerate(files):
        if t not in hit_tiles:
            if not dry:
                for src in SOURCES:
                    clone(seg_dir(src, slug) / f, final_dir(src, slug) / f)
            continue
        tt = time.time()
        a = laspy.read(str(seg_dir("pass1", slug) / f))
        g = laspy.read(str(seg_dir("gt", slug) / f))
        b = laspy.read(str(seg_dir("pass2", slug) / f))
        ka, gmm = keyer(mm_xyz(a)), mm_xyz(g)
        kg = keyer(gmm)
        ac, gc, bc = (np.asarray(x.classification) for x in (a, g, b))
        rm_a = np.isin(ka, rm_keys["pass1"]) & (ac == VEHICLE)
        rm_g = (np.isin(kg, rm_keys["pass1"]) | np.isin(kg, rm_keys["pass2"])) & (gc == VEHICLE)
        in_a = np.isin(kg, ka)
        gtb_rm = np.isin(kg[~in_a], rm_keys["pass2"]) & (gc[~in_a] == VEHICLE)
        rm_b, regions = map_pass2_removals(mm_xyz(b), bc, gmm[~in_a], gtb_rm, lo, rng)
        rec = dict(tile=f, pass1_removed=int(rm_a.sum()), gt_removed=int(rm_g.sum()),
                   pass2_removed=int(rm_b.sum()), pass2_expected=int(gtb_rm.sum()),
                   pass2_regions=regions)
        report["tiles"].append(rec)
        for k in ("pass1", "gt", "pass2"):
            tot[k] += rec[f"{k}_removed"]
        tot["pass2_expected"] += rec["pass2_expected"]
        flag = "" if abs(rec["pass2_removed"] - rec["pass2_expected"]) <= max(5, 0.005 * rec["pass2_expected"]) \
            else "   <-- pass-2 mapping mismatch"
        log(f"    {f}: pass1 -{rec['pass1_removed']:,}  GT -{rec['gt_removed']:,}  "
            f"pass2 -{rec['pass2_removed']:,} (expected {rec['pass2_expected']:,})  "
            f"({time.time() - tt:.1f}s){flag}")
        if not dry:
            for src, las, rm in (("pass1", a, rm_a), ("gt", g, rm_g), ("pass2", b, rm_b)):
                if rm.any():
                    write_like(las, ~rm, final_dir(src, slug) / f)
                else:
                    clone(seg_dir(src, slug) / f, final_dir(src, slug) / f)
        del a, g, b
    report["totals"] = tot
    rep_dir = HERE / "apply_reports"
    rep_dir.mkdir(exist_ok=True)
    (rep_dir / f"{slug}{'_dryrun' if dry else ''}.json").write_text(json.dumps(report, indent=1))
    log(f"  totals: pass1 -{tot['pass1']:,}  GT -{tot['gt']:,}  pass2 -{tot['pass2']:,} "
        f"(expected {tot['pass2_expected']:,})  ({time.time() - t0:.0f}s)")
    return report


# --------------------------------------------------------------------------- #
#  Gallery
# --------------------------------------------------------------------------- #
def _embed(path: Path, width: int, quality: int) -> str:
    """Image as an inline JPEG data URI, so the page is a single self-contained file."""
    import base64
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_index() -> Path:
    """review/index.html (removal candidates) and review/overviews.html (segment
    strips), each one self-contained file with the figures embedded."""
    import html
    css = """<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}
img{max-width:100%;border:1px solid #ddd;margin:4px 0}figure{margin:12px 0 28px}
nav{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ddd}
nav a{margin-right:10px}h2{margin-top:32px}li{margin:3px 0}</style>"""
    rows, ov_rows, segs = [], [], []
    total = {"pass1": [0, 0], "pass2": [0, 0]}
    for slug in segments():
        p = HERE / "review" / slug / "decisions.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        segs.append(slug)
        for n in total:
            total[n][0] += d["summary"][n]["remove"]
            total[n][1] += d["summary"][n]["points_to_remove"]
        cands = [r for r in d["clusters"] if r["remove"] and r.get("is_vehicle", True)]
        rows.append(f'<section id="{slug}"><h2>{slug} &mdash; {len(cands)} candidate(s)</h2>')
        for r in sorted(cands, key=lambda r: (r["pass"], -r["n_points"])):
            nf = sum(1 for q in d["clusters"] if q["pass"] == r["pass"] and q.get("fragment_of") == r["id"])
            frag = f", +{nf} attached fragment(s)" if nf else ""
            rows.append(
                f'<figure><figcaption><b>{r["pass"]} #{r["id"]}</b> &middot; {r["n_points"]:,} pts &middot; '
                f'{r["box_area_m2"]:.1f} m² &middot; initial score {r.get("initial_best_overlap", 0):.2f}, '
                f'final {r["best_overlap"]:.2f}{html.escape(frag)} &middot; '
                f'tiles {", ".join(f"{t:03d}" for t in r["tiles_files"])}</figcaption>'
                + (f'<img src="{_embed(HERE / "review" / slug / r["figure"], 1500, 80)}">'
                   if r.get("figure") else "")
                + "</figure>")
        rows.append("</section>")
        ov_rows.append(f'<section id="{slug}"><h2>{slug}</h2>' + "".join(
            f'<img src="{_embed(o, 1400, 65)}">'
            for o in sorted((HERE / "review" / slug).glob("overview_*.png"))) + "</section>")
    nav = " ".join(f'<a href="#{s}">{s}</a>' for s in segs)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Vehicle removal review</title>
{css}</head><body>
<h1>Vehicle removal review</h1>
<p>{total["pass1"][0]} pass-1 and {total["pass2"][0]} pass-2 vehicles flagged for removal
({total["pass1"][1]:,} + {total["pass2"][1]:,} points, attached fragments included).
Nothing has been removed yet. To override a decision, edit <code>"remove"</code> in
<code>review/&lt;seg&gt;/decisions.json</code>. Segment overview strips are in
<code>overviews.html</code>.</p>
<p>Each figure: <b>left</b> = the pass that contains the candidate vehicle,
<b>right</b> = the same footprint in the other pass; <b>top</b> = plan view,
<b>bottom</b> = side view of a slab through the box.</p>
<ul>
<li><span style="color:#d62728">&#9632;</span> <b>red</b>: this candidate (solid box in its own pass,
dashed box marking the same spot in the other pass)</li>
<li><span style="color:#ff9f40">&#9632;</span> <b>orange</b>: a <i>different</i> vehicle that is also
flagged for removal and happens to be in the same window (it has its own figure)</li>
<li><span style="color:#2ca02c">&#9632;</span> <b>green</b>: vehicle kept (seen in both passes)</li>
<li><span style="color:#8fb3d9">&#9632;</span> <b>light blue</b>: cluster labelled vehicle but too small
to be one (&lt; 1.5 m long, &lt; 1 m², or &lt; 0.5 m tall); left untouched</li>
<li>grey: everything else, darker = higher</li>
</ul>
<nav>{nav}</nav>{"".join(rows)}</body></html>"""
    out = HERE / "review" / "index.html"
    out.write_text(page)
    (HERE / "review" / "overviews.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Vehicle overviews</title>
{css}</head><body><h1>Segment overviews</h1>
<p>Pass 1 above pass 2, 125 m per strip. Green box = kept, red box = removed, number =
cluster id (matches index.html and decisions.json). Light blue = small non-vehicle
cluster, untouched.</p><nav>{nav}</nav>{"".join(ov_rows)}</body></html>""")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--segment", nargs="+", help="e.g. Z1S10 Z2S3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--thr", type=float, default=0.7,
                    help="min persistence score (default 0.7; see docstring)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--apply", action="store_true", help="remove using review/<seg>/decisions.json")
    ap.add_argument("--dry-run", action="store_true", help="with --apply: report only, write no LAS")
    a = ap.parse_args()
    segs = segments()
    if a.list:
        for s, zn in segs.items():
            print(f"{zn}/{s}  {len(tile_files(s))} tiles")
        return
    todo = list(segs) if a.all else (a.segment or [])
    if not todo:
        ap.error("give --segment, --all or --list")
    for s in todo:
        if s not in segs:
            raise SystemExit(f"unknown segment {s}; see --list")
    reports = []
    for s in todo:
        if a.apply:
            reports.append(apply(s, a.dry_run))
        else:
            review(s, a.thr, a.no_figs)
    if a.apply and not a.dry_run:
        summary = dict(
            description="Vehicles (class 8) present in only one pass were removed from that pass and "
                        "from TwinPass. Pass 1 and TwinPass are in the GT frame; Pass 2 keeps raw "
                        "pass-2 coordinates, as in the source tiles.",
            segments={r["segment"]: dict(removed_vehicles=r["removed_vehicles"], totals=r["totals"])
                      for r in reports})
        prev = HERE / "apply_reports" / "summary.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            old["segments"].update(summary["segments"])
            summary["segments"] = old["segments"]
        prev.write_text(json.dumps(summary, indent=1))
        log(f"report: {prev}")
    if not a.apply:
        log(f"gallery: {write_index()}")


if __name__ == "__main__":
    main()
