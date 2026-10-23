#!/usr/bin/env python3
"""
transfer_labels.py -- Transfer manual semantic annotations from the aligned
two-pass GT onto the original single-pass scans (TwinPass3D, step 3 of the
processing chain, after alignment.py and manual annotation).

Why exact matching works
------------------------
  * alignment.py wrote each GT as [pass A rows ; warped pass B rows] in the raw
    files' row order with nothing dropped, so GT row i <-> raw A row i and
    GT row nA+j <-> raw B row j.  This is checked per chunk (pass_id, timestamp,
    laser_id, and bit-exact XYZ for pass A).
  * Annotation (CloudCompare) only deleted points (trimming, outlier removal);
    survivors sit on the same 1 mm LAS grid as their GT source.

So each GT point's integer-mm coordinate is a key into the annotated clouds;
the label is carried to the matching raw single-pass row.  Unmatched points
(trimmed / removed as outliers) are dropped.  An optional +-tol mm neighbour
lookup catches float32 round-trip jitter from the annotation tool.

Inputs (paths are set by the constants below; by default everything lives
next to this script)
  Data/Zone <z>/<s>/{1,2}/agg_transformed_refined.las   raw single-pass scans
  out/Zone<z>_seg<s>/Zone<z>_seg<s>_GT.las               fused cloud from alignment.py
  Z<z>S<s>-annotated/<Class name>.las                    annotated cloud exported
                                                          from CloudCompare, one
                                                          file per class (the file
                                                          name gives the class)

Output: out/<slug>/<slug>_pass{1,2}_labeled.las  -- raw single-pass points
(original coordinates and all fields) that received a label, with
`classification` set from CLASS_MAP; plus <slug>_transfer_report.json.

    python transfer_labels.py --list
    python transfer_labels.py --segment Z1S10 --check     # stats only, no writes
    python transfer_labels.py --segment Z1S10
    python transfer_labels.py --all
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import laspy
from scipy.ndimage import binary_dilation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it

# ---- folder layout: edit these if your data live elsewhere --------------------
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "Data"      # same root as alignment.py --data
GT_ROOT = HERE / "out"         # same root as alignment.py --out (fused GT + _align.npz)
ANN_ROOT = HERE                # folders named Z<z>S<s>-annotated with per-class LAS
OUT_ROOT = HERE / "out"        # labelled single-pass scans are written here

# Intermediate annotation ids, used only in the labelled single-pass scans
# written by this script (0 is reserved for "no label", so ids start at 1).
# The file names of the annotated per-class clouds must map to these keys after
# class_name() normalisation (e.g. "High_Vegetation.las" -> "high_vegetation").
# split_tiles.py then folds shared_pathway -> sidewalk and signal -> pole, drops
# not_classified and writes the released codes 0-10 (Road 0 ... High Vegetation 10).
CLASS_MAP = {
    "not_classified": 1,
    "road": 2,
    "other_road": 3,
    "sidewalk": 4,
    "walkway": 5,
    "shared_pathway": 6,
    "building": 7,
    "fence": 8,
    "pole": 9,
    "sign": 10,
    "signal": 11,
    "vehicle": 12,
    "low_vegetation": 13,
    "high_vegetation": 14,
}

BITS = 21                       # per axis -> 2.097 km span at 1 mm, 63-bit key
SPAN = 1 << BITS
MARGIN_MM = 1000                # key-origin margin below the annotated minimum
CHUNK = 4_000_000
FOOT_CELL = 1.0                 # m, footprint grid (reporting only)


def log(msg: str) -> None:
    print(msg, flush=True)


def class_name(path: Path) -> str:
    """'High_Vegetation - Cloud.las' -> 'high_vegetation', 'other road' -> 'other_road'."""
    s = path.stem.split(" - ")[0].strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


def discover() -> list[dict]:
    segs = []
    for d in sorted(ANN_ROOT.glob("Z*S*-annotated")):
        m = re.fullmatch(r"Z(\d+)S(\d+)-annotated", d.name)
        if not m:
            continue
        z, s = m.groups()
        segs.append(dict(
            slug=f"Z{z}S{s}", ann_dir=d,
            pass1=DATA_ROOT / f"Zone {z}" / s / "1" / "agg_transformed_refined.las",
            pass2=DATA_ROOT / f"Zone {z}" / s / "2" / "agg_transformed_refined.las",
            gt=GT_ROOT / f"Zone{z}_seg{s}" / f"Zone{z}_seg{s}_GT.las",
            gt_report=GT_ROOT / f"Zone{z}_seg{s}" / f"Zone{z}_seg{s}_report.json",
        ))
    return segs


# --------------------------------------------------------------------------- #
#  Integer-mm keys
# --------------------------------------------------------------------------- #
def offset_mm(hdr) -> np.ndarray:
    if not np.allclose(hdr.scales, 0.001):
        raise ValueError(f"expected 1 mm scales, got {hdr.scales}")
    o = np.asarray(hdr.offsets, dtype=np.float64) / 0.001
    if not np.allclose(o, np.round(o), atol=1e-6):
        raise ValueError(f"offsets {hdr.offsets} are not on the 1 mm grid")
    return np.round(o).astype(np.int64)


def abs_mm(rec, off: np.ndarray) -> np.ndarray:
    """(n,3) int64 absolute millimetre coordinates from raw LAS integers."""
    return np.column_stack([np.asarray(rec.X, np.int64) + off[0],
                            np.asarray(rec.Y, np.int64) + off[1],
                            np.asarray(rec.Z, np.int64) + off[2]])


def pack(mm: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack to int64 keys; returns (keys, in_range). Out-of-range keys are -1."""
    d = mm - origin
    ok = np.all((d >= 0) & (d < SPAN), axis=1)
    k = np.full(len(mm), -1, np.int64)
    dd = d[ok]
    k[ok] = (dd[:, 0] << (2 * BITS)) | (dd[:, 1] << BITS) | dd[:, 2]
    return k, ok


# --------------------------------------------------------------------------- #
#  Label table
# --------------------------------------------------------------------------- #
class LabelTable:
    def __init__(self, ann_dir: Path):
        files = sorted(ann_dir.glob("*.las"))
        names = [class_name(p) for p in files]
        unknown = sorted(set(n for n in names if n not in CLASS_MAP))
        if unknown:
            raise SystemExit(f"{ann_dir.name}: unknown class files {unknown}")
        if len(set(names)) != len(names):
            raise SystemExit(f"{ann_dir.name}: duplicate class names {names}")

        mms, labs = [], []
        self.per_class = {}
        for p, n in zip(files, names):
            las = laspy.read(str(p))
            mm = abs_mm(las.points, offset_mm(las.header))
            mms.append(mm)
            labs.append(np.full(len(mm), CLASS_MAP[n], np.uint8))
            self.per_class[n] = len(mm)
            log(f"    {p.name:<28} {n:<16} {len(mm):>12,}")
        mm = np.concatenate(mms)
        lab = np.concatenate(labs)
        del mms, labs
        self.n_annotated = len(mm)
        self.origin = mm.min(axis=0) - MARGIN_MM
        if np.any(mm.max(axis=0) - self.origin >= SPAN):
            raise SystemExit("annotated extent exceeds key span")
        keys, _ = pack(mm, self.origin)

        # footprint grid (reporting: trimmed vs removed-outlier split)
        xy0 = mm[:, :2].min(axis=0)
        ij = ((mm[:, :2] - xy0) // int(FOOT_CELL * 1000)).astype(np.int64)
        grid = np.zeros(ij.max(axis=0) + 1, bool)
        grid[ij[:, 0], ij[:, 1]] = True
        self.foot = binary_dilation(grid, iterations=1)
        self.foot_xy0 = xy0
        del mm, ij

        # sort; resolve same-key/different-label by majority (tie -> dropped)
        order = np.lexsort((lab, keys))
        keys, lab = keys[order], lab[order]
        del order
        new_pair = np.ones(len(keys), bool)
        new_pair[1:] = (keys[1:] != keys[:-1]) | (lab[1:] != lab[:-1])
        ps = np.flatnonzero(new_pair)
        pk, pl = keys[ps], lab[ps]
        pc = np.diff(np.append(ps, len(keys)))
        self.n_dup_points = int(len(keys) - len(np.unique(keys)))  # cheap enough once
        del keys, lab, new_pair, ps
        new_key = np.ones(len(pk), bool)
        new_key[1:] = pk[1:] != pk[:-1]
        ks = np.flatnonzero(new_key)
        n_lab = np.diff(np.append(ks, len(pk)))
        multi = np.flatnonzero(n_lab > 1)
        self.n_conflict_keys = int(len(multi))
        self.n_tie_keys = 0
        out_lab = pl[ks].copy()
        for g in multi:                               # few; python loop is fine
            a, b = ks[g], ks[g] + n_lab[g]
            c = pc[a:b]
            best = np.flatnonzero(c == c.max())
            if len(best) > 1:
                out_lab[g] = 0                        # 0 = ambiguous -> dropped
                self.n_tie_keys += 1
            else:
                out_lab[g] = pl[a + best[0]]
        self.keys = pk[ks]
        self.labels = out_lab
        self.hit = np.zeros(len(self.keys), bool)    # annotated coverage
        log(f"    annotated {self.n_annotated:,} pts -> {len(self.keys):,} unique keys "
            f"({self.n_conflict_keys} conflicting, {self.n_tie_keys} ties dropped)")

    def lookup(self, k: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (label or 0, table index or -1)."""
        idx = np.searchsorted(self.keys, k)
        idx[idx >= len(self.keys)] = 0
        found = valid & (self.keys[idx] == k)
        lab = np.where(found, self.labels[idx], 0).astype(np.uint8)
        return lab, np.where(found, idx, -1)

    def in_footprint(self, mm: np.ndarray) -> np.ndarray:
        ij = (mm[:, :2] - self.foot_xy0) // int(FOOT_CELL * 1000)
        ok = np.all((ij >= 0) & (ij < self.foot.shape), axis=1)
        out = np.zeros(len(mm), bool)
        out[ok] = self.foot[ij[ok, 0], ij[ok, 1]]
        return out


NEIGH = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)],
                 np.int64)


def neighbour_lookup(table: LabelTable, mm: np.ndarray, tol: int):
    """For unmatched points: nearest-key label within +-tol mm (Chebyshev),
    preferring fewer-axis offsets. Returns (label or 0, table index or -1)."""
    lab = np.zeros(len(mm), np.uint8)
    hit = np.full(len(mm), -1, np.int64)
    if tol <= 0 or len(mm) == 0:
        return lab, hit
    offs = np.array([o for o in itertools.product(range(-tol, tol + 1), repeat=3)
                     if any(o)], np.int64)
    offs = offs[np.argsort(np.abs(offs).sum(axis=1), kind="stable")]
    todo = np.arange(len(mm))
    for o in offs:
        if len(todo) == 0:
            break
        k, ok = pack(mm[todo] + o, table.origin)
        l, ix = table.lookup(k, ok)
        f = l > 0
        lab[todo[f]], hit[todo[f]] = l[f], ix[f]
        todo = todo[~f]
    return lab, hit


# --------------------------------------------------------------------------- #
#  Per-segment transfer
# --------------------------------------------------------------------------- #
def check_rows(p: int, raw, g, raw_off, gt_off) -> None:
    if not np.all(np.asarray(g["pass_id"]) == p):
        raise SystemExit(f"GT pass_id != {p} in lockstep chunk: row order broken")
    for d in ("timestamp", "laser_id", "distance_m"):
        if not np.array_equal(np.asarray(g[d]), np.asarray(raw[d])):
            raise SystemExit(f"GT/raw pass {p} mismatch on '{d}': row order broken")
    if p == 1 and not np.array_equal(abs_mm(g, gt_off), abs_mm(raw, raw_off)):
        raise SystemExit("pass 1 XYZ not bit-exact between GT and raw")


def unpack(keys: np.ndarray, origin: np.ndarray) -> np.ndarray:
    m = (1 << BITS) - 1
    return np.column_stack([(keys >> (2 * BITS)) & m, (keys >> BITS) & m,
                            keys & m]) + origin


def recover_pass2(gt_path: Path, table: LabelTable, n_a: int, radius_mm: float):
    """Pass-2 rows whose GT coordinate misses every annotated key by <= radius.

    Some annotations were made on a GT copy whose pass-2 warp requantised a
    millimetre differently (seen at the segment ends of Z1S8 / Z2S6), so those
    annotated points sit ~1 mm off the current GT.  Pass 1 is bit-exact and is
    never touched.  Candidates are only annotated keys with no exact match
    anywhere in the GT; each is given to at most one pass-2 point (its
    nearest), so recovered labels cannot exceed the annotation's own count.
    Returns (pass-2 row indices sorted, labels, table indices, stats).
    """
    from scipy.spatial import cKDTree
    hit = np.zeros(len(table.keys), bool)
    rows, mms = [], []
    with laspy.open(str(gt_path)) as f:
        off = offset_mm(f.header)
        r0 = 0
        for ch in f.chunk_iterator(CHUNK):
            mm = abs_mm(ch, off)
            k, ok = pack(mm, table.origin)
            _, ix = table.lookup(k, ok)
            hit[ix[ix >= 0]] = True
            row = np.arange(r0, r0 + len(ch))
            miss = (ix < 0) & (row >= n_a)
            if miss.any():
                miss[miss] = table.in_footprint(mm[miss])
                rows.append(row[miss] - n_a)
                mms.append(mm[miss])
            r0 += len(ch)
    cand = np.flatnonzero(~hit & (table.labels > 0))
    stats = dict(radius_mm=radius_mm, unmatched_annotated_keys=int(len(cand)),
                 recovered=0)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.int64), stats)
    if len(cand) == 0 or not rows:
        return empty
    rows = np.concatenate(rows)
    mms = np.concatenate(mms)
    tree = cKDTree(unpack(table.keys[cand], table.origin).astype(np.float64))
    d, j = tree.query(mms.astype(np.float64), distance_upper_bound=radius_mm + 1e-6,
                      workers=-1)
    ok = np.isfinite(d)
    rows, d, j = rows[ok], d[ok], j[ok]
    order = np.lexsort((rows, d))                       # nearest first
    _, first = np.unique(j[order], return_index=True)   # one GT point per key
    sel = order[first]
    rows, d, tix = rows[sel], d[sel], cand[j[sel]]
    o = np.argsort(rows)
    rows, d, tix = rows[o], d[o], tix[o]
    stats.update(recovered=int(len(rows)),
                 distance_mm_pct50_90_max=[float(np.percentile(d, 50)),
                                           float(np.percentile(d, 90)),
                                           float(d.max())] if len(d) else None)
    return rows, table.labels[tix], tix, stats


def process(seg: dict, check: bool, tol: int, recover_mm: float = 5.0) -> dict:
    t0 = time.time()
    slug = seg["slug"]
    log(f"\n== {slug} ==")
    for k in ("pass1", "pass2", "gt"):
        if not seg[k].exists():
            raise SystemExit(f"{slug}: missing {k}: {seg[k]}")
    merge = json.loads(seg["gt_report"].read_text())["merge"]
    # The row correspondence below needs the GT to be the full union A + B.
    # (Reports from older alignment runs also carry dropped_a / dropped_b.)
    if (merge.get("dropped_a", 0) or merge.get("dropped_b", 0)
            or merge.get("points_total", merge["points_a_in"] + merge["points_b_in"])
            != merge["points_a_in"] + merge["points_b_in"]):
        raise SystemExit(f"{slug}: GT dropped points; row alignment invalid")

    log("  building label table")
    table = LabelTable(seg["ann_dir"])
    out_dir = OUT_ROOT / slug
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)

    inv = {v: k for k, v in CLASS_MAP.items()}
    report = dict(slug=slug, mode="check" if check else "write", tol_mm=tol,
                  recover_mm=recover_mm,
                  class_map=CLASS_MAP, annotated_points=table.n_annotated,
                  annotated_per_class=table.per_class,
                  annotated_unique_keys=int(len(table.keys)),
                  annotated_duplicate_points=table.n_dup_points,
                  conflicting_keys=table.n_conflict_keys,
                  tie_keys_dropped=table.n_tie_keys, passes={})

    rec_rows = rec_lab = rec_tix = None
    if recover_mm > 0:
        log(f"  pass-2 recovery scan (<= {recover_mm:g} mm)")
        rec_rows, rec_lab, rec_tix, rst = recover_pass2(
            seg["gt"], table, merge["points_a_in"], recover_mm)
        report["pass2_recovery"] = rst
        log(f"    unmatched annotated keys {rst['unmatched_annotated_keys']:,} -> "
            f"recovered {rst['recovered']:,}")

    with laspy.open(str(seg["gt"])) as gt_f:
        if gt_f.header.point_count != merge["points_a_in"] + merge["points_b_in"]:
            raise SystemExit(f"{slug}: GT count != pass A + pass B")
        gt_off = offset_mm(gt_f.header)
        for p in (1, 2):
            src = seg[f"pass{p}"]
            n_exp = merge[f"points_{'a' if p == 1 else 'b'}_in"]
            st = dict(points_in=0, exact=0, neighbour=0, recovered=0,
                      dropped_outside_footprint=0,
                      dropped_inside_footprint=0, ambiguous_dropped=0)
            per = np.zeros(32, np.int64)
            out_path = out_dir / f"{slug}_pass{p}_labeled.las"
            with laspy.open(str(src)) as raw_f:
                if raw_f.header.point_count != n_exp:
                    raise SystemExit(f"{slug} pass {p}: count {raw_f.header.point_count} "
                                     f"!= GT report {n_exp}")
                raw_off = offset_mm(raw_f.header)
                writer = None
                if not check:
                    writer = laspy.open(str(out_path), mode="w",
                                        header=copy.deepcopy(raw_f.header))
                n_chunks = -(-raw_f.header.point_count // CHUNK)
                for raw in tqdm(raw_f.chunk_iterator(CHUNK), total=n_chunks,
                                desc=f"  pass {p}", unit="chunk", leave=False):
                    g = gt_f.read_points(len(raw))
                    check_rows(p, raw, g, raw_off, gt_off)
                    mm = abs_mm(g, gt_off)
                    k, ok = pack(mm, table.origin)
                    lab, ix = table.lookup(k, ok)
                    exact = ix >= 0
                    table.hit[ix[exact]] = True
                    st["ambiguous_dropped"] += int(np.sum(exact & (lab == 0)))
                    st["exact"] += int(np.sum(lab > 0))
                    miss = np.flatnonzero(ix < 0)
                    if tol > 0 and len(miss):
                        foot = table.in_footprint(mm[miss])
                        cand = miss[foot]
                        nl, nix = neighbour_lookup(table, mm[cand], tol)
                        f = nl > 0
                        lab[cand[f]] = nl[f]
                        table.hit[nix[f]] = True
                        st["neighbour"] += int(f.sum())
                    if p == 2 and rec_rows is not None and len(rec_rows):
                        r0 = st["points_in"]
                        a_, b_ = np.searchsorted(rec_rows, [r0, r0 + len(raw)])
                        if b_ > a_:
                            loc = rec_rows[a_:b_] - r0
                            free = (lab[loc] == 0) & (ix[loc] < 0)
                            lab[loc[free]] = rec_lab[a_:b_][free]
                            table.hit[rec_tix[a_:b_][free]] = True
                            st["recovered"] += int(free.sum())
                    left = (lab == 0) & (ix < 0)
                    foot_left = table.in_footprint(mm[left])
                    st["dropped_inside_footprint"] += int(foot_left.sum())
                    st["dropped_outside_footprint"] += int((~foot_left).sum())
                    st["points_in"] += len(raw)
                    per += np.bincount(lab, minlength=32)
                    if writer is not None:
                        keep = lab > 0
                        if keep.any():
                            rec = raw[keep]
                            rec.classification = lab[keep]
                            writer.write_points(rec)
                if writer is not None:
                    writer.close()
            st["points_out"] = int(per[1:].sum())
            st["kept_fraction"] = st["points_out"] / max(1, st["points_in"])
            st["per_class"] = {inv[c]: int(per[c]) for c in range(1, 32) if per[c]}
            if not check:
                st["path"] = str(out_path)
            report["passes"][f"pass{p}"] = st
            log(f"  pass {p}: in {st['points_in']:,}  kept {st['points_out']:,} "
                f"({100 * st['kept_fraction']:.2f}%)  exact {st['exact']:,}  "
                f"+-{tol}mm {st['neighbour']:,}  recovered {st['recovered']:,}  dropped in/out footprint "
                f"{st['dropped_inside_footprint']:,}/{st['dropped_outside_footprint']:,}")

    # every annotated key should come from some GT point
    covered = int(table.hit.sum())
    unmatched = np.flatnonzero(~table.hit)
    report["annotated_keys_matched"] = covered
    report["annotated_keys_unmatched"] = int(len(unmatched))
    report["annotated_key_coverage"] = covered / max(1, len(table.keys))
    report["seconds"] = round(time.time() - t0, 1)
    log(f"  annotated key coverage: {covered:,}/{len(table.keys):,} "
        f"({100 * report['annotated_key_coverage']:.4f}%)  [{report['seconds']} s]")

    name = f"{slug}_transfer_{'check' if check else 'report'}.json"
    (OUT_ROOT / slug).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / slug / name).write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--segment", nargs="+", help="e.g. Z1S10 Z2S3")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="match statistics only; write no LAS")
    ap.add_argument("--tol-mm", type=int, default=0,
                    help="neighbour fallback radius in mm (0 = exact only)")
    ap.add_argument("--recover-mm", type=float, default=5.0,
                    help="pass-2 recovery radius for annotated points with no exact "
                         "GT match (0 = off); no-op where coverage is already 100%%")
    a = ap.parse_args(argv)

    segs = discover()
    if a.list:
        for s in segs:
            ok = all(s[k].exists() for k in ("pass1", "pass2", "gt"))
            print(f"{s['slug']:<7} {'ok' if ok else 'MISSING INPUTS'}")
        return 0
    if a.segment:
        want = {x.upper() for x in a.segment}
        segs = [s for s in segs if s["slug"].upper() in want]
        if len(segs) != len(want):
            raise SystemExit("unknown segment(s); see --list")
    for s in segs:
        process(s, a.check, a.tol_mm, a.recover_mm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
