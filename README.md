# TwinPass3D

[![DOI](https://zenodo.org/badge/1430433678.svg)](https://doi.org/10.5281/zenodo.23584201)

Processing code for **TwinPass3D**, a two-pass micromobility LiDAR dataset for semantic segmentation and far-side scene completion. Every street segment was scanned twice, once from each sidewalk, with a mobility scooter carrying a 32-channel LiDAR. The two traversals are registered and fused at full resolution, annotated into 11 classes, and released as paired single-pass and complete point clouds cut into 25 m tiles.

This repository contains the scripts used to go from the two georeferenced single-pass scans of each segment to the released tiles.

## Dataset at a glance

- 21 street segments in three residential neighbourhoods of Edmonton, Canada (Zone 1: 10 segments, Zone 2: 6, Zone 3: 5)
- three versions per segment: `TwinPass` (complete point cloud) and the labelled single-pass scans `Pass 1` and `Pass 2`
- 403 tiles of 25 m per version, with identical tile boundaries in all three versions
- every point of a single-pass tile appears with identical coordinates in the complete tile of the same name

Released layout:

```
TwinPass3D/
└── Zone <z>/
    ├── TwinPass/<segment>/<segment>_<kkk>.ply   complete point cloud
    ├── Pass 1/<segment>/<segment>_<kkk>.ply     single-pass scan, pass 1
    └── Pass 2/<segment>/<segment>_<kkk>.ply     single-pass scan, pass 2
```

Each PLY file (binary little-endian) has five vertex properties:

| Property | Type | Description |
|---|---|---|
| `x`, `y`, `z` | double | UTM zone 12N, metres, on a 1 mm grid |
| `intensity` | ushort | LiDAR return intensity |
| `classification` | uchar | class code (table below) |

| Code | Class | Code | Class |
|---|---|---|---|
| 0 | Road | 6 | Pole |
| 1 | Other Road | 7 | Sign |
| 2 | Sidewalk | 8 | Vehicle |
| 3 | Walkway | 9 | Low Vegetation |
| 4 | Building | 10 | High Vegetation |
| 5 | Fence | | |

Reading a tile in Python:

```python
import numpy as np
from plyfile import PlyData

v = PlyData.read("TwinPass3D/Zone 2/TwinPass/Z2S1/Z2S1_009.ply")["vertex"].data
xyz = np.column_stack([v["x"], v["y"], v["z"]])     # keep float64 (UTM)
labels = v["classification"]
```

Because a complete tile is the union of the two single-pass tiles with the same name, the pass a point comes from can be recovered by matching coordinates at millimetre resolution, e.g. `np.round(xyz * 1000).astype(np.int64)`.

**Data access:** the dataset is available for peer review via Google Drive and will be deposited in Zenodo with a DOI on acceptance; this README will be updated with the DOI.

## Repository contents

| Script | Step | What it does |
|---|---|---|
| `alignment.py` | 1 | Registers pass 2 to pass 1 (along-track lock, global and piecewise point-to-plane ICP, anchored SLERP blending) and writes the fused point cloud at full resolution |
| `transfer_labels.py` | 3 | Transfers the manual annotations from the fused cloud to the original single-pass scans by exact millimetre-coordinate matching |
| `split_tiles.py` | 4 | Cuts the complete cloud and both single-pass scans into 25 m along-street tiles with shared boundaries, and writes the class codes 0–10 |
| `vehicle_persistence.py` | 5 | Keeps vehicles observed in both passes and removes those present in only one pass, from that pass and from the complete cloud |

Step 2 is manual annotation in [CloudCompare](https://www.cloudcompare.org).

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Processing chain

All scripts look for their inputs next to themselves by default. The folder constants at the top of `transfer_labels.py` and `vehicle_persistence.py` can be edited if your data live elsewhere; `alignment.py` takes `--data` and `--out`.

### 0. Inputs

Each traversal is a georeferenced SLAM point cloud (UTM, 1 mm scale factor) carrying the extra dimensions `distance_m`, `laser_id` and `timestamp`, plus a position-only trajectory:

```
Data/
└── Zone <z>/<s>/
    ├── 1/agg_transformed_refined.las    pass 1 (reference)
    ├── 1/trj_G_Real.las                 pass 1 trajectory
    ├── 2/agg_transformed_refined.las    pass 2 (registered to pass 1)
    └── 2/trj_G_Real.las                 pass 2 trajectory
```

### 1. Registration and fusion

```bash
python alignment.py --data Data --out out --list
python alignment.py --data Data --out out --all
```

For each segment this writes `out/Zone<z>_seg<s>/` with the fused cloud `_GT.las` (pass 1 rows first, then pass 2, nothing dropped, plus a `pass_id` dimension), the registration state `_align.npz`, and a report `_report.json` with the along-track offset, ICP diagnostics and residuals before and after registration. All parameters are in the `Config` class at the top of the script.

### 2. Manual annotation

Annotate the fused cloud in CloudCompare and export one LAS file per class into `Z<z>S<s>-annotated/` (e.g. `Z1S10-annotated/Road.las`, `Z1S10-annotated/High_Vegetation.las`). Only delete points during cleaning; do not move them, so every annotated point keeps its 1 mm coordinates. The file names must match the class names in `transfer_labels.CLASS_MAP`.

### 3. Label transfer

```bash
python transfer_labels.py --list
python transfer_labels.py --segment Z1S10 --check    # statistics only
python transfer_labels.py --all
```

Writes `out/Z<z>S<s>/Z<z>S<s>_pass{1,2}_labeled.las`: the original single-pass points that received a label, with all their attributes.

### 4. Tiling

```bash
python split_tiles.py --source gt pass1 pass2 --all --pass2-aligned
```

Writes `Final_Dataset/{gt,pass1,pass2}/Zone<z>/Z<z>S<s>/Z<z>S<s>_<kkk>.las` and a `tiles.json` per segment with the tile boundaries and class counts. Tiles are measured along the pass 1 trajectory (straight line, or the smoothed trajectory for curved segments given with `--curved`). `--pass2-aligned` writes pass 2 in the frame of the complete cloud; re-applying the warp can differ from the fused cloud by about 1 mm for some points, so for bit-exact correspondence snap each pass 2 point to its counterpart in the complete tile.

### 5. Non-persistent vehicle removal

```bash
python vehicle_persistence.py --all                     # review: figures + decisions
#   inspect review/<segment>/ and edit decisions.json to override
python vehicle_persistence.py --all --apply --dry-run   # counts only
python vehicle_persistence.py --all --apply             # writes TwinPass3D/
```

Writes the tiles in the release layout (`TwinPass3D/Zone <z>/{TwinPass,Pass 1,Pass 2}/`). The released files were then converted to PLY with the five properties listed above.

## Citation

If you use the code or the dataset, please cite:

> E. K. Mahmoud, A. M. Sakr, F. Birhane, and K. El-Basyouny, "A Two-Pass Micromobility LiDAR Dataset for Semantic Segmentation and Far-Side Scene Completion," manuscript under review.

To cite the code itself, use the Zenodo record: https://doi.org/10.5281/zenodo.23584201 (all versions; v1.0.0: https://doi.org/10.5281/zenodo.23584202).

The citation will be updated on publication.

## License

The code is released under the MIT License (see `LICENSE`). The dataset is distributed separately under its own licence.
