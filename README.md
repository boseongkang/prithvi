# Fault Detection with LiDAR DEM + Deep Learning

Automated detection of active fault zones in California using high-resolution LiDAR DEMs
and transformer-based semantic segmentation. Built as part of the
[SCEC (Southern California Earthquake Center)](https://www.scec.org/) seismic hazard
research project at SJSU.

**Live demos**
- Presentation dashboard: https://prithvi-app-88345939641.us-west2.run.app
- Research overview: https://prithvi-paper-88345939641.us-west2.run.app

This README covers the current work. Earlier phases are preserved in
[README_v2.md](README_v2.md) (DGX, 6 regions) and [README_v1.md](README_v1.md) (Colab).

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Two Phases of Work](#2-two-phases-of-work)
3. [Current Results (DGX, automated selection)](#3-current-results)
4. [Earlier Results (Colab, per-region)](#4-earlier-results)
5. [What Changed Between Phases](#5-what-changed-between-phases)
6. [Pipeline](#6-pipeline)
7. [Setup](#7-setup)
8. [Repository Structure](#8-repository-structure)
9. [System Design](#9-system-design)
10. [Findings](#10-findings)
11. [Data Sources](#11-data-sources)

---

## 1. What This Is

This project trains models to detect earthquake fault traces from LiDAR-derived terrain.
Active faults leave physical scars on the landscape, visible as linear scarps and abrupt
slope changes in a hillshade. A segmentation model learns to find these signatures
directly from 0.5 m DEMs, instead of months of manual field survey.

Two research objectives from the SCEC proposal:
- **Objective 1B:** detect faults from LiDAR DEM derivatives (hillshade, slope) with
  semantic segmentation. This is the main line of work.
- **Objective 1A:** detect faults from HLS Sentinel-2 imagery with the Prithvi-EO-2.0
  foundation model. Explored as a satellite-based comparison.

## 2. Two Phases of Work

The project ran in two distinct phases with different infrastructure, regions, and
evaluation. They are reported separately because their numbers are **not directly
comparable** (different regions, buffer width, and train/test split).

| | Earlier (Colab) | Current (DGX) |
|----------------------|---------------------------------|----------------------------------|
| Compute | Google Colab A100 / L4 | DGX Blackwell GB10 (130 GB) |
| Region selection | manual | automated by fault density |
| Regions | 6 (Carrizo, Owens, Mojave, ...) | 4 (Big Pine, Panamint, ...) |
| Input channels | hillshade + slope | tested 2 vs 5 (no difference) |
| Label buffer | 1 m | 10 m |
| Reported metric | test IoU (per-region) | held-out test IoU |
| Mask workflow | QGIS (manual) | code (`make_mask.py`) |
| Best result | Owens 0.486 | Big Pine 0.294 |

## 3. Current Results

Per-region SegFormer-B2 (`nvidia/mit-b2`) at the best pixel threshold. Test IoU is the
held-out number and the one to trust. One model per region; regions are never merged.

| Region        | Place           | Fault type    | Ch | Val IoU | Test IoU | Recall | Precision |
|---------------|-----------------|---------------|----|---------|----------|--------|-----------|
| Big Pine      | Owens Valley    | right-lateral | 2  | 0.424   | 0.294    | 0.443  | 0.467     |
| Panamint      | Panamint Valley | normal        | 5  | 0.202   | 0.157    | 0.355  | 0.219     |
| Death Valley  | Quail Mountains | left + right  | 5  | 0.227   | 0.085    | 0.142  | 0.174     |
| Sierra Nevada | Cantil          | left-lateral  | 5  | 0.151   | 0.074    | 0.097  | 0.236     |

Region selection was automated: `scan_fault_lidar.py` slides an 11 km window over the
EarthScope Southern & Eastern California LiDAR footprint, scores every tile by USGS
Quaternary fault length inside it, and ranks the fault-dense tiles for download.

Study regions (UTM Zone 11N, EPSG:32611):

| Region        | Xmin    | Ymin    | Mean slope |
|---------------|---------|---------|------------|
| Big Pine      | 381962  | 4100953 | 3.6        |
| Sierra Nevada | 405344  | 3899117 | 2.4        |
| Panamint      | 472034  | 4001843 | 17.6       |
| Death Valley  | 504000  | 3935679 | moderate   |

Channel test: on Panamint, head to head, 5 channels (hillshade, slope, sin/cos aspect,
roughness) gave test IoU 0.157 and 2 channels gave 0.143. The extra aspect and roughness
channels carried no useful signal, so the multi-channel hypothesis did not hold here.

Architecture test: SegFormer-B5 on Big Pine reached test IoU 0.272, below B2's 0.294
(the larger model overfits the limited data).

## 4. Earlier Results

From the Colab phase, per-region SegFormer-B2 over 6 California regions, reported as test
IoU on a hillshade + slope input with a 1 m label buffer:

| Region        | Test IoU | Fault type    | System            |
|---------------|----------|---------------|-------------------|
| Owens Valley  | 0.486    | Normal        | ECSZ              |
| Mojave        | 0.393    | Oblique       | ECSZ              |
| Carrizo       | 0.369    | Strike-slip   | San Andreas       |
| Imperial      | 0.324    | Strike-slip   | San Andreas       |
| Pauma Valley  | 0.238    | Reverse       | Peninsular Ranges |
| Sierra Pelona | 0.121    | Reverse       | Transverse Ranges |

This phase produced the **ECSZ hypothesis**: normal and oblique faults in the Eastern
California Shear Zone, with strong vertical displacement, create clear light/shadow
boundaries in LiDAR hillshade and were the easiest to detect, while pure strike-slip and
sediment-covered reverse faults gave weaker signals. The full Colab progression (U-Net
baseline through SegFormer, patch-size ablation, Prithvi/HLS) is in
[README_v1.md](README_v1.md).

## 5. What Changed Between Phases

The current phase is a stricter, more automated redo of the earlier idea:
- **Automated region selection** replaced hand-picking sites, so adding a region is a
  coordinate box and a script run.
- **Held-out test split** by spatial blocks gives an honest generalization number, where
  the earlier phase leaned on more optimistic per-region scores.
- **A slope nodata bug was fixed.** A `-9999` nodata value had been leaking into slope
  normalization (slope mean near -39 instead of ~3); fixing it moved Owens Valley from
  test IoU 0.083 to 0.294. No architecture change moved the number that much.
- **The mask workflow moved from QGIS to code** (`make_mask.py`) with identical
  parameters (0.5 m, 10 m buffer), so it is reproducible from a single script.

Because of the region, buffer, and split differences, the Owens 0.486 (earlier) and
Big Pine 0.294 (current) are not the same measurement, even though both sit in Owens
Valley. The current numbers are the conservative, reproducible ones.

## 6. Pipeline

1. **Region selection** (`scan_fault_lidar.py`): slide an 11 km window over the LiDAR
   footprint, score by USGS fault length, rank fault-dense tiles.
   `check_fault_count.py` confirms a candidate bbox before download.
2. **DEM download**: bare-earth derivatives (hillshade, slope, aspect, roughness) from
   OpenTopography at 0.5 m, UTM Zone 11N. `01_download_dem.py` is a 1 m USGS 3DEP variant.
3. **Mask** (`make_mask.py`): clip USGS fault lines to the DEM, buffer 10 m, dissolve,
   rasterize at 0.5 m so the mask aligns pixel-for-pixel with the DEM.
4. **Patches** (`make_patches.py`): 256x256 at stride 128, train/val/test split by
   spatial blocks (no leakage). 5-channel output.
5. **Train** (`04_train.py`): fine-tune SegFormer-B2, weighted CE + Dice, threshold sweep.
6. **Visualize** (`visualize_contour.py`): overlay USGS label and prediction as contour
   lines on the hillshade.
7. **Collect** (`extract_results.py`): gather config + metrics across runs.

## 7. Setup

```bash
pip install -r dgx/requirements.txt
```

Key packages: `transformers` (SegFormer), `torch`, `rasterio` and `geopandas` (LiDAR /
GIS I/O), `albumentations` (augmentation). Training expects data under `data/<region>/`
as described in the pipeline.

Run the dashboard locally:
```bash
python app4.py      # presentation, port 8080
python app5.py      # research overview, port 8051
```

## 8. Repository Structure

```
prithvi/
  dgx/
    scripts/
      scan_fault_lidar.py    automated region selection by fault density
      check_fault_count.py   estimate fault count/coverage for a bbox
      01_download_dem.py     USGS 3DEP 1 m DEM download + hillshade/slope
      make_mask.py           build fault mask from USGS lines (10 m buffer)
      make_patches.py        cut 256x256 5-channel patches, spatial split
      04_train.py            SegFormer-B2 training, threshold sweep
      visualize_contour.py   contour overlay of label vs prediction
      extract_results.py     collect metrics across runs
    segformer_b2_per_region.py   earlier Colab per-region training
    segformer_train.py           earlier Carrizo baseline
    segformer_b5_train.py        earlier B5 experiment
  notebook/                  earlier U-Net and Prithvi notebooks
  results/                   saved results
  README.md                  this file
  README_v1.md               earlier project README (Colab phase)
```

## 9. System Design

Three tiers, one role each:

```
Local / preprocessing   ->   GPU training (DGX)   ->   Cloud serving
  scan, download,             SegFormer-B2,             Cloud Run (Dash),
  mask, patches               per-region models         results from GCS
```

- **Separation of concerns:** data prep, training, and serving are independent.
- **Stateless inference:** Cloud Run scales horizontally; the dashboard reads results
  from Google Cloud Storage at runtime (with an in-code fallback if GCS is unreachable),
  so new experiments show up without a redeploy.
- **Reproducibility:** preprocessing is scripted; training configs live in code.

## 10. Findings

- **Terrain decides, not the model.** Big Pine (0.294) and Sierra Nevada (0.074) ran the
  same model; a sharp mountain front versus a flat valley. Recall tracks it (0.44 vs 0.10).
- **Extra channels did nothing.** 5 channels vs 2 on Panamint: 0.157 vs 0.143.
- **Too steep backfires.** Panamint is steepest with the most fault coverage, yet
  precision is only 0.22: every hillside looks like a scarp and the model over-predicts.
- **Validation is optimistic.** Death Valley dropped 0.227 (val) to 0.085 (test).
- **Preprocessing beat architecture.** The slope nodata fix moved Owens from 0.083 to
  0.294, more than any model change.
- **Fault kinematics matter (earlier phase).** ECSZ normal/oblique faults with vertical
  displacement scored highest; strike-slip and reverse scored lower.

Current test-IoU values (0.07-0.29) are in line with published DEM fault-detection work:
a hard, class-imbalanced task (fault pixels are under 1.5% of each tile).

## 11. Data Sources

- **OpenTopography** EarthScope Southern & Eastern California LiDAR Project, 0.5 m.
  https://doi.org/10.5069/G9G44N6Q
- **USGS** Quaternary Fault and Fold Database.
  https://www.usgs.gov/programs/earthquake-hazards/faults
- **NASA HLS** Sentinel-2 imagery (Objective 1A).
- **IBM/NASA Prithvi-EO-2.0** geospatial foundation model (Objective 1A).

## License

MIT. See [LICENSE](LICENSE).
