# Fault Detection with LiDAR DEM + Deep Learning

Automated detection of active fault zones in California using high-resolution LiDAR DEMs and transformer-based semantic segmentation. Built as part of the [SCEC (Southern California Earthquake Center)](https://www.scec.org/) seismic hazard research project at SJSU.

🌐 **Live Dashboard:** [prithvi-app-88345939641.us-west2.run.app](https://prithvi-app-88345939641.us-west2.run.app)

<img src="https://github.com/user-attachments/assets/38d510b6-751a-41a1-941e-ead682fef571" alt="Owens Valley Predictions" width="800"/>

*SegFormer-B2 predictions on Owens Valley test set. Columns: Hillshade input → Ground Truth → Predicted probability → Overlay. Best test IoU: 0.486.*

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Latest Results](#2-latest-results)
3. [Setup Instructions](#3-setup-instructions)
4. [Full Pipeline](#4-full-pipeline)
5. [Repository Structure](#5-repository-structure)
6. [System Design](#6-system-design)
7. [Inference Service](#7-inference-service)
8. [Cloud Data Storage](#8-cloud-data-storage)
9. [Geological Insights](#9-geological-insights)

---

## 1. What This Is

This project trains AI models to automatically detect earthquake fault traces from LiDAR-derived terrain data. Instead of geologists spending months physically surveying terrain, the model learns to identify fault signatures (linear scarps, offset features) directly from 0.5m resolution Digital Elevation Models.

**Two research objectives:**

- **Objective 1B:** Detect fault zones from LiDAR-derived DEM (hillshade + slope) using semantic segmentation
- **Objective 1A:** Detect fault zones from HLS Sentinel-2 satellite imagery using Prithvi-EO-2.0 foundation model

The core idea is that active faults leave physical scars on the landscape — visible as linear shadow boundaries in hillshade images and as abrupt slope changes. A well-trained segmentation model can learn to find these signatures automatically.

**Study Regions (6 total):**
- Carrizo Plain (San Andreas Fault — strike-slip)
- Owens Valley (ECSZ — normal fault) ⭐ Best performance
- Mojave (ECSZ — oblique-slip)
- Imperial Valley (Southern San Andreas — strike-slip)
- Pauma Valley (Peninsular Ranges — reverse)
- Sierra Pelona (Transverse Ranges — reverse)

---

## 2. Latest Results

### Per-Region SegFormer-B2 Specialist Models

| Rank | Region | Test IoU | F1 | Fault Type | System |
|------|--------|----------|-----|------------|--------|
| 🥇 | **Owens Valley** | **0.486** | 0.654 | Normal | ECSZ |
| 🥈 | **Mojave** | **0.393** | 0.564 | Oblique | ECSZ |
| 🥉 | Carrizo (baseline) | 0.369 | — | Strike-slip | SAF |
| 4 | Imperial Valley | 0.324 | 0.489 | Strike-slip | SAF |
| 5 | Pauma Valley | 0.238 | 0.384 | Reverse | Peninsular |
| 6 | Sierra Pelona | 0.121 | 0.215 | Reverse | Transverse |

**Key finding:** Owens Valley achieved IoU 0.486, a **+31.7% improvement** over the Carrizo baseline.

### Failed Approaches (Important Negative Results)

| Approach | IoU | Why it Failed |
|----------|-----|---------------|
| Phase 0: All regions combined | 0.032 | Mixed terrain types prevent convergence |
| B5 Multi-region (82M params) | 0.069 | Model too large for available data |
| B5 Carrizo only | 0.012 | Too few fault patches (392) for large model |
| B2 Multi-region | 0.087 | Same terrain mixing problem persists |

**Conclusion:** Per-region specialist models significantly outperform multi-region approaches.

---

## 3. Setup Instructions

### Requirements

```
Python >= 3.10
CUDA-capable GPU (recommended: 16GB+ VRAM for SegFormer-B2 training)
Cloud storage for patches and checkpoints (~10GB per experiment)
QGIS >= 3.x (for fault mask generation — one-time preprocessing only)
```

### Install Dependencies

```bash
pip install -r dgx/requirements.txt
```

Key packages:
- `transformers` (for SegFormer)
- `torch`, `torchvision`
- `albumentations` (augmentation)
- `rasterio` (LiDAR I/O)
- `segmentation-models-pytorch` (U-Net baseline)

---

## 4. Full Pipeline

```
1. Raw LiDAR Data (OpenTopography)
   └─ Download DEM tiles for each study region
        ↓
2. QGIS Preprocessing (one-time per region)
   ├─ Compute hillshade + slope from DEM
   ├─ Load USGS Quaternary Fault Database
   ├─ Extract by extent → Reproject to EPSG:32611
   ├─ Buffer 10m → Rasterize at 0.5m resolution
   └─ Output: region_fault_mask.tif (binary, byte type)
        ↓
3. Patch Extraction (Python)
   ├─ Cut hillshade + slope + mask into 256×256 patches
   ├─ Stride = 128 (50% overlap)
   ├─ Hard Negative Mining: keep 10% of background patches
   ├─ 70/15/15 train/val/test split
   └─ Output: train.npz / val.npz / test.npz
        ↓
4. Per-Region Training (GPU)
   ├─ SegFormer-B2 from nvidia/mit-b2 (ADE20K pretrained)
   ├─ Weighted CE + Dice loss (fault_weight=5.0)
   ├─ WeightedRandomSampler (50x fault oversampling)
   ├─ Enhanced augmentation (Elastic + GaussNoise + ShiftScaleRotate)
   ├─ AdamW differential LR (encoder=1e-5, decoder=1e-4)
   ├─ Early stopping (patience=15)
   └─ Output: best.pth + test_results.json
        ↓
5. Threshold Tuning + Test Evaluation
   ├─ Search threshold 0.10~0.50 (step 0.02)
   ├─ Maximize IoU_fault on validation set
   └─ Apply best threshold to test set
```

---

## 5. Repository Structure

```
prithvi/
├── dgx/                                 # GPU training scripts
│   ├── segformer_train.py               # B2 single-region (Carrizo baseline)
│   ├── segformer_b5_train.py            # B5 enhanced (experimental)
│   ├── segformer_b2_per_region.py       # B2 per-region (final, best results)
│   ├── per_region_summary.json          # All region test metrics
│   └── requirements.txt
├── notebook/                            # Exploratory + baseline notebooks
│   ├── DEM_Unet.ipynb                   # U-Net ResNet34 baseline
│   ├── DEM_buffer1m.ipynb               # 1m buffer ablation
│   ├── USGS_1m_dataset.ipynb            # Ground truth dataset construction
│   ├── obj1A_256x256.ipynb              # Prithvi HLS approach
│   ├── segformer_fault_detection.ipynb  # SegFormer fine-tuning
│   └── README.md                        # Notebook results overview
└── README.md                            # This file
```

---

## 6. System Design

**Three-tier architecture:**

```
┌─────────────────────────────────────────────┐
│  Local Workstation (Preprocessing)          │
│  • QGIS: fault mask generation              │
│  • Python: patch extraction                 │
│  • Output: .npz files                       │
└─────────────────┬───────────────────────────┘
                  │ upload (cloud sync)
                  ↓
┌─────────────────────────────────────────────┐
│  GPU Compute (Training)                     │
│  • SegFormer-B2 fine-tuning                 │
│  • Per-region specialist models             │
│  • Output: model checkpoints + metrics      │
└─────────────────┬───────────────────────────┘
                  │ upload (cloud sync)
                  ↓
┌─────────────────────────────────────────────┐
│  Cloud Inference (Dashboard)                │
│  • Google Cloud Run (stateless containers)  │
│  • Google Cloud Storage (results CSV)       │
│  • Public Dash app for visualization        │
└─────────────────────────────────────────────┘
```

**Design principles:**
- **Separation of concerns:** Each tier handles one role (data prep / training / serving)
- **Stateless inference:** Cloud Run scales horizontally, results loaded from GCS at runtime
- **Reproducibility:** All preprocessing scripts version-controlled, training configs in code
- **Portability:** Cloud-agnostic data formats (NPZ, JSON, CSV); training script runs on any CUDA GPU

---

## 7. Inference Service

A live Dash dashboard hosted on Google Cloud Run visualizes:
- Per-region experiment progress (IoU comparisons)
- Patch-size ablation results
- Pipeline visualization (QGIS preprocessing steps)
- Architecture comparison (U-Net vs SegFormer)

**URL:** [prithvi-app-88345939641.us-west2.run.app](https://prithvi-app-88345939641.us-west2.run.app)

The dashboard reads experiment results from GCS (`cs163class.appspot.com/fault_detection_results.csv`) at runtime, so it always reflects the latest experiments without redeployment.

---

## 8. Cloud Data Storage

**Google Cloud Storage:** `gs://cs163class.appspot.com/`

```
images/
  hillshade_lidar.png            # Carrizo LiDAR hillshade example
  qgis_usmap.png                 # USGS Quaternary Faults overview
  qgis_faultlines.png            # Clipped + buffered fault lines
  qgis_combined.png              # Final binary mask on basemap
  satellite_carrizo.png          # HLS Sentinel-2 over Carrizo
  prithvi_huggingface.png        # Prithvi foundation model demo
  phase0_bad_prediction.png      # Failed Phase 0 example
  unet_test_predictions.png      # U-Net DEM test predictions
fault_detection_results.csv      # All experiment metrics
```

All files publicly accessible via `https://storage.googleapis.com/cs163class.appspot.com/<file>`.

---

## 9. Geological Insights

The model's performance correlates strongly with fault kinematics (movement type):

```
ECSZ (Vertical displacement):
  Owens Valley:  IoU 0.486
  Mojave:        IoU 0.393
  Average: 0.440

San Andreas (Strike-slip):
  Carrizo:        IoU 0.369
  Imperial:       IoU 0.324
  Average: 0.347

Reverse fault systems:
  Pauma Valley:   IoU 0.238
  Sierra Pelona:  IoU 0.121
  Average: 0.180
```

**Hypothesis:** Normal/oblique faults with strong vertical displacement create clear shadow boundaries in LiDAR hillshade data, making them easier for the model to detect. Pure strike-slip faults (horizontal motion only) and reverse faults with high sediment cover produce weaker visual signals.

**Implication for future work:** ECSZ region appears to be the most promising area for LiDAR-based automated fault mapping. Strike-slip and reverse fault detection may benefit from additional data modalities (InSAR, multi-temporal LiDAR).

---

## Data Sources

- **OpenTopography** — High-resolution LiDAR data access
- **USGS** — Quaternary Fault and Fold Database
- **NASA HLS** — Sentinel-2 satellite imagery
- **IBM/NASA Prithvi-EO-2.0** — Geospatial foundation model

---

## Citation

```
EarthScope Southern & Eastern California LiDAR Project.
Distributed by OpenTopography. https://doi.org/10.5069/G9G44N6Q

USGS Quaternary Fault and Fold Database of the United States.
https://www.usgs.gov/programs/earthquake-hazards/faults
```

## License

CC BY 4.0
