# Fault Detection with LiDAR DEM + Deep Learning

Automated detection of active fault zones in California using high-resolution LiDAR DEMs and transformer-based semantic segmentation. Built for the [SCEC (Southern California Earthquake Center)](https://www.scec.org/) seismic hazard research project at SJSU under Professor Kim Blisniuk.

🌐 **Live Dashboard**: [infser-88345939641.us-west2.run.app](https://infser-88345939641.us-west2.run.app/)

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Setup Instructions](#2-setup-instructions)
3. [Full Pipeline](#3-full-pipeline)
4. [Repository Structure](#4-repository-structure)
5. [System Design](#5-system-design)
6. [Inference Service](#6-inference-service)
7. [Cloud Data Storage](#7-cloud-data-storage)
8. [Results](#8-results)

---

## 1. What This Is

I'm working on two objectives from the SCEC research proposal:

- **Objective 1B**: Detect fault zones from LiDAR-derived DEM (hillshade + slope) using a segmentation model
- **Objective 1A**: Detect fault zones from HLS Sentinel-2 satellite imagery using Prithvi-EO-2.0

The core idea is that active faults leave physical scars on the landscape — visible as linear shadow boundaries in hillshade images and as abrupt slope changes. A well-trained segmentation model can learn to find these signatures automatically, enabling large-scale fault mapping that would otherwise require months of manual geological survey work.

**Study Regions:** Parkfield · Carrizo Plain · Coachella Valley · Sierra Pelona (all California)

---

## 2. Setup Instructions

### Requirements

```
Python >= 3.10
CUDA-capable GPU (training was run on Colab A100 / L4)
Google Drive (for checkpoint and patch storage)
QGIS >= 3.x (for fault mask generation — one-time preprocessing only)
```

### Install Dependencies

```bash
pip install torch torchvision transformers segmentation-models-pytorch \
            rasterio geopandas albumentations \
            dash plotly flask gunicorn google-cloud-storage
```

Or using the requirements file:

```bash
git clone https://github.com/boseongkang/prithvi
cd prithvi
pip install -r requirements.txt
```

### Google Colab Environment

All training notebooks mount Google Drive for patch data and checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR   = '/content/drive/MyDrive/prithvi_fault/'
PATCH_BASE = BASE_DIR + 'data/patches/'
CKPT_BASE  = BASE_DIR + 'checkpoints/'
```

### Run Dashboard Locally

```bash
python dashapps/app4.py      # starts dev server at http://localhost:8050
```

---

## 3. Full Pipeline

```
Raw Data Collection
    │
    ├── LiDAR DEM (0.5 m resolution)
    │       Source: OpenTopography / EarthScope NorCal LiDAR Project
    │       Format: GeoTIFF, EPSG:32611
    │
    └── Fault Vectors
            Source: USGS Quaternary Fault and Fold Database
            Format: Shapefile (112,809 features, US-wide)
              │
              ▼
QGIS Preprocessing  (one-time, Mac Mini M2 Pro local)
    ├── 1. Reproject shapefile → EPSG:32611 (metric CRS for buffering)
    ├── 2. Extract Layer Extent per study region (clip to DEM bounds)
    ├── 3. Buffer fault lines: 10 m (initial) / 1 m (strict) with Dissolve
    └── 4. Rasterize → 0.5 m GeoTIFF binary mask  (1 = Fault, 0 = Background)
              │
              ▼
Patch Extraction  (preprocessing/patch_extractor.py)
    ├── Compute DEM derivatives: hillshade (×2), slope → 3-channel input tensor
    ├── Tile into overlapping patches: 128×128 / 256×256 / 512×512
    ├── Hard Negative Mining: fault-free patches sampled at 10% to reduce 2700:1 imbalance
    ├── Normalize per-channel (mean/std from training split)
    └── Save: train.npz / val.npz / test.npz  (70 / 15 / 15 split)
              │
              ▼
Model Training  (Google Colab A100 / L4)
    │
    ├── Phase 1–3: U-Net ResNet34  (segmentation-models-pytorch)
    │       Loss: Weighted CrossEntropy (fault_weight=5.0) + Dice
    │       Per-region training — combined training collapsed at IoU=0.032
    │
    └── Phase 4: SegFormer-B2  (nvidia/mit-b2, ADE20K pretrained)
            Differential LR: encoder_lr=1e-5 / decoder_lr=1e-4
            Best: Carrizo 256×256 → IoU_fault=0.369  (+46.7% over U-Net)
              │
              ▼
Results & Deployment
    ├── Metrics → GCS: gs://cs163class.appspot.com/fault_detection_results.csv
    ├── Interactive dashboard: Dash + Plotly  (4 pages)
    └── Deployed on Cloud Run via Docker  (us-west2)
```

---

## 4. Repository Structure

```
prithvi/
│
├── dashapps/
│   ├── app4.py              # Main Dash app — 4-page interactive dashboard
│   │                        # Pages: /  /objective  /methods  /findings
│   │                        # Reads results CSV from GCS, renders Plotly charts
│   └── assets/              # CSS and static images for dashboard
│
├── notebooks/
│   ├── obj1A/
│   │   ├── obj1A_128x128.ipynb      # Prithvi-EO + HLS 128×128  (IoU=0.071)
│   │   └── obj1A_256x256.ipynb      # Prithvi-EO + HLS 256×256  (IoU=0.080)
│   └── obj1B/
│       ├── DEM_Unet.ipynb                   # U-Net baseline, Parkfield 128×128 10m buffer  (IoU=0.385)
│       ├── DEM_buffer1m.ipynb               # Strict 1m buffer + Boundary Loss  (IoU=0.091)
│       ├── fault_detection_final.ipynb      # Per-region training + Transfer Learning
│       └── segformer_fault_detection.ipynb  # SegFormer-B2 fine-tuning  (IoU=0.369)
│
├── preprocessing/
│   ├── patch_extractor.py       # Cuts GeoTIFF DEMs into .npz patch datasets
│   ├── hard_negative_mining.py  # Handles extreme class imbalance (~2700:1 background:fault)
│   └── qgis_workflow.md         # Step-by-step QGIS rasterization instructions
│
├── Dockerfile                   # Cloud Run container — serves dashapps/app4.py via gunicorn
├── requirements.txt
└── README.md
```

---

## 5. System Design

```
┌──────────────────────────────────────────────────────────────┐
│                        User Browser                          │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTPS
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Google Cloud Run  (us-west2)                    │
│                                                              │
│  Docker Container                                            │
│  ├── Gunicorn  (WSGI server, port 8080)                      │
│  └── Dash / Flask app  (dashapps/app4.py)                    │
│        /           — Landing page + overview                 │
│        /objective  — Research goals and data sources         │
│        /methods    — Model architecture and pipeline         │
│        /findings   — Results, Before/After, metrics          │
│                                                              │
│  Scaling: min-instances=0  max-instances=3                   │
│  Memory: 512 MiB  │  CPU: 1 vCPU                             │
└────────────────────┬─────────────────────────────────────────┘
                     │  google-cloud-storage Python client
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Google Cloud Storage (GCS)                      │
│              Bucket: cs163class.appspot.com                  │
│                                                              │
│   fault_detection_results.csv  ← all experiment metrics     │
└──────────────────────────────────────────────────────────────┘
```

**Scalability:** Cloud Run scales to zero when idle (no cost) and spins up new container instances automatically under load. The app is fully stateless — all data is fetched from GCS at request time, so horizontal scaling requires no coordination between instances. Docker images are versioned in Artifact Registry, enabling zero-downtime redeployments.

---

## 6. Inference Service

The ML inference results are served through the Dash dashboard running on Cloud Run.

| Property | Detail |
|---|---|
| **Endpoint** | `GET /findings` — renders pre-computed prediction visualizations |
| **Model** | SegFormer-B2 fine-tuned (`segformer_carrizo_256/best_segformer.pth`) |
| **Input** | 3-channel numpy array `[hillshade, slope, hillshade]`, shape `(3, 256, 256)`, normalized |
| **Output** | Binary segmentation mask `(256, 256)` — `1 = Fault`, `0 = Background` |
| **Docker entry** | `CMD ["gunicorn", "-b", "0.0.0.0:8080", "app4:server"]` |

The Dash app reads `fault_detection_results.csv` from GCS using the Cloud Run service account (no explicit credentials needed inside GCP) and renders Plotly bar charts from the pre-computed metrics. Model checkpoints are stored on Google Drive and loaded during Colab training — they are not shipped inside the container.

**Redeploy:**

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/prithvi-app
gcloud run deploy prithvi-app \
  --image gcr.io/PROJECT_ID/prithvi-app \
  --platform managed --region us-west2 \
  --allow-unauthenticated --memory 512Mi
```

---

## 7. Cloud Data Storage

All experiment results are stored in **Google Cloud Storage** and consumed by the dashboard at runtime.

**Bucket:** `gs://cs163class.appspot.com/`

| File | Description |
|---|---|
| `fault_detection_results.csv` | Per-experiment metrics: region, model, patch size, IoU, F1, mIoU, buffer, notes |

**CSV Schema:**

```
phase, region, model, patch_size, buffer_m, iou_fault, f1, miou, epochs, notes
```

**Access in app4.py:**

```python
from google.cloud import storage
from io import StringIO

client = storage.Client()
bucket = client.bucket("cs163class.appspot.com")
blob   = bucket.blob("fault_detection_results.csv")
df     = pd.read_csv(StringIO(blob.download_as_text()))
```

---

## 8. Results

### Objective 1B — DEM-based Fault Detection

| Phase | Model | Region | Patch | IoU_fault | F1 | Notes |
|---|---|---|---|---|---|---|
| 0 | U-Net ResNet50 | All 4 combined | 128 | 0.032 | 0.062 | Training collapse ❌ |
| 1 | U-Net ResNet34 | Parkfield | 128 | 0.385 | 0.556 | Baseline (10m buffer) |
| 2 | U-Net ResNet34 | Parkfield | 128 | 0.091 | 0.167 | 1m strict buffer |
| 3 | U-Net ResNet34 | Carrizo | 128 | 0.130 | 0.230 | Per-region training |
| 3 | U-Net ResNet34 | Carrizo | 256 | 0.251 | 0.402 | +93% vs 128×128 |
| **4** | **SegFormer-B2** | **Carrizo** | **256** | **0.369** | **0.539** | **Best result ✅** |
| TL | SegFormer-B2 | Sierra Pelona | 256 | 0.123 | 0.219 | Transfer from Carrizo |

### Training Curves (Parkfield Baseline)

<img width="1389" height="490" alt="Training curves" src="https://github.com/user-attachments/assets/2512aae9-39a1-4526-881d-4434a0cd027b" />

**Left**: Train loss decreases steadily from ~1.4 to ~0.07 over 47 epochs.
**Right**: Validation metrics. IoU_fault and F1 reach **0.40 and 0.57** respectively after threshold tuning to 0.35.

### Test Predictions

<img width="1559" height="2357" alt="Test predictions" src="https://github.com/user-attachments/assets/f7059837-7f58-4e94-84d9-6f8eef57da29" />

Each row: one 128×128 patch from the test set. Columns: Hillshade input → Ground Truth mask → Predicted probability → Overlay at threshold=0.50.

### Objective 1A — Prithvi-EO-2.0 + HLS Satellite Imagery

| Model | Patch Size | Resolution | IoU_fault |
|---|---|---|---|
| Prithvi-EO-2.0-300M | 128×128 | HLS 30m | 0.040 |
| Prithvi-EO-2.0-300M | 128×128 | HLS 10m | 0.071 |
| Prithvi-EO-2.0-300M | 256×256 | HLS 10m | 0.080 |

Performance is limited by data size — only ~71 fault training patches from Parkfield. Full multi-modal fusion (DEM + HLS) requires expanded HLS coverage across all study regions.

### SegFormer-B2 Hyperparameter Tuning

**encoder_lr sweep** (Carrizo 256×256):

| encoder_lr | IoU_fault | F1 |
|---|---|---|
| **1e-5** | **0.369** | **0.539** |
| 5e-5 | 0.336 | 0.504 |
| 1e-4 | 0.308 | 0.470 |

**fault_weight sweep** (encoder_lr=1e-5 fixed):

| fault_weight | IoU_fault | F1 |
|---|---|---|
| 3.0 | 0.280 | 0.438 |
| **5.0** | **0.369** | **0.539** |
| 10.0 | 0.308 | 0.470 |

### Key Learnings

**What worked:**
- Training per region — combined multi-terrain training collapses to background-only predictions (IoU = 0.032)
- Larger patches (256×256 = 128 m coverage) — nearly doubled IoU because fault traces need spatial context
- SegFormer's transformer attention captures long-range linear patterns better than U-Net's local convolutions
- ADE20K-pretrained encoder preserves pixel-level segmentation knowledge; `encoder_lr=1e-5` prevents catastrophic forgetting

**What didn't work:**
- Coachella Valley — desert terrain has too many fault-like linear features (roads, irrigation channels, alluvial fan edges). Consistent failure across all model variants.
- 1m buffer ground truth — technically more precise but produces only 2-pixel-wide labels at 0.5m resolution, which is too thin for reliable learning

---

## Data Sources

- **LiDAR DEM**: OpenTopography airborne LiDAR, 0.5m — 4 regions in California
- **Ground truth**: USGS Quaternary Fault and Fold Database, rasterized at 0.5m via QGIS
- **Satellite imagery**: HLS Sentinel-2 via NASA Earthdata, 4 seasons × 6 bands
- **Total patches**: ~252,000 across 7 datasets

---

## Tech Stack

| Layer | Tools |
|---|---|
| Training | PyTorch, segmentation-models-pytorch, HuggingFace Transformers |
| Preprocessing | QGIS, rasterio, geopandas |
| Compute | Google Colab A100 (main runs), L4 (hyperparameter sweeps) |
| Storage — training | Google Drive |
| Storage — results | Google Cloud Storage |
| Dashboard | Plotly Dash, Flask |
| Deployment | Docker, Google Cloud Run, Artifact Registry |
