# Fault Detection with LiDAR DEM + Deep Learning

Automated detection of active fault zones in California using high-resolution LiDAR DEMs. Built for the [SCEC (Southern California Earthquake Center)](https://www.scec.org/) seismic hazard research project at SJSU under Professor Kim Blisniuk.

🌐 **Live Dashboard**: [infser-88345939641.us-west2.run.app](https://infser-88345939641.us-west2.run.app/)

---

## What This Is

I'm working on two objectives from the SCEC research proposal:

- **Objective 1B**: Detect fault zones from LiDAR-derived DEM (hillshade + slope) using a segmentation model
- **Objective 1A**: Detect fault zones from HLS Sentinel-2 satellite imagery using Prithvi-EO-2.0

The core idea is that active faults leave physical scars on the landscape — you can see them as linear shadow boundaries in hillshade images and as abrupt slope changes. A well-trained segmentation model can learn to find these signatures automatically, which could help with large-scale fault mapping that would otherwise take geologists a long time to do manually.

---

## Results So Far

### Objective 1B — DEM-based Detection

I started with Parkfield (where the San Andreas Fault runs through) and gradually expanded to 4 California regions.

| Model | Region | Patch Size | IoU_fault | F1 | Notes |
|---|---|---|---|---|---|
| U-Net ResNet34 | Parkfield | 128×128 | 0.385 | 0.556 | Baseline |
| U-Net ResNet34 | Parkfield | 128×128 | 0.091 | 0.167 | 1m buffer (stricter GT) |
| U-Net ResNet34 | Carrizo | 128×128 | 0.130 | 0.230 | Per-region training |
| U-Net ResNet34 | Carrizo | 256×256 | 0.251 | 0.402 | +93% vs 128×128 |
| U-Net ResNet34 | Carrizo | 512×512 | 0.266 | 0.420 | Diminishing returns |
| **SegFormer-B2** | **Carrizo** | **256×256** | **0.369** | **0.539** | **Best result** |

The biggest improvements came from:
1. Using 256×256 patches instead of 128×128 — fault traces need enough spatial context to be recognizable
2. Switching from U-Net to SegFormer-B2 — pretrained on ADE20K segmentation, not just ImageNet classification

### Objective 1A — Satellite-based Detection

| Model | Patch Size | Resolution | IoU_fault |
|---|---|---|---|
| Prithvi-EO-2.0-300M | 128×128 | HLS 30m | 0.040 |
| Prithvi-EO-2.0-300M | 128×128 | HLS 10m | 0.071 |
| Prithvi-EO-2.0-300M | 256×256 | HLS 10m | 0.080 |

Still limited by data size — only ~71 fault training patches from the Parkfield area. Need to expand to more regions.

---

## Key Learnings

**What worked:**
- Training regions separately rather than combining them — each terrain type (flat plains, desert, mountains) produces very different DEM signatures and a single model can't learn all of them at once (combined training gave IoU = 0.032)
- Larger patches (256×256 = 128m coverage) vs smaller (128×128 = 64m) — nearly doubled performance because fault traces need room to be distinguishable from random terrain variation
- SegFormer's Transformer attention captures long-range linear patterns better than U-Net's local convolutions

**What didn't work:**
- Coachella Valley — the desert terrain has too many linear features that look like faults (roads, irrigation channels, alluvial fan edges). The model kept getting confused. Still haven't solved this.
- 1m buffer ground truth — technically more precise but only 2 pixels wide at 0.5m resolution, which is too thin for the model to learn reliably

**Hyperparameter tuning (SegFormer):**
- `encoder_lr = 1e-5` is important — if you update the encoder too fast, it forgets what it learned on ADE20K
- `fault_weight = 5.0` for the loss function class weighting (fault pixels are ~1% of the data)

---

## Data

- **LiDAR DEM**: OpenTopography airborne LiDAR, 0.5m resolution — 4 regions in California
- **Ground truth labels**: USGS Quaternary Fault and Fold Database, 10m buffer, rasterized at 0.5m
- **Satellite imagery**: HLS Sentinel-2 via NASA Earthdata, 4 seasons × 6 bands
- **Total patches generated**: ~252,000 across 7 datasets

---

## What's Next

A few things I want to try:

1. **More data for Carrizo** — download additional LiDAR tiles from OpenTopography along the same fault strand to push the fault patch count above 1000+ for 256×256 training
2. **Curriculum Learning** — start with Carrizo (simple flat terrain), then transfer to Sierra Pelona (mountainous), then attempt Coachella (complex desert) rather than treating all regions equally
3. **SegFormer-B5** — larger model, should improve further if the data is sufficient
4. **Expand Obj 1A** — get HLS data for Carrizo and Sierra Pelona so Prithvi has more than 71 fault patches to learn from
5. **Obj 1C (fusion)** — eventually combine the DEM and satellite approaches into one model

---

## Repo Structure

```
notebooks/
  obj1A/
    obj1A_128x128.ipynb          # Prithvi HLS 128×128 (IoU=0.071)
    obj1A_256x256.ipynb          # Prithvi HLS 256×256 (IoU=0.080)
  obj1B/
    DEM_Unet.ipynb               # U-Net baseline (IoU=0.385, Parkfield 10m)
    DEM_buffer1m.ipynb           # Buffer=1m experiment (IoU=0.091)
    fault_detection_final.ipynb  # Per-region + Transfer Learning
    segformer_fault_detection.ipynb  # SegFormer-B2 fine-tuning (IoU=0.369)
  preprocessing/
    USGS_1m_dataset.ipynb        # Patch generation pipeline
```

---

## Tech Stack

- **Training**: PyTorch, segmentation-models-pytorch, HuggingFace transformers
- **Preprocessing**: QGIS (fault masks), rasterio, geopandas
- **Compute**: Google Colab A100 / L4
- **Storage**: Google Drive
- **Dashboard**: Plotly Dash, deployed on Google Cloud Run

---

## Environment

```bash
pip install torch torchvision transformers segmentation-models-pytorch \
            rasterio geopandas albumentations optuna
```

Training was done on Google Colab (A100-SXM4-80GB for main runs, L4 for hyperparameter sweeps). Local preprocessing on Mac Mini M2 Pro using QGIS for fault mask generation.