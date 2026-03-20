## Visualizations

### Training Curves
<img width="1389" height="490" alt="Image" src="https://github.com/user-attachments/assets/2512aae9-39a1-4526-881d-4434a0cd027b" />


**Left**: Train loss decreases steadily from ~1.4 to ~0.07 over 47 epochs, indicating stable learning.  
**Right**: Validation metrics over training. mIoU (green) stabilizes around 0.52. IoU_fault (red dashed) and F1 (blue dotted) remain low at threshold=0.5, but reach **0.40 and 0.57** respectively after threshold tuning to 0.35 (see Test Results).

---

### Test Predictions
<img width="1559" height="2357" alt="Image" src="https://github.com/user-attachments/assets/f7059837-7f58-4e94-84d9-6f8eef57da29" />

Each row shows one 128×128 patch from the test set containing a fault. Columns from left to right:

| Column | Description |
|--------|-------------|
| **Hillshade** | Input LiDAR-derived hillshade image. The fault trace is visible as a linear landform. |
| **Ground Truth** | USGS fault mask (10m buffer, rasterized at 0.5m). The diagonal band reflects the fault trace crossing the patch. |
| **Pred (prob)** | Model predicted fault probability (0–1). Brighter = higher confidence. |
| **Overlay (t=0.50)** | Prediction at threshold=0.50 overlaid on hillshade. |

**Observations**:
- Rows 3–4: The model correctly identifies the fault zone direction and location.
- Rows 1–2, 5–6: Fault signal is weaker; model confidence stays below threshold=0.50.
- After lowering threshold to **0.35**, IoU_fault improves from 0.066 to **0.385** on the full test set, indicating the model predicts fault regions with moderate but consistent probability.


DEM_buffer1m update
## Objective 1B — DEM-based Fault Detection
<img width="1388" height="490" alt="Image" src="https://github.com/user-attachments/assets/312dcb4f-4432-40f0-8642-4e1649ac32fa" />

<img width="1559" height="2357" alt="Image" src="https://github.com/user-attachments/assets/c6c84978-e1eb-4689-b9c8-a1aae1caaed8" />


### DEM_Unet.ipynb (buffer=10m)
- IoU_fault: 0.385, F1: 0.556

### DEM_buffer1m.ipynb (buffer=1m + Boundary Loss)
- IoU_fault: 0.091 (strict), 0.118 (5m tolerance)
- [training curve image]
- [test predictions image]

## Objective 1A — Prithvi + HLS

### obj1A_256x256.ipynb
- Patch size: 256×256
- IoU_fault: 0.080 (val, threshold=0.50)
