"""
visualize_contour.py

Outline-style prediction figure. Instead of filling the fault mask (which is a
thick 10 m buffer and covers half the tile), this draws the ground-truth and the
prediction as contour lines over the hillshade, so the terrain stays visible.

Two columns per row:
    hillshade  |  hillshade + GT outline (red) + prediction outline (cyan)

Usage:
    python3 scripts/visualize_contour.py \
        --run-dir runs/20260531_030350 \
        --data-dir data/owens_bigpine/patches \
        --n-patches 6

Output:
    <run-dir>/contour_predictions.png
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import binary_erosion


def _load_build_model():
    spec = importlib.util.spec_from_file_location("train_mod", "scripts/04_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


def rank_val_patches(val_dir, n):
    mask_dir = Path(val_dir) / "masks"
    scored = []
    for mp in sorted(mask_dir.glob("*.npy")):
        m = np.load(mp)
        fpx = int((m == 1).sum())
        if fpx > 0:
            scored.append((fpx, mp.stem))
    scored.sort(reverse=True)
    return [stem for _, stem in scored[:n]]


def normalize(img, mean, std):
    img = img.astype(np.float32).copy()
    for c in range(img.shape[0]):
        img[c] = (img[c] - mean[c]) / (std[c] + 1e-8)
    return img


@torch.no_grad()
def predict(model, img_norm, device):
    x = torch.from_numpy(img_norm).unsqueeze(0).to(device)
    logits = model(pixel_values=x).logits
    logits = F.interpolate(logits, size=img_norm.shape[1:], mode="bilinear",
                           align_corners=False)
    return F.softmax(logits, dim=1)[:, 1].squeeze(0).cpu().numpy()


def patch_iou(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union else 0.0


def render(rows, out_path, suptitle, threshold):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 4.0 * n), squeeze=False)

    for r, row in enumerate(rows):
        hs = row["hillshade"]
        gt = row["gt"].astype(float)
        prob = row["prob"]
        pred = (prob >= threshold).astype(float)

        if hs.std() > 0:
            lo, hi = np.percentile(hs, [2, 98])
        else:
            lo, hi = 0, 255

        # col 0: hillshade only
        axes[r, 0].imshow(hs, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if r == 0:
            axes[r, 0].set_title("Hillshade", fontsize=11)
        axes[r, 0].set_ylabel(row["label"], fontsize=9)

        # col 1: hillshade + outlines
        axes[r, 1].imshow(hs, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        # contour() draws the boundary of each mask region
        if gt.any():
            axes[r, 1].contour(gt, levels=[0.5], colors="#E24B4A", linewidths=1.6)
        if pred.any():
            axes[r, 1].contour(pred, levels=[0.5], colors="#00E5E5", linewidths=1.6)
        iou = patch_iou(pred, gt)
        if r == 0:
            axes[r, 1].set_title("GT (red) vs prediction (cyan)", fontsize=11)
        axes[r, 1].text(0.97, 0.04, f"IoU {iou:.2f}", transform=axes[r, 1].transAxes,
                        ha="right", va="bottom", fontsize=9, color="#111",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

        for c in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    handles = [Line2D([0], [0], color="#E24B4A", lw=2, label="USGS fault (ground truth)"),
               Line2D([0], [0], color="#00E5E5", lw=2, label="Model prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=11, y=0.998)
    fig.tight_layout(rect=[0, 0.015, 1, 0.99])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-patches", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    channels = int(cfg["channels"])
    mean, std = cfg["mean"], cfg["std"]

    if args.threshold is not None:
        threshold = args.threshold
    else:
        met_f = run_dir / "metrics.json"
        threshold = json.loads(met_f.read_text()).get("best_threshold_pixel", 0.5) if met_f.exists() else 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Run: {run_dir.name}  channels: {channels}  threshold: {threshold:.2f}")

    build_model = _load_build_model()
    model = build_model(num_channels=channels).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    val_dir = Path(args.data_dir) / "val"
    stems = rank_val_patches(val_dir, args.n_patches)
    if not stems:
        raise RuntimeError(f"No fault patches in {val_dir}")
    print(f"Picked {len(stems)} patches")

    img_dir, mask_dir = val_dir / "images", val_dir / "masks"
    rows = []
    for stem in stems:
        img = np.load(img_dir / f"{stem}.npy")
        gt = np.load(mask_dir / f"{stem}.npy")
        prob = predict(model, normalize(img, mean, std), device)
        rows.append({"label": stem, "hillshade": img[0], "gt": gt, "prob": prob})

    out_path = args.output or str(run_dir / "contour_predictions.png")
    region = Path(args.data_dir).parts[-2] if len(Path(args.data_dir).parts) >= 2 else run_dir.name
    suptitle = f"{region}  (channels={channels}, threshold={threshold:.2f})"
    render(rows, out_path, suptitle, threshold)


if __name__ == "__main__":
    main()
