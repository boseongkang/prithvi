"""
make_patches.py <region>  -  5-channel version

Channels: ch0 hillshade, ch1 slope, ch2 sin(aspect), ch3 cos(aspect), ch4 roughness
Output:   (5, 256, 256) float32 patches

Patches are split into train/val/test by spatial blocks (not random pixels), so
no patch leaks across the split boundary. Patches that are mostly nodata are
skipped.
"""

import sys
import numpy as np
import rasterio
from pathlib import Path
import shutil

if len(sys.argv) < 2:
    print("Usage: python3 make_patches.py <region_name>")
    sys.exit(1)
REGION = sys.argv[1]
SRC_DIR = Path(f"data/{REGION}/processed")
HILLSHADE_PATH = SRC_DIR / "viz.be_hillshade.tif"
SLOPE_PATH = SRC_DIR / "viz.be_slope.tif"
ASPECT_PATH = SRC_DIR / "viz.be_aspect.tif"
ROUGHNESS_PATH = SRC_DIR / "viz.be_roughness.tif"
MASK_PATH = SRC_DIR / f"{REGION}_fault_mask.tif"
PATCH_DIR = Path(f"data/{REGION}/patches")

PATCH_SIZE = 256
STRIDE = 128
NODATA_THRESH = 0.10
SPLIT_BLOCK_PX = 2000
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
N_CHANNELS = 5


def load_rasters():
    """Load 5 channels: hillshade, slope, sin(aspect), cos(aspect), roughness."""
    with rasterio.open(HILLSHADE_PATH) as src:
        hillshade = src.read(1).astype(np.float32)
        hs_nodata = src.nodata
    with rasterio.open(SLOPE_PATH) as src:
        slope = src.read(1).astype(np.float32)
    with rasterio.open(ASPECT_PATH) as src:
        aspect = src.read(1).astype(np.float32)
    with rasterio.open(ROUGHNESS_PATH) as src:
        roughness = src.read(1).astype(np.float32)
    with rasterio.open(MASK_PATH) as src:
        mask = src.read(1)

    def clean(arr, fill=0.0):
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
        arr = np.where(arr == -9999, fill, arr)
        return arr

    # hillshade: nodata is 0, keep as-is (0-255)
    hillshade = clean(hillshade)
    # slope: -9999 -> 0, negatives -> 0
    slope = clean(slope)
    slope = np.where(slope < 0, 0.0, slope)
    # roughness: -9999 -> 0
    roughness = clean(roughness)
    # aspect: -9999/NaN means undefined (flat ground). Only encode valid 0-360.
    aspect_valid = (aspect != -9999) & (~np.isnan(aspect))
    aspect_filled = np.where(aspect_valid, aspect, 0.0)
    aspect_rad = np.deg2rad(aspect_filled)
    aspect_sin = np.where(aspect_valid, np.sin(aspect_rad), 0.0).astype(np.float32)
    aspect_cos = np.where(aspect_valid, np.cos(aspect_rad), 0.0).astype(np.float32)

    print(f"Hillshade: {hillshade.shape}, range=[{hillshade.min():.1f}, {hillshade.max():.1f}]")
    print(f"Slope:     range=[{slope.min():.2f}, {slope.max():.2f}]")
    print(f"Aspect:    valid={aspect_valid.mean()*100:.1f}%, sin=[{aspect_sin.min():.2f},{aspect_sin.max():.2f}], cos=[{aspect_cos.min():.2f},{aspect_cos.max():.2f}]")
    print(f"Roughness: range=[{roughness.min():.3f}, {roughness.max():.3f}]")
    print(f"Mask:      {mask.shape}, fault%={np.mean(mask == 1) * 100:.3f}")

    assert hillshade.shape == slope.shape == aspect.shape == roughness.shape == mask.shape, "Shape mismatch"
    channels = [hillshade, slope, aspect_sin, aspect_cos, roughness]
    return channels, mask, hs_nodata


def assign_blocks_to_splits(h, w, block_size, ratios, seed):
    rng = np.random.RandomState(seed)
    min_blocks = 6
    while (h + block_size - 1) // block_size * ((w + block_size - 1) // block_size) < min_blocks and block_size > PATCH_SIZE:
        block_size = block_size // 2

    n_block_rows = (h + block_size - 1) // block_size
    n_block_cols = (w + block_size - 1) // block_size
    total_blocks = n_block_rows * n_block_cols

    indices = np.arange(total_blocks)
    rng.shuffle(indices)
    n_train = int(total_blocks * ratios["train"])
    n_val = int(total_blocks * ratios["val"])

    split_map = {}
    for i, idx in enumerate(indices):
        br = idx // n_block_cols
        bc = idx % n_block_cols
        if i < n_train:
            split_map[(br, bc)] = "train"
        elif i < n_train + n_val:
            split_map[(br, bc)] = "val"
        else:
            split_map[(br, bc)] = "test"

    print(f"\nSpatial blocks: {n_block_rows}x{n_block_cols} = {total_blocks} blocks ({block_size}px each)")
    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        counts[s] += 1
    print(f"Block assignment: {counts}")
    return split_map, n_block_cols, block_size


def extract_patches(channels, mask, hs_nodata, split_map, block_size):
    h, w = channels[0].shape
    hillshade = channels[0]  # used for nodata test

    if PATCH_DIR.exists():
        shutil.rmtree(PATCH_DIR)
    for split in ["train", "val", "test"]:
        (PATCH_DIR / split / "images").mkdir(parents=True)
        (PATCH_DIR / split / "masks").mkdir(parents=True)

    stats = {s: {"total": 0, "fault": 0, "bg": 0, "skipped_nodata": 0} for s in ["train", "val", "test"]}
    patch_id = 0

    for r in range(0, h - PATCH_SIZE + 1, STRIDE):
        for c in range(0, w - PATCH_SIZE + 1, STRIDE):
            hs_patch = hillshade[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            mk_patch = mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            br = r // block_size
            bc = c // block_size
            split = split_map.get((br, bc), "train")

            if hs_nodata is not None:
                nodata_frac = np.mean(hs_patch == hs_nodata)
                if nodata_frac > NODATA_THRESH:
                    stats[split]["skipped_nodata"] += 1
                    continue

            # stack all channels -> (5, 256, 256)
            img = np.stack([
                ch[r:r+PATCH_SIZE, c:c+PATCH_SIZE].astype(np.float32)
                for ch in channels
            ], axis=0)

            has_fault = np.any(mk_patch == 1)
            if has_fault:
                stats[split]["fault"] += 1
            else:
                stats[split]["bg"] += 1
            stats[split]["total"] += 1

            fname = f"patch_{patch_id:05d}.npy"
            np.save(PATCH_DIR / split / "images" / fname, img)
            np.save(PATCH_DIR / split / "masks" / fname, mk_patch.astype(np.uint8))
            patch_id += 1

    return stats


def print_stats(stats):
    print(f"\n=== Patch Extraction Results ({REGION}, {N_CHANNELS}-channel) ===\n")
    print(f"{'Split':8s} {'Total':>6s} {'Fault':>6s} {'BG':>6s} {'Fault%':>7s} {'Skipped':>8s}")
    print("-" * 45)
    grand_total = 0
    grand_fault = 0
    for split in ["train", "val", "test"]:
        s = stats[split]
        frac = s["fault"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"{split:8s} {s['total']:6d} {s['fault']:6d} {s['bg']:6d} {frac:6.1f}% {s['skipped_nodata']:8d}")
        grand_total += s["total"]
        grand_fault += s["fault"]
    grand_frac = grand_fault / grand_total * 100 if grand_total > 0 else 0
    print("-" * 45)
    print(f"{'TOTAL':8s} {grand_total:6d} {grand_fault:6d} {grand_total - grand_fault:6d} {grand_frac:6.1f}%")


def main():
    print("=== Loading rasters (5ch: hillshade, slope, sin/cos aspect, roughness) ===")
    channels, mask, hs_nodata = load_rasters()

    h, w = channels[0].shape
    split_map, _, block_size = assign_blocks_to_splits(
        h, w, SPLIT_BLOCK_PX, SPLIT_RATIOS, SEED
    )

    print(f"\n=== Extracting {PATCH_SIZE}x{PATCH_SIZE} patches (stride={STRIDE}, {N_CHANNELS}ch) ===")
    stats = extract_patches(channels, mask, hs_nodata, split_map, block_size)
    print_stats(stats)

    print(f"\nSaved to: {PATCH_DIR}/")


if __name__ == "__main__":
    main()
