"""
01_download_dem.py

Download USGS 3DEP 1 m LiDAR DEM tiles, merge, clip to a bbox, and compute
hillshade and slope.

No API key required (USGS TNM is public). Run on a machine with enough RAM and
internet access.

Outputs:
  data/raw/dem_1m.tif
  data/processed/hillshade.tif
  data/processed/slope.tif
"""

import numpy as np
import requests
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from shapely.geometry import box as shapely_box
from pathlib import Path
import geopandas as gpd

SOUTH, NORTH = 37.58, 37.75
WEST, EAST = -119.10, -118.85

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
TILES_DIR = RAW_DIR / "tiles_1m"


def download_tiles():
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying USGS TNM for 1m DEM tiles...")
    resp = requests.get(
        "https://tnmaccess.nationalmap.gov/api/v1/products",
        params={
            "bbox": f"{WEST},{SOUTH},{EAST},{NORTH}",
            "datasets": "Digital Elevation Model (DEM) 1 meter",
            "prodFormats": "GeoTIFF",
            "max": 50,
        },
        timeout=60,
    )
    resp.raise_for_status()

    items = [it for it in resp.json().get("items", []) if it.get("downloadURL")]
    print(f"  Found {len(items)} tiles")

    tile_paths = []
    for i, item in enumerate(items):
        url = item["downloadURL"]
        filename = url.split("/")[-1]
        tile_path = TILES_DIR / filename

        if tile_path.exists() and tile_path.stat().st_size > 1000:
            print(f"  [{i+1}/{len(items)}] {filename} (cached)")
        else:
            print(f"  [{i+1}/{len(items)}] Downloading {filename}...")
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            tile_path.write_bytes(r.content)
            print(f"    {len(r.content)/1e6:.1f} MB")

        tile_paths.append(tile_path)

    return tile_paths


def merge_and_clip(tile_paths):
    print(f"\nMerging {len(tile_paths)} tiles...")
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, mosaic_transform = merge(datasets)
    meta = datasets[0].meta.copy()
    src_crs = datasets[0].crs
    for ds in datasets:
        ds.close()

    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                transform=mosaic_transform)

    merged_path = RAW_DIR / "dem_1m_merged.tif"
    with rasterio.open(merged_path, "w", **meta) as dst:
        dst.write(mosaic)
    del mosaic
    print(f"  Merged: {merged_path} ({merged_path.stat().st_size/1e9:.2f} GB)")

    print("Clipping to bbox...")
    bbox_4326 = shapely_box(WEST, SOUTH, EAST, NORTH)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_4326], crs="EPSG:4326").to_crs(src_crs)

    with rasterio.open(merged_path) as src:
        clipped, clipped_transform = rio_mask(src, [bbox_gdf.geometry.iloc[0]], crop=True)
        clip_meta = src.meta.copy()

    clip_meta.update(height=clipped.shape[1], width=clipped.shape[2],
                     transform=clipped_transform)

    dem_path = RAW_DIR / "dem_1m.tif"
    with rasterio.open(dem_path, "w", **clip_meta) as dst:
        dst.write(clipped)
    del clipped

    merged_path.unlink()
    print(f"  Clipped: {dem_path} ({dem_path.stat().st_size/1e6:.0f} MB)")
    return dem_path


def compute_derivatives(dem_path):
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        meta = src.meta.copy()
        cell_m = src.res[0]

    print(f"\nDEM: {dem.shape[0]}x{dem.shape[1]}, cell={cell_m}m")

    nodata_mask = (dem == nodata) if nodata is not None else np.zeros_like(dem, dtype=bool)
    dem[nodata_mask] = np.nan

    dy, dx = np.gradient(dem, cell_m)

    # Hillshade (azimuth=315, altitude=45)
    az_rad = np.radians(360 - 315 + 90)
    alt_rad = np.radians(45)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    hs = (np.sin(alt_rad) * np.cos(slope_rad)
          + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad))
    hs = np.clip(hs * 255, 0, 255).astype(np.uint8)
    hs[nodata_mask] = 0

    hs_meta = meta.copy()
    hs_meta.update(dtype="uint8", nodata=0, count=1)
    hs_path = PROC_DIR / "hillshade.tif"
    with rasterio.open(hs_path, "w", **hs_meta) as dst:
        dst.write(hs, 1)
    print(f"  Hillshade: {hs_path}")

    # Slope (degrees)
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[nodata_mask] = -1

    sl_meta = meta.copy()
    sl_meta.update(dtype="float32", nodata=-1, count=1)
    sl_path = PROC_DIR / "slope.tif"
    with rasterio.open(sl_path, "w", **sl_meta) as dst:
        dst.write(slope_deg, 1)
    print(f"  Slope:     {sl_path}")


def main():
    tile_paths = download_tiles()
    dem_path = merge_and_clip(tile_paths)
    compute_derivatives(dem_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
