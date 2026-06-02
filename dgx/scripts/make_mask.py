"""
Generic fault mask generator.

Usage:
    python3 make_mask.py <region_name>

Reads  data/<region>/processed/viz.be_hillshade.tif
Writes data/<region>/processed/<region>_fault_mask.tif

The mask is built from the USGS Quaternary Fault Database: fault lines are
clipped to the DEM extent, buffered by 10 m, dissolved, and rasterized at the
DEM resolution (0.5 m) so the mask aligns pixel-for-pixel with the hillshade.
"""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

if len(sys.argv) < 2:
    print("Usage: python3 make_mask.py <region_name>")
    sys.exit(1)
REGION = sys.argv[1]

PROC = Path(f"data/{REGION}/processed")
HILLSHADE = PROC / "viz.be_hillshade.tif"
QFAULTS = Path("data/raw/qfaults/SHP/Qfaults_US_Database.shp")
OUT_MASK = PROC / f"{REGION}_fault_mask.tif"
BUFFER_M = 10.0

with rasterio.open(HILLSHADE) as src:
    grid_crs, grid_transform = src.crs, src.transform
    grid_shape = (src.height, src.width)
    grid_bounds = src.bounds
print(f"[{REGION}] grid {grid_shape}, crs={grid_crs}")

# load faults, reproject to the DEM CRS, keep only those intersecting the tile
gdf = gpd.read_file(QFAULTS).to_crs(grid_crs)
bbox = box(grid_bounds.left, grid_bounds.bottom, grid_bounds.right, grid_bounds.top)
clip = gdf[gdf.intersects(bbox)].copy()
print(f"[{REGION}] faults in bbox: {len(clip)}")
if len(clip) == 0:
    print("ERROR: no faults found in this region"); sys.exit(1)
if 'fault_name' in clip.columns:
    print(f"  names: {list(clip['fault_name'].dropna().unique()[:5])}")

# buffer + dissolve + rasterize at DEM resolution
clip['geometry'] = clip.geometry.buffer(BUFFER_M)
dissolved = clip.dissolve()
mask = rasterize([(g, 1) for g in dissolved.geometry], out_shape=grid_shape,
                 transform=grid_transform, fill=0, all_touched=True, dtype=np.uint8)
pct = 100 * mask.sum() / mask.size
print(f"[{REGION}] fault coverage: {pct:.3f}%")

with rasterio.open(HILLSHADE) as src:
    profile = src.profile.copy()
profile.update(dtype=np.uint8, count=1, nodata=0)
with rasterio.open(OUT_MASK, 'w', **profile) as dst:
    dst.write(mask, 1)
print(f"[{REGION}] saved: {OUT_MASK}")
