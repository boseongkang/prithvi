"""
Scan fault density inside the LiDAR footprint.

Uses the LiDAR tile index (SoCAL_TileIndex.shp) to clip the scanning grid to the
area actually covered by LiDAR, then slides an 11 km window (5.5 km stride) and
scores each tile by USGS fault coverage. Prints the top-N fault-dense tiles so
their coordinates can be fed straight into OpenTopography.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import box

QFAULTS = "data/raw/qfaults/SHP/Qfaults_US_Database.shp"
LIDAR_BOUNDARY = "data/SoCAL_TileIndex.shp"
BUFFER_M = 10.0
TILE_KM = 11.0
STRIDE_KM = 5.5
TOP_N = 20

# load LiDAR footprint
print("[load] LiDAR footprint...")
lidar = gpd.read_file(LIDAR_BOUNDARY).to_crs("EPSG:32611")
lidar_geom = lidar.union_all()
print(f"[load] LiDAR area: {lidar_geom.area/1e6:.0f} km^2")
lb = lidar.total_bounds
print(f"[load] LiDAR bbox: X[{lb[0]:.0f}, {lb[2]:.0f}] Y[{lb[1]:.0f}, {lb[3]:.0f}]")

# load faults, keep those inside the footprint bbox
print("[load] QFaults...")
gdf = gpd.read_file(QFAULTS).to_crs("EPSG:32611")
region_box = box(lb[0], lb[1], lb[2], lb[3])
gdf = gdf[gdf.intersects(region_box)].copy()
print(f"[load] faults in region: {len(gdf)}")

# slide a window over the footprint
tile_m = TILE_KM * 1000
stride_m = STRIDE_KM * 1000
results = []

x = lb[0]
while x + tile_m <= lb[2]:
    y = lb[1]
    while y + tile_m <= lb[3]:
        tile = box(x, y, x + tile_m, y + tile_m)
        overlap = tile.intersection(lidar_geom).area / tile.area
        if overlap < 0.2:                      # skip tiles barely covered by LiDAR
            y += stride_m
            continue
        clip = gdf[gdf.intersects(tile)]
        if len(clip) > 0:
            clipped = gpd.clip(clip, tile)
            if len(clipped) > 0:
                buffered = clipped.buffer(BUFFER_M).union_all()
                pct = 100 * buffered.area / (tile_m * tile_m)
                results.append({
                    'xmin': x, 'ymin': y, 'xmax': x + tile_m, 'ymax': y + tile_m,
                    'n_fault': len(clip), 'pct': pct, 'lidar_overlap': overlap,
                    'names': list(clip['fault_name'].dropna().unique()[:2]) if 'fault_name' in clip.columns else []
                })
        y += stride_m
    x += stride_m

results.sort(key=lambda r: r['pct'], reverse=True)

print(f"\n{len(results)} tiles inside LiDAR. Top {TOP_N}:\n")
print(f"{'rank':>4} {'fault%':>7} {'n':>4} {'lidar%':>7}  {'xmin':>9} {'ymin':>10} {'xmax':>9} {'ymax':>10}  names")
print("-" * 110)
for i, r in enumerate(results[:TOP_N]):
    names = ', '.join(r['names'])[:30]
    print(f"{i+1:>4} {r['pct']:>6.3f}% {r['n_fault']:>4} {100*r['lidar_overlap']:>6.1f}%  "
          f"{r['xmin']:>9.0f} {r['ymin']:>10.0f} {r['xmax']:>9.0f} {r['ymax']:>10.0f}  {names}")

print("\nLiDAR coverage confirmed. Download the top tiles by their coordinates.")
