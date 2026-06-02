"""
Estimate fault count and coverage for a candidate region before downloading.

Given a bounding box (UTM 11N, EPSG:32611), reports how many USGS faults fall
inside, their slip sense and names, and an estimated fault-coverage percentage
(buffered fault area / bbox area). Use this to decide whether a tile is worth
downloading from OpenTopography.

Usage:
    python3 check_fault_count.py <xmin> <ymin> <xmax> <ymax>
"""
import sys
import geopandas as gpd
from shapely.geometry import box

QFAULTS = "data/raw/qfaults/SHP/Qfaults_US_Database.shp"
BUFFER_M = 10.0

XMIN = float(sys.argv[1])
YMIN = float(sys.argv[2])
XMAX = float(sys.argv[3])
YMAX = float(sys.argv[4])

print(f"[bbox] X[{XMIN:.0f}, {XMAX:.0f}] Y[{YMIN:.0f}, {YMAX:.0f}]")
area_km2 = (XMAX - XMIN) * (YMAX - YMIN) / 1e6
print(f"[area] {area_km2:.0f} km^2")

# load faults, reproject to UTM, clip to bbox
gdf = gpd.read_file(QFAULTS).to_crs("EPSG:32611")
bbox = box(XMIN, YMIN, XMAX, YMAX)
clip = gdf[gdf.intersects(bbox)].copy()
print(f"\n[fault] faults in bbox: {len(clip)}")

if len(clip) == 0:
    print("  -> no faults here, try another tile.")
    sys.exit(0)

if 'slip_sense' in clip.columns:
    print("[slip_sense]")
    print(clip['slip_sense'].value_counts().to_string())
if 'fault_name' in clip.columns:
    names = clip['fault_name'].dropna().unique()[:8]
    print(f"[names] {list(names)}")

# estimate coverage: buffered fault area / bbox area
clip_box = gpd.clip(clip, bbox)
buffered = clip_box.buffer(BUFFER_M).union_all()
fault_area = buffered.area
total_area = (XMAX - XMIN) * (YMAX - YMIN)
pct = 100 * fault_area / total_area
print(f"\n[estimated fault coverage] {pct:.3f}%  (reference: Owens 0.93%, Big Pine 1.257%)")
if pct < 0.5:
    print("  low - prefer a tile with more fault length.")
elif pct < 1.0:
    print("  moderate - may work.")
else:
    print("  good - comparable to Big Pine.")
