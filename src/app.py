# src/app.py
import os
import json
import random
import numpy as np
from shapely.geometry import Polygon, Point, mapping

from grid.hexgrid import generate_hex_grid, compute_hex_side_for_acres, hex_area
from assign.capacity_kmeans import assign_hexes_to_uavs

# reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- CONFIG ----------
TOTAL_ACRES = 200
ACRE_M2 = 4046.8564224
FARM_W = 1000.0
TOTAL_AREA_M2 = TOTAL_ACRES * ACRE_M2
FARM_H = TOTAL_AREA_M2 / FARM_W

NUM_UAV = 15
CELL_ACRES = 0.5
OUTPUT_DIR = "experiments/exp_001_default"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- build synthetic farm polygon ----------
farm = Polygon([(0, 0), (FARM_W, 0), (FARM_W, FARM_H), (0, FARM_H)])

# ---------- generate hex grid ----------
cells = generate_hex_grid(farm, cell_acres=CELL_ACRES)

print(f"Total hexes: {len(cells)}")

# ---------- filter hexes: keep those whose centroid is inside farm (recommended) ----------
filtered = []
for h in cells:
    cx, cy = h['center']
    if Point(cx, cy).within(farm):
        filtered.append(h)
# if filtering produced too few, fallback to original
if len(filtered) < int(0.8 * len(cells)):
    filtered = cells

print(f"Hexes after centroid-inside filter: {len(filtered)}")

# add extra required fields (priority & restricted) for assignment
# simple random high-priority selection for demo
HIGH_PRIORITY_FRACTION = 0.05
num_high = max(1, int(len(filtered) * HIGH_PRIORITY_FRACTION))
candidates = [h for h in filtered]
random.shuffle(candidates)
high_set = set(x['id'] for x in candidates[:num_high])

for h in filtered:
    h['priority'] = 2 if h['id'] in high_set else 1
    h['restricted'] = False  # set some restricted = True if you want

# ---------- create UAVs with heterogeneous battery (%) ----------
uavs = []
for i in range(NUM_UAV):
    batt_pct = random.uniform(0.35, 1.00)
    uavs.append({'id': i, 'battery_pct': batt_pct})

# ---------- compute capacities (will also be computed in assign if missing) ----------
# Here we compute capacity_workload proportionally to battery and total workload
total_workload = sum(h['area_m2'] * h['priority'] for h in filtered)
battery_sum = sum(u['battery_pct'] for u in uavs)
for u in uavs:
    u['capacity_workload'] = total_workload * (u['battery_pct'] / battery_sum)

# ---------- perform assignment ----------
assignments, stats = assign_hexes_to_uavs(filtered, uavs, k=NUM_UAV, seed=SEED)

# ---------- export zone allocation CSV ----------
import csv
csv_path = os.path.join(OUTPUT_DIR, "zone_allocation.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["uav_id", "hex_id", "centroid_x", "centroid_y", "area_m2", "priority", "restricted"])
    for uav_id, hexes in assignments.items():
        for h in hexes:
            cx, cy = h['center']
            writer.writerow([uav_id, h['id'], cx, cy, h['area_m2'], h['priority'], h['restricted']])
print(f"Saved zone allocation CSV to {csv_path}")

# ---------- export per-UAV GeoJSON (hex polygons colored by UAV id) ----------
features = []
for uav_id, hexes in assignments.items():
    for h in hexes:
        feat = {
            "type": "Feature",
            "properties": {
                "uav_id": uav_id,
                "hex_id": int(h['id']),
                "area_m2": float(h['area_m2']),
                "priority": int(h['priority']),
                "restricted": bool(h['restricted'])
            },
            "geometry": mapping(h['poly'])
        }
        features.append(feat)

geo = {"type": "FeatureCollection", "features": features}
geo_path = os.path.join(OUTPUT_DIR, "uav_assignments.geojson")
with open(geo_path, "w") as f:
    json.dump(geo, f)
print(f"Saved GeoJSON to {geo_path}")

# ---------- print summary ----------
print("---- Assignment summary ----")
total_assigned = 0
for u in uavs:
    s = stats[u['id']]
    print(f"UAV {u['id']}: hexes={s['count_hexes']}, workload={s['workload']:.1f}, capacity={s['capacity']:.1f}, battery={s['battery_pct']:.2f}")
    total_assigned += s['count_hexes']
print(f"Total hexes assigned: {total_assigned}")
print(f"Avg hexes per UAV: {total_assigned / len(uavs):.2f}")
