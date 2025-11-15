import math
import numpy as np
from shapely.geometry import Polygon, Point

ACRE_M2 = 4046.8564224
HEX_CONST = 3.0 * math.sqrt(3) / 2.0   # for regular hexagon area formula


def hex_area(side_length_m: float) -> float:
    """
    Calculate area of a regular hexagon in square meters.
    Formula: A = (3 * sqrt(3) / 2) * s^2
    """
    return HEX_CONST * (side_length_m ** 2)


def compute_hex_side_for_acres(cell_acres: float) -> float:
    """
    Given area in acres, compute required hexagon side length (meters).
    """
    target_area_m2 = cell_acres * ACRE_M2
    side = math.sqrt(target_area_m2 / HEX_CONST)
    return side


def axial_to_xy(q, r, s):
    """
    Convert axial hex coordinates to cartesian (meters) for pointy-top hexes.
    """
    x = s * math.sqrt(3) * (q + r / 2.0)
    y = s * 1.5 * r
    return x, y


def hex_center_to_polygon(cx, cy, s):
    """
    Build a Shapely polygon for a regular hexagon centered at (cx, cy).
    """
    vertices = []
    for k in range(6):
        angle_deg = 30 + 60 * k
        angle_rad = math.radians(angle_deg)
        vx = cx + s * math.cos(angle_rad)
        vy = cy + s * math.sin(angle_rad)
        vertices.append((vx, vy))
    return Polygon(vertices)


def generate_hex_grid(farm_polygon, cell_acres=0.5, ignore_slivers=True):
    """
    Generate hexagonal tiling clipped to the farm boundary.

    This version GENERATES a larger hex cloud, recenters it to the farm centroid,
    then clips hexes to the farm polygon so that interior hexes are full (not
    half-cut) and edge hexes are the only clipped ones.

    Returns:
        List of dicts:
            {
                'id': int,
                'center': (x, y),        # center in meters (after translation)
                'poly': shapely_polygon, # clipped polygon
                'area_m2': float
            }
    """
    # 1. compute hex side length for given acre size
    side_length = compute_hex_side_for_acres(cell_acres)
    s = side_length
    sqrt3 = math.sqrt(3.0)

    # 2. get farm bounding box
    minx, miny, maxx, maxy = farm_polygon.bounds
    farm_cx = (minx + maxx) / 2.0
    farm_cy = (miny + maxy) / 2.0

    # 3. create an initial generous axial grid around origin (centered near 0,0)
    # pick ranges large enough to cover the farm after translation
    q_min = int(math.floor((minx - 3 * s) / (s * sqrt3))) - 3
    q_max = int(math.ceil((maxx + 3 * s) / (s * sqrt3))) + 3
    r_min = int(math.floor((miny - 3 * s) / (1.5 * s))) - 3
    r_max = int(math.ceil((maxy + 3 * s) / (1.5 * s))) + 3

    raw_hexes = []
    for r in range(r_min, r_max + 1):
        for q in range(q_min, q_max + 1):
            cx, cy = axial_to_xy(q, r, s)
            poly = hex_center_to_polygon(cx, cy, s)
            raw_hexes.append({'q': q, 'r': r, 'center': (cx, cy), 'poly': poly})

    if not raw_hexes:
        return []

    # 4. compute mean center of raw hex cloud
    all_x = [h['center'][0] for h in raw_hexes]
    all_y = [h['center'][1] for h in raw_hexes]
    mean_x, mean_y = np.mean(all_x), np.mean(all_y)

    # 5. compute offset to align cloud center to farm center
    off_x = farm_cx - mean_x
    off_y = farm_cy - mean_y

    # 6. translate hex centers & polys so grid is centered on farm centroid
    translated_hexes = []
    for h in raw_hexes:
        cx, cy = h['center']
        cx_t = cx + off_x
        cy_t = cy + off_y
        poly_t = hex_center_to_polygon(cx_t, cy_t, s)
        translated_hexes.append({'center': (cx_t, cy_t), 'poly': poly_t})

    # 7. clip to farm polygon and collect valid hex cells
    hex_cells = []
    hex_id = 0
    for h in translated_hexes:
        hex_poly = h['poly']
        if hex_poly.intersects(farm_polygon):
            clipped = hex_poly.intersection(farm_polygon)
            area = clipped.area
            if ignore_slivers:
                if area > 1.0:  # ignore tiny slivers
                    hex_cells.append({
                        'id': hex_id,
                        'center': h['center'],
                        'poly': clipped,
                        'area_m2': area
                    })
                    hex_id += 1
            else:
                hex_cells.append({
                    'id': hex_id,
                    'center': h['center'],
                    'poly': clipped,
                    'area_m2': area
                })
                hex_id += 1

    return hex_cells
