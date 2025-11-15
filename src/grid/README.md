from shapely.geometry import Polygon
from grid.hexgrid import generate_hex_grid, compute_hex_side_for_acres, hex_area

def main():
    # synthetic farm rectangle: W x H (meters) approximating 200 acres
    FARM_W = 1000.0
    ACRE_M2 = 4046.8564224
    TOTAL_AREA_M2 = 200 * ACRE_M2
    FARM_H = TOTAL_AREA_M2 / FARM_W

    farm = Polygon([(0,0), (FARM_W,0), (FARM_W,FARM_H), (0,FARM_H)])
    cells = generate_hex_grid(farm, cell_acres=0.5)

    print(f"Total hexes: {len(cells)}")

    # find one hex that is not edge-clipped (i.e., its polygon equals a full hex area within tolerance)
    side = compute_hex_side_for_acres(0.5)
    full_area = hex_area(side)
    # print first few areas and check one close to full_area
    for h in cells[:30]:
        print("id:", h['id'], "area:", round(h['area_m2'], 3))
    # find hex whose area is close to full_area (within 1%)
    tol = 0.01 * full_area
    full_hex = next((h for h in cells if abs(h['area_m2'] - full_area) < tol), None)
    if full_hex:
        print("Found full hex! center:", full_hex['center'], "area:", full_hex['area_m2'])
    else:
        print("No full hex found in sample — but interior hexes should be full. full_area:", full_area)

if __name__ == '__main__':
    main()
