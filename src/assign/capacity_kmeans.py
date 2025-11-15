# src/assign/capacity_kmeans.py
import math
import numpy as np
from sklearn.cluster import KMeans

def assign_hexes_to_uavs(hex_cells, uavs, k=None, seed=42):
    """
    Capacity-constrained clustering assignment.

    Args:
        hex_cells: list of dicts, each must contain:
            - 'id', 'center' (x,y), 'area_m2', 'priority' (1 or 2), 'restricted' (bool)
        uavs: list of dicts, each must contain:
            - 'id', 'battery_pct' (0..1), optionally 'capacity_workload' (will be computed if absent)
        k: number of clusters (if None, will use len(uavs))
        seed: random seed for reproducibility

    Returns:
        assignments: dict mapping uav_id -> list of hex dicts (assigned)
        summary: dict with stats (workloads, counts)
    """
    if k is None:
        k = len(uavs)
    rng = np.random.RandomState(seed)

    # compute workload for each hex = area * priority
    workloads = np.array([h['area_m2'] * h.get('priority', 1) for h in hex_cells])
    centers = np.array([h['center'] for h in hex_cells])

    total_workload = float(workloads.sum())
    if total_workload <= 0:
        raise ValueError("Total workload is zero or negative - check hex cell areas/priorities.")

    # compute capacities if not provided (proportional to battery pct)
    battery_sum = sum(u.get('battery_pct', 1.0) for u in uavs)
    for u in uavs:
        if 'capacity_workload' not in u:
            # proportion of total_workload proportional to battery_pct
            u['capacity_workload'] = total_workload * (u.get('battery_pct', 1.0) / battery_sum)

    capacities = [u['capacity_workload'] for u in uavs]

    # 1) seed clusters with KMeans on centers
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto').fit(centers)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # build buckets
    buckets = {i: [] for i in range(k)}
    for idx, lab in enumerate(labels):
        buckets[lab].append(hex_cells[idx])

    def bucket_workload(bucket):
        return sum(h['area_m2'] * h.get('priority', 1) for h in bucket)

    # 2) greedy rebalancing to respect capacities
    changed = True
    iter_count = 0
    max_iters = max(200, 10 * k)
    while changed and iter_count < max_iters:
        changed = False
        iter_count += 1
        for b in range(k):
            w = bucket_workload(buckets[b])
            cap = capacities[b]
            if w <= cap:
                continue
            # bucket too heavy, try to move farthest hex out
            centroid = cluster_centers[b]
            # sort hexes by distance descending from centroid (prefer moving outskirts)
            hexes_sorted = sorted(buckets[b], key=lambda h: -math.hypot(h['center'][0] - centroid[0],
                                                                          h['center'][1] - centroid[1]))
            moved = False
            for h in hexes_sorted:
                # Try to find another bucket with spare capacity
                candidates = []
                for b2 in range(k):
                    if b2 == b:
                        continue
                    spare = capacities[b2] - bucket_workload(buckets[b2])
                    # require the other bucket to accept at least this hex (some slack)
                    if spare >= h['area_m2'] * h.get('priority', 1) * 0.5:
                        # distance to that bucket center
                        dist = math.hypot(h['center'][0] - cluster_centers[b2][0], h['center'][1] - cluster_centers[b2][1])
                        candidates.append((dist, b2))
                if not candidates:
                    continue
                # pick nearest candidate bucket and move hex
                candidates.sort()
                chosen_b = candidates[0][1]
                buckets[b].remove(h)
                buckets[chosen_b].append(h)
                changed = True
                moved = True
                break
            if moved:
                # continue outer loop after first move to recompute workloads
                continue

    # map buckets -> uav ids (1-to-1 in index order)
    assignments = {}
    stats = {}
    for i, u in enumerate(uavs):
        assigned = buckets.get(i, [])
        assignments[u['id']] = assigned
        stats[u['id']] = {
            'count_hexes': len(assigned),
            'workload': bucket_workload(assigned),
            'capacity': u['capacity_workload'],
            'battery_pct': u.get('battery_pct', None)
        }

    return assignments, stats
