from typing import List

import numpy as np


def _find_distance_to(points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    n_ref = len(ref_points)
    if n_ref == 0:
        return np.zeros(0)
    if n_ref == 1:
        return np.linalg.norm(points - ref_points, axis=1)
    ref_points = np.array(sorted(ref_points, key=lambda p: p[0]))
    px = points.T[0]
    rx = ref_points.T[0]
    local_unit_vecs = ref_points[1:] - ref_points[:-1]
    dists = []
    bins = np.digitize(px, rx) - 1
    for point, left_ref_p in zip(points, bins):
        if left_ref_p == -1:
            left_ref_p = 0
        to_left_ref = ref_points[left_ref_p] - point
        local_unit_vec = (
            local_unit_vecs[-1]
            if left_ref_p >= n_ref - 1
            else local_unit_vecs[left_ref_p]
        )
        projection = np.dot(local_unit_vec, to_left_ref) / np.linalg.norm(
            local_unit_vec
        )
        dist = np.sqrt(np.linalg.norm(to_left_ref) ** 2 - projection ** 2)
        dists.append(dist)
    return np.array(dists)


def is_pareto_efficient(points: np.ndarray, take_n: int = None) -> List[int]:
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for idx, c in enumerate(points):
        if is_pareto[idx]:
            # Keep any point with a higher value
            is_pareto[is_pareto] = np.any(points[is_pareto] > c, axis=1)
            is_pareto[idx] = True  # And keep self
    non_pareto = np.logical_not(is_pareto)
    pareto_idx = is_pareto.nonzero()[0]
    non_pareto_idx = non_pareto.nonzero()[0]

    non_pareto_dist_to_pareto = _find_distance_to(points[non_pareto], points[is_pareto])
    dist_order = np.argsort(non_pareto_dist_to_pareto)
    take_n_non_pareto = 0 if take_n is None else take_n - len(pareto_idx)
    dist_order = dist_order[:take_n_non_pareto]
    taken_non_pareto_idx = non_pareto_idx[dist_order]
    return pareto_idx.tolist() + taken_non_pareto_idx.tolist()
