"""Put a VGGT reconstruction upright for a simulator: Z up, floor at z = 0.

VGGT's world frame is the first camera's OpenCV frame (x right, y down,
z forward), so "up" is wherever the first camera's -y happened to point.
Isaac Sim uses Z-up with gravity along -Z, so we:

  1. estimate up as the mean of all cameras' up vectors (people hold phones
     roughly upright), then
  2. refine it with the lowest large plane whose normal is within
     `max_tilt_deg` of that estimate (the floor, if it is visible),
  3. rotate up -> +Z, yaw so the first camera looks along +X,
  4. translate the floor to z = 0 and the scene's XY center to the origin.

Everything is a rigid transform; scale is handled separately (VGGT units are arbitrary).
"""

from __future__ import annotations

import numpy as np

# OpenCV camera axes -> Z-up world (x forward, y left, z up) for a level camera.
CV_TO_ZUP = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R @ a = b (both 3-vectors, any length)."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: 180 deg about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = _unit(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def camera_up(extrinsics: np.ndarray) -> np.ndarray:
    """Mean world-frame up vector (camera -y) over all views. extrinsics: (S, 3, 4) camera-from-world."""
    R = extrinsics[:, :3, :3]
    return _unit((-R[:, 1, :]).mean(0))  # R^T @ (0, -1, 0) = -(row 1 of R)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return -np.einsum("sji,sj->si", R, t)  # -R^T t


def fit_plane_ransac(points: np.ndarray, threshold: float, iters: int = 500, seed: int = 0):
    """Largest plane n.x = d. Returns (n, d, inlier mask), n unit length."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(points), size=(iters, 3))
    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-12
    n, p0 = n[ok] / norm[ok, None], p0[ok]
    d = (n * p0).sum(1)

    best, best_count = 0, -1
    for s in range(0, len(n), 64):  # chunked to bound memory
        counts = (np.abs(points @ n[s:s + 64].T - d[s:s + 64]) < threshold).sum(0)
        j = int(counts.argmax())
        if counts[j] > best_count:
            best, best_count = s + j, int(counts[j])

    inl = np.abs(points @ n[best] - d[best]) < threshold
    # Least-squares refinement on the inliers.
    c = points[inl].mean(0)
    n_ref = np.linalg.svd(points[inl] - c, full_matrices=False)[2][-1]
    d_ref = float(n_ref @ c)
    return n_ref, d_ref, np.abs(points @ n_ref - d_ref) < threshold


def gravity_align(
    vertices: np.ndarray,
    extrinsics: np.ndarray,
    max_tilt_deg: float = 30.0,
    plane_threshold_frac: float = 0.01,
    n_planes: int = 4,
    min_plane_frac: float = 0.03,
    sample: int = 50_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """4x4 rigid transform from VGGT's world to a Z-up, floor-at-zero frame, plus diagnostics."""
    rng = np.random.default_rng(seed)
    pts = vertices if len(vertices) <= sample else vertices[rng.choice(len(vertices), sample, replace=False)]
    up0 = camera_up(extrinsics)
    extent = float(np.linalg.norm(np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0)))
    thr = plane_threshold_frac * extent

    # Peel off the largest planes; keep the roughly horizontal ones.
    candidates = []
    remaining = pts
    for k in range(n_planes):
        if len(remaining) < 100:
            break
        n, d, inl = fit_plane_ransac(remaining, thr, seed=seed + k)
        if inl.sum() < min_plane_frac * len(pts):
            break
        if n @ up0 < 0:
            n, d = -n, -d
        tilt = float(np.degrees(np.arccos(np.clip(n @ up0, -1, 1))))
        if tilt <= max_tilt_deg:
            candidates.append({"normal": n, "points": remaining[inl], "tilt": tilt,
                               "height": float((remaining[inl] @ up0).mean())})
        remaining = remaining[~inl]

    floor = min(candidates, key=lambda c: c["height"]) if candidates else None
    up = floor["normal"] if floor else up0

    R = rotation_between(up, [0.0, 0.0, 1.0])
    fwd = R @ extrinsics[0, :3, :3][2]  # first camera's optical axis in world
    if np.linalg.norm(fwd[:2]) > 1e-6:
        yaw = -np.arctan2(fwd[1], fwd[0])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R

    rv = vertices @ R.T
    z0 = float(np.median(floor["points"] @ R[2])) if floor else float(np.percentile(rv[:, 2], 2))
    lo, hi = rv.min(0), rv.max(0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -z0]

    info = {
        "floor_found": floor is not None,
        "floor_tilt_vs_camera_up_deg": None if floor is None else round(floor["tilt"], 2),
        "floor_point_frac": None if floor is None else round(len(floor["points"]) / len(pts), 3),
        "plane_threshold": thr,
    }
    return T, info


def opencv_to_zup_transform() -> np.ndarray:
    """No leveling: just re-axis VGGT's world (first camera, OpenCV) to Z-up."""
    T = np.eye(4)
    T[:3, :3] = CV_TO_ZUP
    return T


def world_from_camera(extrinsics: np.ndarray) -> np.ndarray:
    """(S, 3, 4) camera-from-world -> (S, 4, 4) world-from-camera (OpenCV camera axes)."""
    S = len(extrinsics)
    M = np.tile(np.eye(4), (S, 1, 1))
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    M[:, :3, :3] = np.swapaxes(R, 1, 2)
    M[:, :3, 3] = camera_centers(extrinsics)
    return M
