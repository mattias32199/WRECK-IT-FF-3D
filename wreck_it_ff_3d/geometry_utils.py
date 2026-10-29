"""Shared camera geometry (numpy, batched over views).

Starts with what the run layer needs. The score layer will add `project`,
`visibility` (depth test), and support counting here, because Exp 1 support,
Exp 2 head scoring, and evaluation all use the same unproject -> transform ->
reproject chain. Keep every function here unit-tested.

Conventions: see `scene.py` (OpenCV, camera-from-world [R | t], pixel centers
at integer coordinates).
"""

from __future__ import annotations

import numpy as np


def invert_extrinsics(E: np.ndarray) -> np.ndarray:
    """(..., 3, 4) camera-from-world -> (..., 3, 4) world-from-camera."""
    R, t = E[..., :3, :3], E[..., :3, 3:]
    Rt = np.swapaxes(R, -1, -2)
    return np.concatenate([Rt, -Rt @ t], axis=-1)


def pixel_grid(h: int, w: int) -> np.ndarray:
    """(H, W, 3) homogeneous pixel coordinates [u, v, 1]."""
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)


def unproject_depth(depth: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Depth maps -> world points.

    depth (S, H, W), K (S, 3, 3), E (S, 3, 4) camera-from-world  ->  (S, H, W, 3)
    """
    S, H, W = depth.shape
    rays = pixel_grid(H, W) @ np.linalg.inv(K).transpose(0, 2, 1)[:, None]  # (S, H, W, 3), z = 1
    cam = rays * depth[..., None]
    Ewc = invert_extrinsics(E)
    world = cam @ Ewc[:, None, :, :3].transpose(0, 1, 3, 2) + Ewc[:, None, None, :, 3]
    return world.astype(np.float32)
