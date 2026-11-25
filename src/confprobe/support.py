"""Cross-view visibility and per-pixel *geometric support*.

For a pixel p in view i with ground-truth surface point X (from GT depth):

  visible_j(p)  = X projects inside view j, in front of the camera, and the GT
                  depth of view j at that location agrees with X's depth in j
                  (i.e. X is not self-occluded in view j).
  support(p)    = #{ j : visible_j(p) and view j is not masked at X's projection }
  n_visible(p)  = #{ j : visible_j(p) }  (support if nothing were masked)

`support` counts the pixel's own view when it is not masked there, so an
unmasked pixel seen by no other view has support 1 (monocular), and a pixel
hidden in every view has support 0 (pure guess).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import project, unproject_depth


@dataclass
class SupportMaps:
    support: np.ndarray  # (S, H, W) uint8
    n_visible: np.ndarray  # (S, H, W) uint8
    support_bits: np.ndarray  # (S, H, W) int64 bitmask of views providing support
    self_visible: np.ndarray  # (S, H, W) bool (pixel not masked in its own view)


def _depth_consistent(Dj: torch.Tensor, uv: torch.Tensor, z: torch.Tensor, rel_tol: float, abs_tol: float) -> torch.Tensor:
    """Is a point at depth z visible at continuous location uv of a depth map?

    Accept if either the bilinearly interpolated depth matches (smooth, slanted
    surfaces) or any of the 4 surrounding pixel-center depths matches (near depth
    edges, where interpolation mixes foreground and background).
    """
    H, W = Dj.shape
    x = uv[:, 0] - 0.5  # continuous coords in pixel-center units
    y = uv[:, 1] - 0.5
    x0 = torch.floor(x).long().clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    fx = (x - x0.float()).clamp(0, 1)
    fy = (y - y0.float()).clamp(0, 1)
    d00, d01, d10, d11 = Dj[y0, x0], Dj[y0, x1], Dj[y1, x0], Dj[y1, x1]
    tol = rel_tol * z + abs_tol
    bil = (1 - fy) * ((1 - fx) * d00 + fx * d01) + fy * ((1 - fx) * d10 + fx * d11)
    ok = (bil - z).abs() <= tol
    for d in (d00, d01, d10, d11):
        ok |= (d - z).abs() <= tol
    return ok


@torch.no_grad()
def compute_support(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: np.ndarray | None = None,
    rel_tol: float = 0.01,
    abs_tol: float = 0.005,
    device: str = "cpu",
    normals: np.ndarray | None = None,
    min_cos: float = 0.1,
) -> SupportMaps:
    """Compute support maps.

    Args:
        depth: (S, H, W) GT z-depth.
        intrinsics/extrinsics: (S, 3, 3) / (S, 3, 4), world->cam.
        masks: (S, H, W) bool, True where the input image is hidden from the model.
        rel_tol/abs_tol: depth-consistency tolerance for the visibility test.
        normals: optional (S, H, W, 3) world normals. If given, a point only counts as
            visible in view j when it faces camera j at more than a grazing angle
            (cos > min_cos); this removes aliasing at near-silhouette points.
    """
    S, H, W = depth.shape
    if S > 62:
        raise ValueError("support_bits supports at most 62 views")
    D = torch.as_tensor(depth, dtype=torch.float32, device=device)
    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    E = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    M = (
        torch.zeros((S, H, W), dtype=torch.bool, device=device)
        if masks is None
        else torch.as_tensor(masks, dtype=torch.bool, device=device)
    )

    support = torch.zeros((S, H, W), dtype=torch.int32, device=device)
    n_vis = torch.zeros((S, H, W), dtype=torch.int32, device=device)
    bits = torch.zeros((S, H, W), dtype=torch.int64, device=device)

    N = None if normals is None else torch.as_tensor(normals, dtype=torch.float32, device=device)
    centers = -torch.einsum("sji,sj->si", E[:, :, :3], E[:, :, 3])

    for i in range(S):
        X = unproject_depth(D[i], K[i], E[i]).reshape(-1, 3)
        Ni = None if N is None else N[i].reshape(-1, 3)
        for j in range(S):
            if j == i:
                vis = torch.ones(H * W, dtype=torch.bool, device=device)
                hidden = M[i].reshape(-1)
            else:
                uv, z = project(X, K[j], E[j])
                col = torch.floor(uv[:, 0]).long()
                row = torch.floor(uv[:, 1]).long()
                inb = (z > 1e-6) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
                col_c, row_c = col.clamp(0, W - 1), row.clamp(0, H - 1)
                vis = inb & _depth_consistent(D[j], uv, z, rel_tol, abs_tol)
                if Ni is not None:
                    to_cam = centers[j] - X
                    cos = (Ni * to_cam).sum(-1) / to_cam.norm(dim=-1).clamp(min=1e-8)
                    vis &= cos > min_cos
                hidden = M[j][row_c, col_c]
            sup = vis & ~hidden
            n_vis[i] += vis.reshape(H, W).int()
            support[i] += sup.reshape(H, W).int()
            bits[i] |= sup.reshape(H, W).long() << j

    return SupportMaps(
        support=support.cpu().numpy().astype(np.uint8),
        n_visible=n_vis.cpu().numpy().astype(np.uint8),
        support_bits=bits.cpu().numpy(),
        self_visible=(~M).cpu().numpy(),
    )
