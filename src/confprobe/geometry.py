"""Camera and geometry utilities.

Conventions (identical to VGGT):
  * OpenCV camera frame: +x right, +y down, +z forward.
  * Extrinsic E = [R | t] (3x4) maps world -> camera:  X_cam = R @ X_world + t.
  * Intrinsic K is 3x3 in pixel units; pixel (u, v) refers to the pixel *center*
    at column u + 0.5, row v + 0.5 in continuous image coordinates.

All functions accept torch tensors; numpy wrappers live at the bottom.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Intrinsics / extrinsics
# --------------------------------------------------------------------------- #
def intrinsics_from_fov(width: int, height: int, fov_x_deg: float) -> torch.Tensor:
    """Pinhole intrinsics with square pixels and principal point at image center."""
    fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    return torch.tensor(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """World->camera extrinsic (3x4) for an OpenCV camera at `eye` looking at `target`.

    `up` is the world up direction (default +z). The camera's -y axis is aligned
    as closely as possible with `up`.
    """
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, 1.0]) if up is None else torch.as_tensor(up, dtype=torch.float32)

    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(z, up)
    if x.norm() < 1e-6:  # looking straight up/down: pick any perpendicular
        x = torch.linalg.cross(z, torch.tensor([1.0, 0.0, 0.0]))
    x = x / x.norm()
    y = torch.linalg.cross(z, x)  # points "down" in world when up=+z
    R = torch.stack([x, y, z], dim=0)  # rows = camera axes expressed in world
    t = -R @ eye
    return torch.cat([R, t[:, None]], dim=1)


def invert_extrinsic(E: torch.Tensor) -> torch.Tensor:
    """Invert (..., 3, 4) world->cam into cam->world (and vice versa)."""
    R, t = E[..., :3, :3], E[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    return torch.cat([Rt, -Rt @ t], dim=-1)


def to_homogeneous_4x4(E: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros(*E.shape[:-2], 1, 4, dtype=E.dtype, device=E.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat([E, bottom], dim=-2)


def compose_extrinsics(E_a: torch.Tensor, E_b: torch.Tensor) -> torch.Tensor:
    """Return E_a ∘ E_b (apply E_b first), both (..., 3, 4)."""
    return (to_homogeneous_4x4(E_a) @ to_homogeneous_4x4(E_b))[..., :3, :]


def camera_centers(E: torch.Tensor) -> torch.Tensor:
    """Camera centers in world coords for (..., 3, 4) extrinsics."""
    return invert_extrinsic(E)[..., :3, 3]


# --------------------------------------------------------------------------- #
# Rays, projection, unprojection
# --------------------------------------------------------------------------- #
def pixel_grid(height: int, width: int, device=None, offset: float = 0.5) -> torch.Tensor:
    """(H, W, 2) tensor of (u, v) pixel coordinates (centers by default)."""
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + offset,
        torch.arange(width, dtype=torch.float32, device=device) + offset,
        indexing="ij",
    )
    return torch.stack([u, v], dim=-1)


def camera_rays(K: torch.Tensor, E: torch.Tensor, height: int, width: int, uv: torch.Tensor | None = None):
    """World-space rays through pixels.

    Returns:
        origins: (..., 3) camera center broadcast to rays
        dirs:    (..., 3) unit ray directions in world
        z_scale: (...)    cos factor so that z-depth = t * z_scale for unit dirs
    """
    device = E.device
    if uv is None:
        uv = pixel_grid(height, width, device=device)
    ones = torch.ones_like(uv[..., :1])
    pix_h = torch.cat([uv, ones], dim=-1)  # (..., 3)
    d_cam = pix_h @ torch.linalg.inv(K).T.to(device)  # z = 1 in camera frame
    norm = d_cam.norm(dim=-1, keepdim=True)
    d_cam_unit = d_cam / norm
    R = E[:3, :3]
    d_world = d_cam_unit @ R  # R^T @ d  (row-vector form)
    origin = camera_centers(E).expand_as(d_world)
    z_scale = 1.0 / norm[..., 0]
    return origin, d_world, z_scale


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """(H, W) z-depth -> (H, W, 3) world points."""
    H, W = depth.shape
    uv = pixel_grid(H, W, device=depth.device)
    ones = torch.ones_like(uv[..., :1])
    d_cam = torch.cat([uv, ones], dim=-1) @ torch.linalg.inv(K).T.to(depth.device)
    X_cam = d_cam * depth[..., None]
    R, t = E[:3, :3], E[:3, 3]
    return (X_cam - t) @ R  # R^T (X - t)


def world_to_camera(X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return X @ E[:3, :3].T + E[:3, 3]


def project(X_world: torch.Tensor, K: torch.Tensor, E: torch.Tensor):
    """Project world points (..., 3).

    Returns (uv, z): uv is continuous pixel coords (..., 2) (pixel centers at +0.5),
    z is camera-frame depth (...).
    """
    X_cam = world_to_camera(X_world, E)
    z = X_cam[..., 2]
    xy = X_cam[..., :2] / z.clamp(min=1e-8)[..., None]
    uv = xy @ K[:2, :2].T + K[:2, 2]
    return uv, z


# --------------------------------------------------------------------------- #
# Rotations / pose errors
# --------------------------------------------------------------------------- #
def rotation_angle_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations (..., 3, 3) in degrees."""
    R = R_a.transpose(-1, -2) @ R_b
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def angle_between_deg(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Scalar-last quaternion (..., 4) -> rotation matrix (..., 3, 3). Matches vggt.utils.rotation.quat_to_mat."""
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    two = 2.0
    m = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> scalar-last quaternion (..., 4) with w >= 0."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = torch.sqrt((1 + tr).clamp(min=1e-12)) / 2
    qx = torch.sqrt((1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp(min=1e-12)) / 2
    qy = torch.sqrt((1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]).clamp(min=1e-12)) / 2
    qz = torch.sqrt((1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]).clamp(min=1e-12)) / 2
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q / q.norm(dim=-1, keepdim=True)


def pose_encoding_to_extri_intri(pose_enc: torch.Tensor, image_hw: tuple[int, int]):
    """Re-implementation of vggt.utils.pose_enc.pose_encoding_to_extri_intri
    for the default "absT_quaR_FoV" encoding: [T(3), quat_xyzw(4), fov_h, fov_w].

    Returns extrinsics (..., 3, 4) world->cam and intrinsics (..., 3, 3).
    """
    H, W = image_hw
    T = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    R = quat_xyzw_to_matrix(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    K = torch.zeros(*pose_enc.shape[:-1], 3, 3, dtype=pose_enc.dtype, device=pose_enc.device)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    K[..., 2, 2] = 1.0
    return extrinsics, K


def extri_intri_to_pose_encoding(E: torch.Tensor, K: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Inverse of :func:`pose_encoding_to_extri_intri` (useful for tests and fake models)."""
    H, W = image_hw
    quat = matrix_to_quat_xyzw(E[..., :3, :3])
    T = E[..., :3, 3]
    fov_h = 2 * torch.atan((H / 2.0) / K[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2.0) / K[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1)


# --------------------------------------------------------------------------- #
# numpy convenience
# --------------------------------------------------------------------------- #
def np_unproject_depth(depth: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    return unproject_depth(torch.from_numpy(depth).float(), torch.from_numpy(K).float(), torch.from_numpy(E).float()).numpy()
