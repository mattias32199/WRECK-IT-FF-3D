"""Prediction -> one watertight-ish triangle mesh, by TSDF fusion of VGGT's depth maps.

Why depth + cameras and not the point-map head: the VGGT paper reports that
unprojecting predicted depth with predicted cameras is usually more accurate,
and TSDF fusion averages the per-view depth noise into one consistent surface,
which is what a physics collider needs.

Units are VGGT's (arbitrary scale), so the voxel size is set relative to the
median depth instead of in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from wreck_it_ff_3d.models.prediction import Prediction
from wreck_it_ff_3d.vis_utils import conf_mask


@dataclass
class MeshData:
    vertices: np.ndarray  # (N, 3) float
    faces: np.ndarray     # (M, 3) int
    colors: np.ndarray    # (N, 3) float in [0, 1], sRGB
    normals: Optional[np.ndarray] = None  # (N, 3)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def transformed(self, T: np.ndarray, scale: float = 1.0) -> "MeshData":
        """Apply a rigid 4x4 transform, then a uniform scale."""
        R, t = T[:3, :3], T[:3, 3]
        n = None if self.normals is None else self.normals @ R.T
        return MeshData((self.vertices @ R.T + t) * scale, self.faces, self.colors, n)

    def save_ply(self, path: str | Path) -> Path:
        import open3d as o3d

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(path), to_open3d(self))
        return path


def to_open3d(mesh: MeshData):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)),
    )
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.colors, np.float64))
    if mesh.normals is not None:
        m.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.normals, np.float64))
    return m


def from_open3d(m) -> MeshData:
    normals = np.asarray(m.vertex_normals) if m.has_vertex_normals() else None
    return MeshData(np.asarray(m.vertices).copy(), np.asarray(m.triangles).copy(),
                    np.asarray(m.vertex_colors).copy(), None if normals is None else normals.copy())


def edge_mask(depth: np.ndarray, max_rel_jump: float = 0.05) -> np.ndarray:
    """False at depth discontinuities ("flying pixels" smeared between foreground and background).

    A pixel is dropped if any 4-neighbour differs from it by more than max_rel_jump * depth.
    These pixels are what turn into the comb-like streaks hanging off walls after fusion.
    """
    d = depth
    jump = np.zeros_like(d)
    for axis in (-1, -2):
        diff = np.abs(np.diff(d, axis=axis))
        lo = [slice(None)] * d.ndim
        hi = [slice(None)] * d.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        jump[tuple(lo)] = np.maximum(jump[tuple(lo)], diff)
        jump[tuple(hi)] = np.maximum(jump[tuple(hi)], diff)
    return jump <= max_rel_jump * np.maximum(d, 1e-8)


def consistency_mask(
    depth: np.ndarray, K: np.ndarray, E: np.ndarray,
    neighbors: int = 3, rel_tol: float = 0.03, min_agree: int = 2,
) -> np.ndarray:
    """Keep a pixel only if at least `min_agree` nearby views (within `neighbors` in sequence
    order) see the same 3D point: projecting it into view j lands on a depth within rel_tol.

    Removes depth that the other views contradict: noise, and "ghost" copies of surfaces
    caused by slightly wrong camera poses (doubled walls after fusion).
    depth (S, H, W) with 0 = invalid, K (S, 3, 3), E (S, 3, 4) camera-from-world.
    """
    from wreck_it_ff_3d.geometry_utils import unproject_depth

    S, H, W = depth.shape
    world = unproject_depth(depth, K, E)
    agree = np.zeros((S, H, W), np.int16)
    for i in range(S):
        vi = depth[i] > 0
        X = world[i][vi].astype(np.float64)
        if len(X) == 0:
            continue
        count = np.zeros(len(X), np.int16)
        for j in range(max(0, i - neighbors), min(S, i + neighbors + 1)):
            if j == i:
                continue
            Xc = X @ E[j, :, :3].T.astype(np.float64) + E[j, :, 3]
            z = Xc[:, 2]
            front = z > 1e-6
            uv = (Xc[front] / z[front, None]) @ K[j].T.astype(np.float64)
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx = np.flatnonzero(front)[inside]
            dj = depth[j][v[inside], u[inside]]
            ok = (dj > 0) & (np.abs(z[idx] - dj) <= rel_tol * dj)
            count[idx[ok]] += 1
        agree[i][vi] = count
    return agree >= min_agree


def masked_depth(
    pred: Prediction,
    keep_conf: float = 0.6,
    max_depth_quantile: float = 0.98,
    max_rel_jump: Optional[float] = 0.05,
    consistency: Optional[dict] = None,
) -> np.ndarray:
    """Depth with unreliable pixels set to 0 (= ignored by TSDF): low confidence, far tail,
    depth edges (if max_rel_jump), and multi-view disagreement (if consistency, kwargs for consistency_mask)."""
    d = pred.depth.astype(np.float32)
    m = conf_mask(pred.depth_conf, keep_conf) & np.isfinite(d) & (d > 0)
    if max_rel_jump is not None:
        m &= edge_mask(np.where(np.isfinite(d), d, 0), max_rel_jump)
    if m.any():
        m &= d <= np.quantile(d[m], max_depth_quantile)
    d = np.where(m, d, 0.0).astype(np.float32)
    if consistency is not None:
        d = np.where(consistency_mask(d, pred.intrinsics, pred.extrinsics, **consistency), d, 0.0)
    return d.astype(np.float32)


def fuse_tsdf(
    pred: Prediction,
    keep_conf: float = 0.8,
    voxel_frac: float = 1 / 80,
    trunc_voxels: float = 4.0,
    max_depth_quantile: float = 0.98,
    max_rel_jump: Optional[float] = 0.05,
    consistency: Optional[dict] = None,
) -> tuple[MeshData, dict]:
    """Integrate every view's (masked) depth + color into a TSDF and extract a mesh.

    keep_conf:  fraction of pixels kept, by depth confidence, across all views.
    voxel_frac: voxel size as a fraction of the median kept depth.
    max_rel_jump, consistency: extra pixel filters, see masked_depth.
    """
    import open3d as o3d

    d = masked_depth(pred, keep_conf, max_depth_quantile, max_rel_jump, consistency)
    kept = d[d > 0]
    if kept.size == 0:
        raise ValueError("No depth pixels left after confidence masking; raise keep_conf")
    voxel = float(np.median(kept) * voxel_frac)
    depth_trunc = float(kept.max() * 1.001)

    S, H, W = d.shape
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc_voxels * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(S):
        K = pred.intrinsics[i].astype(np.float64)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(pred.images[i], dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(d[i])),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        E = np.eye(4)
        E[:3, :4] = pred.extrinsics[i]  # camera-from-world, which is what Open3D expects
        vol.integrate(rgbd, intr, E)

    mesh = vol.extract_triangle_mesh()
    if len(mesh.triangles) == 0 and o3d.__version__.startswith("0.20"):
        # Checked 2026-11: ScalableTSDFVolume returns nothing in open3d 0.20.0, even for
        # one frame with an identity pose; 0.19.0 works (see tests/test_video_to_usd.py).
        raise RuntimeError("open3d 0.20 TSDF regression produced an empty mesh: pip install 'open3d<0.20'")
    info = {"voxel": voxel, "sdf_trunc": trunc_voxels * voxel, "keep_conf": keep_conf,
            "max_rel_jump": max_rel_jump, "consistency": consistency,
            "pixel_frac_used": round(float(kept.size / d.size), 4),
            "pixels_used": int(kept.size), "raw_faces": len(mesh.triangles)}
    return from_open3d(mesh), info


def clean_mesh(mesh: MeshData, min_component_frac: float = 0.005,
               max_faces: Optional[int] = 300_000) -> MeshData:
    """Drop small floating pieces, decimate for physics, recompute normals."""
    m = to_open3d(mesh)
    if len(m.triangles) == 0:
        return mesh
    clusters, sizes, _ = m.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    m.remove_triangles_by_mask(sizes[clusters] < min_component_frac * len(clusters))
    m.remove_unreferenced_vertices()
    if max_faces and len(m.triangles) > max_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return from_open3d(m)
