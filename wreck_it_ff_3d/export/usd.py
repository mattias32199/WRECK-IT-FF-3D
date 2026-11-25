"""Mesh (+ cameras) -> a self-contained USD file Isaac Sim can reference.

Layout (default prim /Reconstruction, so it can be referenced under any path):

    /Reconstruction                  Xform
        /Mesh                        Mesh, vertex colors, static triangle-mesh collider
        /Cameras/view_000 ...        one Camera per input frame (look through one
                                     in the viewport to check it lines up with the frame)

Stage is Z-up with metersPerUnit = 1. Only `usd-core` (pxr) is needed, not Isaac Sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from wreck_it_ff_3d.recon.fusion import MeshData

# OpenCV camera (x right, y down, looks +z) -> USD camera (x right, y up, looks -z).
_CV_TO_USD_CAM = np.diag([1.0, -1.0, -1.0, 1.0])
_H_APERTURE = 20.955  # USD default horizontal aperture; only the focal/aperture ratio matters


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _vec3f_array(a: np.ndarray):
    from pxr import Vt

    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32))


def _matrix(M: np.ndarray):
    from pxr import Gf

    return Gf.Matrix4d(M.T.tolist())  # USD stores row-vector matrices (translation in the last row)


def write_scene_usd(
    path: str | Path,
    mesh: MeshData,
    cameras: Optional[Sequence[tuple[np.ndarray, np.ndarray, tuple[int, int]]]] = None,
    collision: bool = True,
) -> Path:
    """cameras: (world_from_camera 4x4 in OpenCV axes, K 3x3, (H, W)) per view, already in the output frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Reconstruction")
    stage.SetDefaultPrim(root.GetPrim())

    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int32)
    m = UsdGeom.Mesh.Define(stage, "/Reconstruction/Mesh")
    m.CreatePointsAttr(_vec3f_array(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)  # fused surfaces are seen from both sides of thin parts
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))]))
    if mesh.normals is not None:
        m.CreateNormalsAttr(_vec3f_array(mesh.normals))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    color = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    color.Set(_vec3f_array(srgb_to_linear(np.asarray(mesh.colors))))

    if collision:
        # No RigidBodyAPI -> static collider; "none" = exact triangle mesh (valid for static bodies).
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)

    if cameras:
        UsdGeom.Scope.Define(stage, "/Reconstruction/Cameras")
        for i, (M, K, (H, W)) in enumerate(cameras):
            cam = UsdGeom.Camera.Define(stage, f"/Reconstruction/Cameras/view_{i:03d}")
            cam.AddTransformOp().Set(_matrix(M @ _CV_TO_USD_CAM))
            cam.CreateHorizontalApertureAttr(_H_APERTURE)
            cam.CreateVerticalApertureAttr(_H_APERTURE * H / W)
            cam.CreateFocalLengthAttr(float(K[0, 0]) * _H_APERTURE / W)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

    stage.GetRootLayer().Save()
    return path
