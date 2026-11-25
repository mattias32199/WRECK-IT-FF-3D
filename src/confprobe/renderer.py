"""Dependency-free torch ray caster for :mod:`confprobe.scene` scenes.

Produces RGB (anti-aliased by supersampling), exact z-depth, surface normals,
primitive ids and face ids. Runs on CPU or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import camera_rays, pixel_grid, unproject_depth
from .scene import KIND_BOX, KIND_ROOM, KIND_SPHERE, Cameras, Primitive, Scene

_EPS = 1e-4
_INF = float("inf")


@dataclass
class RenderOutput:
    rgb: np.ndarray  # (S, H, W, 3) uint8
    depth: np.ndarray  # (S, H, W) float32 z-depth (camera frame)
    normals: np.ndarray  # (S, H, W, 3) float32 world-space unit normals
    prim_id: np.ndarray  # (S, H, W) int16 index into scene.primitives
    face_id: np.ndarray  # (S, H, W) int8

    def world_points(self, cams: Cameras) -> np.ndarray:
        out = []
        for i in range(len(cams)):
            out.append(
                unproject_depth(
                    torch.from_numpy(self.depth[i]),
                    torch.from_numpy(cams.intrinsics[i]),
                    torch.from_numpy(cams.extrinsics[i]),
                ).numpy()
            )
        return np.stack(out)


# --------------------------------------------------------------------------- #
# Ray-primitive intersection. All return (t, normal_world, face) with t=inf on miss.
# --------------------------------------------------------------------------- #
def _to_local(prim: Primitive, o: torch.Tensor, d: torch.Tensor):
    R = torch.as_tensor(prim.rotation, dtype=o.dtype, device=o.device)
    c = torch.as_tensor(prim.center, dtype=o.dtype, device=o.device)
    return (o - c) @ R, d @ R, R  # R^T(o - c), R^T d


def _safe_inv(d: torch.Tensor) -> torch.Tensor:
    d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    return 1.0 / d


def intersect_box(prim: Primitive, o: torch.Tensor, d: torch.Tensor):
    ol, dl, R = _to_local(prim, o, d)
    h = torch.as_tensor(prim.half_extents, dtype=o.dtype, device=o.device)
    inv = _safe_inv(dl)
    t1 = (-h - ol) * inv
    t2 = (h - ol) * inv
    tmin = torch.minimum(t1, t2)
    tmax = torch.maximum(t1, t2)
    t_near, ax_near = tmin.max(dim=-1)
    t_far, ax_far = tmax.min(dim=-1)
    hit = (t_near <= t_far) & (t_far > _EPS)
    outside = t_near > _EPS
    t = torch.where(outside, t_near, t_far)
    axis = torch.where(outside, ax_near, ax_far)
    t = torch.where(hit, t, torch.full_like(t, _INF))
    d_ax = dl.gather(-1, axis[:, None])[:, 0]
    # outward normal on the entry face points against the ray direction
    sign = torch.where(outside, -torch.sign(d_ax), torch.sign(d_ax))
    n_local = torch.zeros_like(ol)
    n_local.scatter_(-1, axis[:, None], sign[:, None])
    n_world = n_local @ R.T
    face = axis * 2 + (sign > 0).long()
    return t, n_world, face


def intersect_room(prim: Primitive, o: torch.Tensor, d: torch.Tensor):
    """Interior of an axis-aligned box: the ray exits through a wall; normal points inward."""
    ol, dl, R = _to_local(prim, o, d)
    h = torch.as_tensor(prim.half_extents, dtype=o.dtype, device=o.device)
    inv = _safe_inv(dl)
    t1 = (-h - ol) * inv
    t2 = (h - ol) * inv
    tmax = torch.maximum(t1, t2)
    t_far, axis = tmax.min(dim=-1)
    d_ax = dl.gather(-1, axis[:, None])[:, 0]
    sign = -torch.sign(d_ax)  # inward
    n_local = torch.zeros_like(ol)
    n_local.scatter_(-1, axis[:, None], sign[:, None])
    t = torch.where(t_far > _EPS, t_far, torch.full_like(t_far, _INF))
    face = axis * 2 + (d_ax > 0).long()
    return t, n_local @ R.T, face


def intersect_sphere(prim: Primitive, o: torch.Tensor, d: torch.Tensor):
    c = torch.as_tensor(prim.center, dtype=o.dtype, device=o.device)
    r = prim.radius
    oc = o - c
    b = (oc * d).sum(-1)
    cc = (oc * oc).sum(-1) - r * r
    disc = b * b - cc
    sq = torch.sqrt(disc.clamp(min=0))
    t0 = -b - sq
    t1 = -b + sq
    t = torch.where(t0 > _EPS, t0, t1)
    hit = (disc >= 0) & (t > _EPS)
    t = torch.where(hit, t, torch.full_like(t, _INF))
    p = o + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
    n = (p - c) / r
    return t, n, torch.zeros_like(t, dtype=torch.long)


_INTERSECT = {KIND_ROOM: intersect_room, KIND_BOX: intersect_box, KIND_SPHERE: intersect_sphere}


def cast(scene: Scene, o: torch.Tensor, d: torch.Tensor):
    """Nearest hit over all primitives. Returns t, normal, prim index, face."""
    N = o.shape[0]
    best_t = torch.full((N,), _INF, dtype=o.dtype, device=o.device)
    best_n = torch.zeros((N, 3), dtype=o.dtype, device=o.device)
    best_p = torch.full((N,), -1, dtype=torch.long, device=o.device)
    best_f = torch.zeros((N,), dtype=torch.long, device=o.device)
    for i, prim in enumerate(scene.primitives):
        t, n, f = _INTERSECT[prim.kind](prim, o, d)
        closer = t < best_t
        best_t = torch.where(closer, t, best_t)
        best_n = torch.where(closer[:, None], n, best_n)
        best_p = torch.where(closer, torch.full_like(best_p, i), best_p)
        best_f = torch.where(closer, f, best_f)
    return best_t, best_n, best_p, best_f


# --------------------------------------------------------------------------- #
# Shading
# --------------------------------------------------------------------------- #
def albedo(scene: Scene, p: torch.Tensor, prim_idx: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(p)
    for i, prim in enumerate(scene.primitives):
        sel = prim_idx == i
        if not sel.any():
            continue
        tex = prim.texture
        q = p[sel]
        dev, dt = p.device, p.dtype
        freqs = torch.as_tensor(tex.freqs, device=dev, dtype=dt)
        phases = torch.as_tensor(tex.phases, device=dev, dtype=dt)
        amps = torch.as_tensor(tex.amps, device=dev, dtype=dt)
        m = (torch.sin(q @ freqs.T + phases) * amps).sum(-1) / amps.sum() * 3.0
        w = torch.sigmoid(tex.sharpness * m)[:, None]
        f = face[sel].clamp(max=tex.colors_a.shape[0] - 1)
        ca = torch.as_tensor(tex.colors_a, device=dev, dtype=dt)[f]
        cb = torch.as_tensor(tex.colors_b, device=dev, dtype=dt)[f]
        col = ca * (1 - w) + cb * w
        ff = torch.as_tensor(tex.fine_freqs, device=dev, dtype=dt)  # (3, Kf, 3)
        fp = torch.as_tensor(tex.fine_phases, device=dev, dtype=dt)  # (3, Kf)
        fine = torch.sin(torch.einsum("nd,ckd->nck", q, ff) + fp).mean(-1)  # (n, 3)
        out[sel] = col + tex.fine_amp * fine
    return out.clamp(0, 1)


def shade(scene: Scene, p, n, prim_idx, face) -> torch.Tensor:
    a = albedo(scene, p, prim_idx, face)
    L = torch.as_tensor(scene.light_dirs, device=p.device, dtype=p.dtype)
    s = torch.as_tensor(scene.light_strengths, device=p.device, dtype=p.dtype)
    lambert = ((n @ L.T).clamp(min=0) * s).sum(-1, keepdim=True)
    return (a * (scene.ambient + lambert)).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _render_rays(scene: Scene, o, d, chunk: int):
    ts, ns, ps, fs = [], [], [], []
    for s in range(0, o.shape[0], chunk):
        t, n, p, f = cast(scene, o[s : s + chunk], d[s : s + chunk])
        ts.append(t), ns.append(n), ps.append(p), fs.append(f)
    return torch.cat(ts), torch.cat(ns), torch.cat(ps), torch.cat(fs)


@torch.no_grad()
def render_view(
    scene: Scene,
    K: np.ndarray,
    E: np.ndarray,
    height: int,
    width: int,
    ssaa: int = 2,
    device: str | torch.device = "cpu",
    chunk: int = 1 << 18,
):
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    Et = torch.as_tensor(E, dtype=torch.float32, device=device)

    # geometry pass through pixel centers -> exact GT depth / ids
    o, d, zs = camera_rays(Kt, Et, height, width)
    o, d, zs = o.reshape(-1, 3), d.reshape(-1, 3), zs.reshape(-1)
    t, n, pid, face = _render_rays(scene, o, d, chunk)
    if torch.isinf(t).any():
        raise RuntimeError("Some rays escaped the scene; is the camera inside the room?")
    depth = (t * zs).reshape(height, width)

    # color pass with ssaa x ssaa sub-pixel samples
    rgb = torch.zeros(height * width, 3, device=device)
    offsets = [(i + 0.5) / ssaa for i in range(ssaa)]
    base = pixel_grid(height, width, device=device, offset=0.0)
    for oy in offsets:
        for ox in offsets:
            uv = base + torch.tensor([ox, oy], device=device)
            o2, d2, _ = camera_rays(Kt, Et, height, width, uv=uv)
            o2, d2 = o2.reshape(-1, 3), d2.reshape(-1, 3)
            t2, n2, p2, f2 = _render_rays(scene, o2, d2, chunk)
            pts = o2 + d2 * t2[:, None]
            rgb += shade(scene, pts, n2, p2, f2)
    rgb /= ssaa * ssaa
    return {
        "rgb": rgb.reshape(height, width, 3),
        "depth": depth,
        "normals": n.reshape(height, width, 3),
        "prim_id": pid.reshape(height, width),
        "face_id": face.reshape(height, width),
    }


def render(scene: Scene, cams: Cameras, ssaa: int = 2, device: str | torch.device = "cpu") -> RenderOutput:
    outs = [
        render_view(scene, cams.intrinsics[i], cams.extrinsics[i], cams.height, cams.width, ssaa=ssaa, device=device)
        for i in range(len(cams))
    ]
    return RenderOutput(
        rgb=np.stack([(o["rgb"].cpu().numpy() * 255 + 0.5).astype(np.uint8) for o in outs]),
        depth=np.stack([o["depth"].cpu().numpy().astype(np.float32) for o in outs]),
        normals=np.stack([o["normals"].cpu().numpy().astype(np.float32) for o in outs]),
        prim_id=np.stack([o["prim_id"].cpu().numpy().astype(np.int16) for o in outs]),
        face_id=np.stack([o["face_id"].cpu().numpy().astype(np.int8) for o in outs]),
    )
