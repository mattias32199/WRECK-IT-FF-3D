"""Synthetic data generation: scenes, cameras, renderer."""
import numpy as np
import pytest
import torch

from confprobe import geometry as G
from confprobe.renderer import render, render_view
from confprobe.scene import (
    KIND_BOX, KIND_ROOM, KIND_SPHERE, Cameras, CameraConfig, Primitive, Scene, SceneConfig,
    point_inside_any_object, random_cameras, random_scene, random_texture,
)

H, W = 56, 70


def _single_object_scene(kind, center, he):
    rng = np.random.default_rng(0)
    cfg = SceneConfig()
    room = Primitive(KIND_ROOM, np.array([0, 0, 5.0], np.float32), np.array([10, 10, 5.0], np.float32),
                     np.eye(3, dtype=np.float32), random_texture(rng, 6, cfg))
    obj = Primitive(kind, np.asarray(center, np.float32), np.asarray(he, np.float32),
                    np.eye(3, dtype=np.float32), random_texture(rng, 6 if kind == KIND_BOX else 1, cfg))
    return Scene([room, obj], np.array([[0, 0, 1.0]], np.float32), np.array([0.6], np.float32), 0.4)


def _cam_looking_down_y(eye_y=-4.0, z=1.0):
    K = G.intrinsics_from_fov(W, H, 60.0).numpy()
    E = G.look_at(torch.tensor([0.0, eye_y, z]), torch.tensor([0.0, 0.0, z])).numpy()
    return K, E


def test_sphere_depth_at_center_pixel():
    scene = _single_object_scene(KIND_SPHERE, [0, 0, 1.0], [0.5, 0.5, 0.5])
    K, E = _cam_looking_down_y()
    out = render_view(scene, K, E, H, W, ssaa=1)
    # pixel centers straddle the optical axis; nearest center is within half a pixel
    d = out["depth"][H // 2 - 1 : H // 2 + 1, W // 2 - 1 : W // 2 + 1]
    assert torch.allclose(d, torch.full_like(d, 3.5), atol=5e-3)
    assert (out["prim_id"][H // 2, W // 2]).item() == 1


def test_box_front_face_depth_is_planar():
    scene = _single_object_scene(KIND_BOX, [0, 0, 1.0], [0.8, 0.5, 0.6])
    K, E = _cam_looking_down_y()
    out = render_view(scene, K, E, H, W, ssaa=1)
    on_box = out["prim_id"] == 1
    assert on_box.sum() > 50
    # front face is at y = -0.5, camera at y = -4: z-depth must be exactly 3.5 on that face
    front = on_box & (out["normals"][..., 1] < -0.99)
    assert torch.allclose(out["depth"][front], torch.full_like(out["depth"][front], 3.5), atol=1e-4)


def test_room_walls_hit_everywhere_and_normals_inward():
    scene = random_scene(0)
    cams = random_cameras(0, scene, CameraConfig(n_views=2, height=H, width=W))
    out = render(scene, cams, ssaa=1)
    assert np.isfinite(out.depth).all() and (out.depth > 0).all()
    room_px = out.prim_id == 0
    pts = out.world_points(cams)[room_px]
    nrm = out.normals[room_px]
    # inward normals point toward the room center
    to_center = scene.room.center - pts
    assert ((nrm * to_center).sum(-1) > 0).mean() > 0.99


def test_unprojected_points_lie_on_surfaces():
    scene = _single_object_scene(KIND_SPHERE, [0.3, 0.2, 1.1], [0.6] * 3)
    K, E = _cam_looking_down_y()
    out = render_view(scene, K, E, H, W, ssaa=1)
    X = G.unproject_depth(out["depth"], torch.from_numpy(K), torch.from_numpy(E))
    sph = out["prim_id"] == 1
    r = (X[sph] - torch.tensor([0.3, 0.2, 1.1])).norm(dim=-1)
    assert torch.allclose(r, torch.full_like(r, 0.6), atol=1e-3)


def test_multiview_photometric_consistency(small_sample):
    """A 3D point seen in two views has (almost) the same color: textures are solid and
    shading is view independent. This is what makes the scenes valid multi-view input."""
    s = small_sample
    wp = s.world_points
    i, j = 0, 1
    X = torch.from_numpy(wp[i]).reshape(-1, 3)
    uv, z = G.project(X, torch.from_numpy(s.cams.intrinsics[j]), torch.from_numpy(s.cams.extrinsics[j]))
    col, row = uv[:, 0].floor().long(), uv[:, 1].floor().long()
    inb = (col >= 0) & (col < s.cams.width) & (row >= 0) & (row < s.cams.height) & (z > 0)
    col, row = col.clamp(0, s.cams.width - 1), row.clamp(0, s.cams.height - 1)
    dj = torch.from_numpy(s.render.depth[j])[row, col]
    vis = inb & ((dj - z).abs() < 0.003 * z)
    assert vis.sum() > 200
    ci = s.render.rgb[i].reshape(-1, 3)[vis.numpy()].astype(float)
    cj = s.render.rgb[j][row[vis].numpy(), col[vis].numpy()].astype(float)
    # median color difference is small (residual from texture sampling at different pixel footprints)
    assert np.median(np.abs(ci - cj).mean(-1)) < 12


def test_scene_generation_is_deterministic():
    a, b = random_scene(5), random_scene(5)
    assert len(a.primitives) == len(b.primitives)
    for p, q in zip(a.primitives, b.primitives):
        assert np.allclose(p.center, q.center) and np.allclose(p.texture.freqs, q.texture.freqs)
    assert not np.allclose(random_scene(6).primitives[0].half_extents, a.primitives[0].half_extents)


def test_scene_dict_roundtrip():
    s = random_scene(1)
    s2 = Scene.from_dict(s.to_dict())
    cams = random_cameras(1, s, CameraConfig(n_views=1, height=H, width=W))
    r1, r2 = render(s, cams, ssaa=1), render(s2, cams, ssaa=1)
    assert np.array_equal(r1.rgb, r2.rgb) and np.allclose(r1.depth, r2.depth)


def test_cameras_inside_room_and_outside_objects():
    for seed in range(5):
        s = random_scene(seed)
        cams = random_cameras(seed, s, CameraConfig(n_views=8))
        C = G.camera_centers(torch.from_numpy(cams.extrinsics)).numpy()
        lo = s.room.center - s.room.half_extents
        hi = s.room.center + s.room.half_extents
        assert ((C > lo) & (C < hi)).all()
        assert not any(point_inside_any_object(s, c, margin=0.0) for c in C)


def test_render_output_format(small_sample):
    r = small_sample.render
    S, Hs, Ws = r.depth.shape
    assert r.rgb.shape == (S, Hs, Ws, 3) and r.rgb.dtype == np.uint8
    assert Hs % 14 == 0 and Ws % 14 == 0
    assert r.rgb.std() > 20  # textured, not flat
    assert set(np.unique(r.prim_id)) >= {0}
