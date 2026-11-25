import math

import numpy as np
import pytest
import torch

from confprobe import geometry as G


def random_extrinsic(seed=0):
    g = torch.Generator().manual_seed(seed)
    eye = torch.randn(3, generator=g) * 2
    target = torch.randn(3, generator=g)
    return G.look_at(eye, target)


def test_look_at_is_rotation_and_points_at_target():
    eye, target = torch.tensor([3.0, -1.0, 1.5]), torch.tensor([0.0, 0.5, 0.3])
    E = G.look_at(eye, target)
    R = E[:, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    assert torch.det(R) == pytest.approx(1.0, abs=1e-5)
    Xc = G.world_to_camera(target, E)
    assert Xc[0].abs() < 1e-5 and Xc[1].abs() < 1e-5 and Xc[2] > 0  # target on optical axis, in front
    assert torch.allclose(G.camera_centers(E), eye, atol=1e-5)


def test_look_at_up_is_image_up():
    # world +z should map to negative image y (up in the image) for a level camera
    E = G.look_at(torch.tensor([0.0, -3.0, 1.0]), torch.tensor([0.0, 0.0, 1.0]))
    up_cam = E[:, :3] @ torch.tensor([0.0, 0.0, 1.0])
    assert up_cam[1] < -0.99


def test_invert_extrinsic_roundtrip():
    E = random_extrinsic(1)
    Einv = G.invert_extrinsic(E)
    X = torch.randn(10, 3)
    back = G.world_to_camera(G.world_to_camera(X, E), Einv)
    assert torch.allclose(back, X, atol=1e-5)


def test_intrinsics_from_fov():
    K = G.intrinsics_from_fov(518, 392, 90.0)
    assert K[0, 0] == pytest.approx(259.0, rel=1e-5)
    assert K[0, 2] == 259.0 and K[1, 2] == 196.0


def test_project_unproject_roundtrip():
    H, W = 28, 42
    K = G.intrinsics_from_fov(W, H, 60.0)
    E = random_extrinsic(2)
    depth = torch.rand(H, W) * 3 + 1
    X = G.unproject_depth(depth, K, E)
    uv, z = G.project(X, K, E)
    assert torch.allclose(z, depth, atol=1e-4)
    assert torch.allclose(uv, G.pixel_grid(H, W), atol=1e-3)


def test_camera_rays_hit_unprojected_points():
    H, W = 14, 28
    K = G.intrinsics_from_fov(W, H, 70.0)
    E = random_extrinsic(3)
    depth = torch.rand(H, W) + 2
    o, d, zs = G.camera_rays(K, E, H, W)
    t = depth / zs  # distance along unit ray
    pts = o + d * t[..., None]
    assert torch.allclose(pts, G.unproject_depth(depth, K, E), atol=1e-4)
    assert torch.allclose(d.norm(dim=-1), torch.ones(H, W), atol=1e-5)


def test_quaternion_roundtrip():
    q = torch.nn.functional.normalize(torch.randn(50, 4), dim=-1)
    q = q * torch.sign(q[:, 3:4])  # w >= 0 representative
    R = G.quat_xyzw_to_matrix(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(50, 3, 3), atol=1e-5)
    q2 = G.matrix_to_quat_xyzw(R)
    assert torch.allclose(q2, q, atol=1e-4)


def test_pose_encoding_roundtrip():
    E = torch.stack([random_extrinsic(s) for s in range(4)])
    K = G.intrinsics_from_fov(518, 392, 65.0).expand(4, 3, 3)
    pe = G.extri_intri_to_pose_encoding(E, K, (392, 518))
    E2, K2 = G.pose_encoding_to_extri_intri(pe, (392, 518))
    assert torch.allclose(E2, E, atol=1e-4)
    assert torch.allclose(K2, K, atol=1e-2)


def test_pose_decoding_matches_official_vggt():
    pytest.importorskip("vggt")
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri as official

    pe = torch.randn(1, 5, 9)
    pe[..., 7:] = pe[..., 7:].abs() * 0.3 + 0.8  # plausible FoVs
    E_ref, K_ref = official(pe, (392, 518))
    E, K = G.pose_encoding_to_extri_intri(pe, (392, 518))
    assert torch.allclose(E, E_ref, atol=1e-5)
    assert torch.allclose(K, K_ref, atol=1e-4)


def test_rotation_angle():
    a = math.radians(30)
    Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    assert G.rotation_angle_deg(torch.eye(3), Rz).item() == pytest.approx(30.0, abs=1e-3)
