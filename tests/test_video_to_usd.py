"""Geometry half of the video -> USD pipeline, on a synthetic scene with known answers.

No GPU or VGGT needed: we ray-cast exact depth maps of a floor + a unit box,
express everything in the first camera's OpenCV frame (exactly how VGGT
reports it), and check the pipeline recovers a level floor at z = 0, the box
top at z = 1, and the camera heights.

    pytest tests/test_video_to_usd.py
"""

from __future__ import annotations

import numpy as np
import pytest

from wreck_it_ff_3d.models.prediction import Prediction

H, W, F = 120, 160, 140.0
CAM_HEIGHT = 1.5
BOX = (np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0]))  # z-up world, floor z = 0


def look_at(eye, target):
    """Camera-from-world [R|t], OpenCV axes, in a z-up world."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return np.concatenate([R, (-R @ eye)[:, None]], 1)


def raycast(E, K):
    """Exact z-depth and color of floor (z=0, |x|,|y| < 4) + box, for one camera."""
    R, t = E[:, :3], E[:, 3]
    o = -R.T @ t
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    d = rays @ R  # world directions (cam z component = 1 -> ray parameter is z-depth)
    best = np.full((H, W), np.inf)
    color = np.zeros((H, W, 3), np.uint8)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o[2] / d[..., 2]
        p = o + s[..., None] * d
        hit = (s > 0) & (np.abs(p[..., 0]) < 4) & (np.abs(p[..., 1]) < 4)
        best[hit] = s[hit]
        checker = ((np.floor(p[..., 0] * 2) + np.floor(p[..., 1] * 2)) % 2 == 0)
        color[hit] = np.where(checker[hit, None], 200, 90)

        lo, hi = BOX  # slab test
        t1, t2 = (lo - o) / d, (hi - o) / d
        tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        box = (tn <= tf) & (tn > 0) & (tn < best)
        best[box] = tn[box]
        color[box] = (200, 40, 40)
    depth = np.where(np.isfinite(best), best, 0.0).astype(np.float32)
    return depth, color


def synthetic_prediction(n_views=12, tilt_noise_deg=8.0, seed=0) -> Prediction:
    rng = np.random.default_rng(seed)
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    Es, depths, imgs = [], [], []
    for a in np.linspace(0, 1.5 * np.pi, n_views):
        eye = np.array([3.5 * np.cos(a), 3.5 * np.sin(a), CAM_HEIGHT])
        E = look_at(eye, np.array([0.0, 0.0, 0.4]))
        # hand-held roll/pitch noise
        axis = rng.normal(size=3)
        ang = np.radians(rng.uniform(0, tilt_noise_deg))
        ax = axis / np.linalg.norm(axis)
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rn = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
        E = np.concatenate([Rn @ E[:, :3], (Rn @ E[:, 3])[:, None]], 1)
        d, c = raycast(E, K)
        Es.append(E), depths.append(d), imgs.append(c)

    # Re-express in camera 0's OpenCV frame, as VGGT does.
    E0 = np.vstack([Es[0], [0, 0, 0, 1]])
    E0_inv = np.linalg.inv(E0)
    Es = np.stack([(np.vstack([E, [0, 0, 0, 1]]) @ E0_inv)[:3] for E in Es])
    S = n_views
    depth = np.stack(depths)
    return Prediction(
        extrinsics=Es.astype(np.float32),
        intrinsics=np.repeat(K[None], S, 0).astype(np.float32),
        depth=depth,
        depth_conf=np.where(depth > 0, 5.0, 1.0).astype(np.float32) + rng.uniform(0, 0.1, depth.shape).astype(np.float32),
        world_points=np.zeros((S, H, W, 3), np.float32),
        world_points_conf=np.ones((S, H, W), np.float32),
        images=np.stack(imgs),
        meta={"synthetic": True},
    )


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    pytest.importorskip("open3d")
    pytest.importorskip("pxr")
    from wreck_it_ff_3d.pipelines.video_to_usd import prediction_to_usd

    out = tmp_path_factory.mktemp("scene")
    pred = synthetic_prediction()
    summary = prediction_to_usd(pred, out, keep_conf=0.95, voxel_frac=1 / 150)
    return out, summary, pred


def test_floor_level_and_box_height(result):
    out, summary, _ = result
    assert summary["align"]["floor_found"]
    voxel = summary["fusion"]["voxel"]
    assert abs(summary["bbox_min"][2]) < 3 * voxel, summary["bbox_min"]
    assert abs(summary["bbox_max"][2] - 1.0) < 3 * voxel, summary["bbox_max"]  # box top


def test_usd_structure_and_cameras(result):
    from pxr import Usd, UsdGeom, UsdPhysics

    out, summary, pred = result
    stage = Usd.Stage.Open(str(out / "scene.usd"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetPath() == "/Reconstruction"

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Reconstruction/Mesh"))
    n_pts = len(mesh.GetPointsAttr().Get())
    assert n_pts == summary["vertices"]
    assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor").Get()) == n_pts
    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)

    box_center = None
    for i in range(pred.num_views):
        cam = stage.GetPrimAtPath(f"/Reconstruction/Cameras/view_{i:03d}")
        assert cam.IsValid()
        M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)).T  # column-vector form
        assert abs(M[2, 3] - CAM_HEIGHT) < 0.05, (i, M[2, 3])
        # USD cameras look down -Z; every view was aimed at the box at height 0.4
        look = -M[:3, 2]
        if box_center is None:
            box_center = M[:3, 3] + look * (0.4 - M[2, 3]) / look[2]
        to_box = box_center - M[:3, 3]
        cos = look @ to_box / np.linalg.norm(to_box)
        assert cos > np.cos(np.radians(12)), (i, np.degrees(np.arccos(cos)))


def test_ply_written(result):
    out, _, _ = result
    assert (out / "mesh.ply").stat().st_size > 0


def test_extract_frames_prefers_sharp(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from wreck_it_ff_3d.data.video import extract_frames

    rng = np.random.default_rng(0)
    base = (rng.uniform(0, 255, (240, 320, 3))).astype(np.uint8)
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(100):
        frame = np.roll(base, i, axis=1)
        if i % 2:  # odd frames motion-blurred
            frame = cv2.GaussianBlur(frame, (0, 0), 4)
        vw.write(frame)
    vw.release()

    paths = extract_frames(path, tmp_path / "frames", num_frames=10, candidates_per_frame=5, spacing="time")
    idx = [int(p.stem.split("_")[1]) for p in paths]
    assert len(idx) == 10 and idx == sorted(idx)
    assert all(i % 2 == 0 for i in idx), idx
    assert idx[0] < 10 and idx[-1] >= 90  # covers the whole clip


def test_camera_height_sets_scale(tmp_path):
    pytest.importorskip("open3d")
    pytest.importorskip("pxr")
    from wreck_it_ff_3d.pipelines.video_to_usd import prediction_to_usd

    s = prediction_to_usd(synthetic_prediction(), tmp_path, keep_conf=0.95, voxel_frac=1 / 150,
                          camera_height=2 * CAM_HEIGHT, cameras=False)
    assert abs(s["scale_m_per_vggt_unit"] - 2.0) < 0.05, s["scale_m_per_vggt_unit"]
    assert abs(s["bbox_max"][2] - 2.0) < 0.1, s["bbox_max"]  # box top, doubled


def test_motion_spacing_puts_frames_where_the_camera_moves(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from wreck_it_ff_3d.data.video import extract_frames

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (240, 1600, 3)).astype(np.uint8), (0, 0), 1.5)
    path = tmp_path / "pan.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    x = 0
    for i in range(120):  # slow pan (1 px/frame) for 60 frames, then fast (8 px/frame) like a turn
        x += 1 if i < 60 else 8
        vw.write(np.ascontiguousarray(base[:, x:x + 320]))
    vw.release()

    def picks(spacing):
        out = extract_frames(path, tmp_path / spacing, num_frames=10, spacing=spacing)
        return [int(p.stem.split("_")[1]) for p in out]

    fast_time = sum(i >= 60 for i in picks("time"))
    fast_motion = sum(i >= 60 for i in picks("motion"))
    assert fast_time <= 6
    assert fast_motion >= 8, picks("motion")  # 8 * 60 / (60 + 8 * 60) ~ 89% of the motion
