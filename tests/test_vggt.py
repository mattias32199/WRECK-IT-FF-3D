"""VGGT runner.

Fast tests use (a) a constant fake model with VGGT's output contract and (b) a tiny,
randomly initialized model built from the official vggt modules. The `slow` tests
load the real VGGT-1B weights and need a GPU:

    pytest tests/test_vggt.py -m slow                   # HF hub download
    VGGT_CHECKPOINT=/path/model.pt pytest -m slow       # offline
"""
import os
import warnings

import numpy as np
import pytest
import torch

from confprobe import geometry as G
from confprobe.vggt_runner import VGGTPrediction, VGGTRunner
from conftest import ConstantVGGT
from tiny_vggt import build_tiny_vggt

S, H, W = 3, 42, 56


def _imgs(seed=0, s=S, h=H, w=W):
    return np.random.default_rng(seed).integers(0, 256, (s, h, w, 3), dtype=np.uint8)


@pytest.fixture()
def runner():
    return VGGTRunner(ConstantVGGT(), device="cpu", strict_size=False)


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def test_uint8_and_float_inputs_are_equivalent(runner):
    x = _imgs()
    a = runner.preprocess(x)
    b = runner.preprocess(torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255)
    c = runner.preprocess(torch.from_numpy(x).permute(0, 3, 1, 2))  # uint8 tensor
    assert a.shape == (S, 3, H, W) and a.dtype == torch.float32
    assert torch.allclose(a, b) and torch.allclose(a, c)
    assert 0 <= a.min() and a.max() <= 1


def test_channel_order_preserved(runner):
    x = np.zeros((1, 14, 14, 3), np.uint8)
    x[..., 0] = 255  # pure red
    t = runner.preprocess(x)
    assert t[0, 0].min() == 1.0 and t[0, 1:].max() == 0.0


@pytest.mark.parametrize("shape", [(S, 40, 56, 3), (S, 42, 50, 3)])
def test_rejects_non_multiple_of_14(runner, shape):
    with pytest.raises(ValueError, match="multiples of 14"):
        runner.preprocess(np.zeros(shape, np.uint8))


def test_rejects_bad_layout_and_range(runner):
    with pytest.raises(ValueError):
        runner.preprocess(np.zeros((S, 3, H, W), np.uint8))  # channels-first numpy
    with pytest.raises(ValueError):
        runner.preprocess(torch.full((S, 3, H, W), 2.0))
    with pytest.raises(TypeError):
        runner.preprocess([[0]])


def test_warns_on_non_518_long_side():
    r = VGGTRunner(ConstantVGGT(), device="cpu", strict_size=True)
    with pytest.warns(UserWarning, match="518"):
        r.preprocess(_imgs())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.preprocess(_imgs(s=1, h=392, w=518))


# --------------------------------------------------------------------------- #
# output conversion
# --------------------------------------------------------------------------- #
def test_run_output_shapes_and_types(runner):
    p = runner.run(_imgs())
    assert isinstance(p, VGGTPrediction)
    assert p.depth.shape == (S, H, W) and p.depth_conf.shape == (S, H, W)
    assert p.world_points.shape == (S, H, W, 3) and p.world_points_conf.shape == (S, H, W)
    assert p.extrinsic.shape == (S, 3, 4) and p.intrinsic.shape == (S, 3, 3)
    assert all(getattr(p, k).dtype == np.float32 for k in ("depth", "depth_conf", "extrinsic"))
    assert np.allclose(p.depth, 2.0)


def test_model_receives_the_images(runner):
    x = _imgs(1)
    p = runner.run(x)
    seen = runner.model.last_input
    assert torch.allclose(seen, torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255)
    # ConstantVGGT's confidence is 1 + mean intensity -> verifies per-view/per-pixel plumbing
    assert np.allclose(p.depth_conf, 1 + x.astype(np.float32).mean(-1) / 255, atol=1e-5)


def test_pose_decoding_recovers_known_cameras():
    Es = torch.stack([G.look_at(torch.tensor([float(i), -2.0, 1.0]), torch.zeros(3)) for i in range(S)])
    r = VGGTRunner(ConstantVGGT(extrinsics=Es), device="cpu", strict_size=False)
    p = r.run(_imgs())
    assert np.allclose(p.extrinsic, Es.numpy(), atol=1e-4)
    assert np.allclose(p.intrinsic[:, 0, 0], 100.0, atol=1e-2)
    assert np.allclose(p.intrinsic[:, :2, 2], [W / 2, H / 2])


def test_shape_mismatch_raises():
    class Bad(ConstantVGGT):
        def forward(self, images):
            out = super().forward(images)
            out["depth"] = out["depth"][:, :, :-1]
            return out

    with pytest.raises(RuntimeError, match="depth"):
        VGGTRunner(Bad(), device="cpu", strict_size=False).run(_imgs())


def test_prediction_save_load(tmp_path, runner):
    p = runner.run(_imgs())
    p.save(tmp_path / "p.npz", half=False)
    q = VGGTPrediction.load(tmp_path / "p.npz")
    for k in ("depth", "depth_conf", "world_points", "extrinsic", "intrinsic"):
        assert np.allclose(getattr(p, k), getattr(q, k))


def test_depth_points_unprojection(runner):
    p = runner.run(_imgs())
    pts = p.depth_points()
    assert pts.shape == (S, H, W, 3)
    assert np.allclose(pts[..., 2], 2.0, atol=1e-4)  # identity cameras, depth 2


# --------------------------------------------------------------------------- #
# real architecture (random weights)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_runner():
    m = build_tiny_vggt()
    if m is None:
        pytest.skip("vggt package not installed")
    return VGGTRunner(m, device="cpu", strict_size=False)


def test_tiny_vggt_contract(tiny_runner):
    p = tiny_runner.run(_imgs(2))
    assert p.depth.shape == (S, H, W)
    assert (p.depth > 0).all()  # exp activation
    assert (p.depth_conf >= 1).all() and (p.world_points_conf >= 1).all()  # expp1 activation
    assert np.isfinite(p.world_points).all()
    R = p.extrinsic[:, :, :3]
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-4)


def test_tiny_vggt_is_deterministic_and_input_dependent(tiny_runner):
    a, b = tiny_runner.run(_imgs(3)), tiny_runner.run(_imgs(3))
    assert np.allclose(a.depth, b.depth) and np.allclose(a.depth_conf, b.depth_conf)
    c = tiny_runner.run(_imgs(4))
    assert not np.allclose(a.depth, c.depth)


def test_tiny_vggt_frames_interact(tiny_runner):
    """Changing view 2 changes predictions for view 0 (global attention is wired up)."""
    x = _imgs(5)
    y = x.copy()
    y[2] = 255 - y[2]
    a, b = tiny_runner.run(x), tiny_runner.run(y)
    # random init uses LayerScale 0.01, so the effect is small but must be nonzero;
    # as a control, re-running the unchanged input gives bit-identical output
    assert np.abs(a.depth[0] - b.depth[0]).max() > 1e-6
    assert np.array_equal(a.depth[0], tiny_runner.run(x).depth[0])


# --------------------------------------------------------------------------- #
# real VGGT-1B (opt-in)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_runner():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("vggt")
    return VGGTRunner.from_pretrained(checkpoint=os.environ.get("VGGT_CHECKPOINT"), device="cuda")


@pytest.mark.slow
def test_real_vggt_reconstructs_synthetic_scene(real_runner):
    """Sanity check that rendered scenes are in-distribution enough for VGGT."""
    from confprobe.dataset import ConditionConfig, build_conditions, generate_scene
    from confprobe.metrics import EvalConfig, evaluate_condition

    sample = generate_scene(0, device="cuda")
    clean = build_conditions(sample, ConditionConfig(n_masked=0), seed=0, device="cuda")[0]
    pred = real_runner.run(clean.images)
    pix, poses, info = evaluate_condition(clean, clean.gt(sample), pred, EvalConfig())
    assert np.allclose(pred.extrinsic[0], np.eye(3, 4), atol=1e-2)  # view 0 is the world frame
    assert pix.abs_rel.median() < 0.1, pix.abs_rel.median()
    assert poses.rot_err_deg[1:].mean() < 10
    assert (pred.depth_conf >= 1).all()


@pytest.mark.slow
def test_real_vggt_masking_changes_masked_pixels_most(real_runner):
    from confprobe.dataset import ConditionConfig, build_conditions, generate_scene

    sample = generate_scene(1, device="cuda")
    clean, masked = build_conditions(sample, ConditionConfig(n_masked=1), seed=1, device="cuda")
    a, b = real_runner.run(clean.images), real_runner.run(masked.images)
    d = np.abs(np.log(a.depth_conf) - np.log(b.depth_conf))
    assert d[masked.masks].mean() > d[~masked.masks].mean()
