"""Alignment, errors and confidence metrics."""
import numpy as np
import pandas as pd
import pytest
import torch

from confprobe import geometry as G
from confprobe.metrics import (
    EvalConfig, auroc, evaluate_condition, median_scale, paired_region_deltas, percentile_rank,
    relative_extrinsics, sparsification, spearman, summarize,
)
from confprobe.vggt_runner import VGGTPrediction


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_auroc_known_values():
    labels = np.array([0, 0, 1, 1], bool)
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert np.isnan(auroc(np.ones(3), np.ones(3, bool)))


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    s, l = rng.normal(size=60), rng.uniform(size=60) < 0.4
    pos, neg = s[l], s[~l]
    ref = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert auroc(s, l) == pytest.approx(ref)


def test_spearman_and_subsampling():
    x = np.arange(1000.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x, max_n=100) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(10), x[:10]))


def test_median_scale_recovers_scale():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 5, size=(2, 8, 8))
    pred = gt / 3.0
    valid = np.ones_like(gt, bool)
    pred_corrupt = pred.copy()
    pred_corrupt[0, :2] *= 10  # outliers
    assert median_scale(pred_corrupt, gt, valid) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        median_scale(pred, gt, np.zeros_like(valid))


def test_relative_extrinsics_first_is_identity():
    Es = np.stack([G.look_at(torch.randn(3) * 2, torch.zeros(3)).numpy() for _ in range(3)])
    rel = relative_extrinsics(Es)
    assert np.allclose(rel[0], np.eye(3, 4), atol=1e-5)
    # relative poses map cam0-frame points to cam_i exactly like the absolute ones map world points
    X = np.random.default_rng(0).normal(size=(5, 3))
    X0 = X @ Es[0, :, :3].T + Es[0, :, 3]
    for i in range(3):
        a = X @ Es[i, :, :3].T + Es[i, :, 3]
        b = X0 @ rel[i, :, :3].T + rel[i, :, 3]
        assert np.allclose(a, b, atol=1e-4)


def test_percentile_rank():
    x = np.array([[3.0, 1.0], [2.0, 4.0]])
    assert np.allclose(percentile_rank(x), [[0.75, 0.25], [0.5, 1.0]])


def test_sparsification_perfect_confidence_has_zero_ause():
    err = np.random.default_rng(0).uniform(size=1000)
    fr, curve, oracle, ause = sparsification(err, -err)
    assert ause == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(curve, oracle)
    _, _, _, ause_rand = sparsification(err, np.random.default_rng(1).uniform(size=1000))
    assert ause_rand > 0.2


# --------------------------------------------------------------------------- #
# evaluate_condition on synthetic predictions
# --------------------------------------------------------------------------- #
def _pred_from_gt(cond, gt, scale=0.5, depth_noise=None):
    S, H, W = gt["depth"].shape
    depth = gt["depth"] * scale
    if depth_noise is not None:
        depth = depth * (1 + depth_noise)
    X0 = (gt["world_points"] @ gt["extrinsics"][0, :, :3].T + gt["extrinsics"][0, :, 3]) * scale
    E = relative_extrinsics(gt["extrinsics"])
    E[:, :, 3] *= scale
    conf = 1 + cond.support.astype(np.float32)
    return VGGTPrediction(
        depth=depth.astype(np.float32), depth_conf=conf, world_points=X0.astype(np.float32),
        world_points_conf=conf, extrinsic=E.astype(np.float32), intrinsic=gt["intrinsics"],
        pose_enc=np.zeros((S, 9), np.float32),
    )


def test_perfect_prediction_has_zero_error(small_conditions, small_sample):
    cond = small_conditions[1]  # masked
    gt = cond.gt(small_sample)
    pred = _pred_from_gt(cond, gt, scale=0.5)
    pix, poses, info = evaluate_condition(cond, gt, pred, EvalConfig(pixel_stride=1))
    assert info["scale_depth"] == pytest.approx(2.0, rel=1e-4)
    assert info["scale_points"] == pytest.approx(2.0, rel=1e-4)
    assert pix.abs_rel.max() < 1e-4 and pix.point_err.max() < 1e-3
    assert poses.rot_err_deg.max() < 0.1 and poses.center_err.max() < 1e-3
    S, H, W = gt["depth"].shape
    assert len(pix) == S * H * W
    assert set(pix.columns) >= {"support", "self_visible", "region_id", "target_support", "depth_conf_pct"}
    # table rows line up with condition arrays
    r = pix.sample(20, random_state=0)
    assert np.array_equal(r.support.to_numpy(), cond.support[r.view, r.y, r.x])
    assert np.array_equal(r.self_visible.to_numpy(), ~cond.masks[r.view, r.y, r.x])


def test_pixel_stride(small_conditions, small_sample):
    cond = small_conditions[0]
    gt = cond.gt(small_sample)
    pix, _, _ = evaluate_condition(cond, gt, _pred_from_gt(cond, gt), EvalConfig(pixel_stride=4))
    S, H, W = gt["depth"].shape
    assert len(pix) == S * len(range(0, H, 4)) * len(range(0, W, 4))
    assert (pix.y % 4 == 0).all() and (pix.x % 4 == 0).all()


def test_alignment_ignores_unsupported_pixels(small_conditions, small_sample):
    cond = small_conditions[1]
    gt = cond.gt(small_sample)
    noise = np.zeros_like(gt["depth"])
    noise[cond.support < 2] = 5.0  # wildly wrong only where support is low
    pred = _pred_from_gt(cond, gt, depth_noise=noise)
    pix, _, info = evaluate_condition(cond, gt, pred, EvalConfig(align_on="supported"))
    assert info["scale_depth"] == pytest.approx(2.0, rel=1e-4)
    assert pix[pix.support >= 2].abs_rel.max() < 1e-4


def test_shape_mismatch(small_conditions, small_sample):
    cond = small_conditions[0]
    gt = cond.gt(small_sample)
    pred = _pred_from_gt(cond, gt)
    pred.depth = pred.depth[:, :-14]
    with pytest.raises(ValueError):
        evaluate_condition(cond, gt, pred)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def _toy_table(n=4000, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    support = rng.integers(0, 5, n)
    err = 0.01 + 0.2 / (1 + support) * rng.uniform(0.5, 1.5, n)
    conf = (1 + support + rng.normal(0, 0.3, n)) if informative else rng.uniform(1, 5, n)
    return pd.DataFrame({
        "scene_id": "s", "condition": "masked_0", "kind": "masked", "view": 0,
        "y": np.arange(n) // 64, "x": np.arange(n) % 64,
        "abs_rel": err, "point_err": err, "depth_conf": conf, "point_conf": conf,
        "depth_conf_pct": percentile_rank(conf), "point_conf_pct": percentile_rank(conf),
        "support": support.astype(np.uint8), "self_visible": support > 0, "region_id": -1, "target_support": -1,
    })


def test_summary_informative_vs_random_confidence():
    good = summarize(_toy_table(informative=True))["depth/depth_conf"]
    bad = summarize(_toy_table(informative=False))["depth/depth_conf"]
    assert good["spearman_conf_err"] < -0.8 and abs(bad["spearman_conf_err"]) < 0.1
    assert good["auroc_err_quantile"] > 0.9 and 0.4 < bad["auroc_err_quantile"] < 0.6
    assert good["auroc_support_le1_vs_2plus"] > 0.9
    assert good["ause"] < bad["ause"]


def test_summary_by_support_table():
    s = summarize(_toy_table())
    rows = s["depth/by_support"]
    labels = [r["support"] for r in rows]
    assert "0" in labels and "4+" in labels
    assert sum(r["n"] for r in rows) == 4000
    means = {r["support"]: r["abs_rel_mean"] for r in rows if r["self_visible"]}
    assert means["1"] > means["4+"]


def test_paired_region_deltas():
    base = _toy_table()
    base["condition"], base["kind"] = "clean", "clean"
    masked = base.copy()
    masked["condition"], masked["kind"] = "masked_0", "masked"
    masked.loc[:99, "region_id"] = 0
    masked.loc[:99, "target_support"] = 0
    masked.loc[:99, "depth_conf_pct"] -= 0.3
    d = paired_region_deltas(pd.concat([base, masked]))
    assert len(d) >= 1 and (d.region_id == 0).all()
    assert np.isclose((d.d_conf_pct * d.n).sum() / d.n.sum(), -0.3)
