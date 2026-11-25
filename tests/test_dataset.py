"""Conditions and on-disk storage of the synthetic dataset."""
import numpy as np

from confprobe.dataset import ConditionSample, SceneSample


def test_conditions_built(small_conditions, small_sample):
    names = [c.name for c in small_conditions]
    assert names[0] == "clean" and "masked_0" in names and "subset2_0" in names
    S = len(small_sample.cams)
    for c in small_conditions:
        n = len(c.view_indices)
        assert c.images.shape == (n, *small_sample.render.rgb.shape[1:])
        for arr in (c.masks, c.support, c.n_visible, c.region_map):
            assert arr.shape == c.images.shape[:3]
        assert (c.support <= c.n_visible).all()
        assert (c.n_visible <= n).all()
        assert max(c.view_indices) < S


def test_clean_condition_is_unmasked(small_conditions, small_sample):
    clean = small_conditions[0]
    assert not clean.masks.any() and clean.regions == []
    assert np.array_equal(clean.images, small_sample.render.rgb)
    assert np.array_equal(clean.support, clean.n_visible)
    assert clean.support.min() >= 1  # every pixel is at least seen by its own view


def test_masked_condition_has_support_zero_and_kinds(small_conditions):
    m = next(c for c in small_conditions if c.name == "masked_0")
    assert m.masks.any() and len(m.regions) >= 1
    assert m.kind == "masked"
    assert (m.support[m.masks] <= m.n_visible[m.masks] - 0).all()
    if any(r["target_support"] == 0 for r in m.regions):
        assert (m.support == 0).any()
    # masked pixels show the fill
    assert (m.images[m.masks] == 127).all()
    tsm = m.target_support_map
    assert set(np.unique(tsm[m.region_map >= 0])) <= {r["target_support"] for r in m.regions}


def test_subset_condition_reduces_support(small_conditions):
    sub = next(c for c in small_conditions if c.kind == "subset")
    assert len(sub.view_indices) == 2 and sub.support.max() <= 2


def test_gt_slicing(small_conditions, small_sample):
    sub = next(c for c in small_conditions if c.kind == "subset")
    gt = sub.gt(small_sample)
    assert np.array_equal(gt["depth"], small_sample.render.depth[sub.view_indices])
    assert gt["world_points"].shape == (2, *small_sample.render.depth.shape[1:], 3)


def test_save_load_roundtrip(tmp_path, small_sample, small_conditions):
    small_sample.save(tmp_path)
    for c in small_conditions:
        c.save(tmp_path)
    s2 = SceneSample.load(tmp_path, small_sample.scene_id)
    assert np.array_equal(s2.render.rgb, small_sample.render.rgb)
    assert np.allclose(s2.render.depth, small_sample.render.depth)
    assert np.allclose(s2.cams.extrinsics, small_sample.cams.extrinsics)
    assert sorted(ConditionSample.list_names(tmp_path, small_sample.scene_id)) == sorted(c.name for c in small_conditions)
    for c in small_conditions:
        c2 = ConditionSample.load(tmp_path, small_sample.scene_id, c.name)
        assert c2.view_indices == c.view_indices
        for k in ("images", "masks", "support", "support_bits", "region_map"):
            assert np.array_equal(getattr(c2, k), getattr(c, k))
        assert c2.regions == c.regions
