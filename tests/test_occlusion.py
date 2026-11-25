"""Occlusion control and geometric support (the experiment's independent variable)."""
import numpy as np
import pytest
import torch

from confprobe import geometry as G
from confprobe.occlusion import (
    OcclusionConfig, apply_fill, footprint_to_mask, region_footprints, sample_regions,
)
from confprobe.renderer import render
from confprobe.scene import KIND_BOX, KIND_ROOM, Cameras, Primitive, Scene, SceneConfig, random_texture
from confprobe.support import compute_support

H, W = 56, 70


def _wall_scene(with_blocker: bool):
    """Two cameras facing a wall at y = 2; optionally a box that hides part of the wall from camera 1."""
    rng = np.random.default_rng(0)
    cfg = SceneConfig()
    prims = [Primitive(KIND_ROOM, np.array([0, 0, 2.0], np.float32), np.array([4, 2, 2.0], np.float32),
                       np.eye(3, dtype=np.float32), random_texture(rng, 6, cfg))]
    if with_blocker:
        prims.append(Primitive(KIND_BOX, np.array([1.2, 0.0, 1.0], np.float32), np.array([0.3, 0.1, 0.4], np.float32),
                               np.eye(3, dtype=np.float32), random_texture(rng, 6, cfg)))
    scene = Scene(prims, np.array([[0, 0, 1.0]], np.float32), np.array([0.6], np.float32), 0.4)
    K = G.intrinsics_from_fov(W, H, 60.0).numpy()
    Es = [G.look_at(torch.tensor([x, -1.5, 1.0]), torch.tensor([x, 2.0, 1.0])).numpy() for x in (0.0, 0.4)]
    cams = Cameras(np.stack(Es), np.stack([K, K]), H, W)
    return scene, cams, render(scene, cams, ssaa=1)


def test_support_overlapping_views_without_masks():
    scene, cams, out = _wall_scene(False)
    sup = compute_support(out.depth, cams.intrinsics, cams.extrinsics, normals=out.normals)
    # pixels whose wall point lies in the other camera's frustum have support 2, the rest 1
    assert sup.support.min() == 1 and sup.support.max() == 2
    assert np.array_equal(sup.support, sup.n_visible)
    assert sup.self_visible.all()
    # camera 1 is shifted +x, so view 0's left edge is not seen by view 1
    assert (sup.support[0][:, :3] == 1).all()
    assert (sup.support[0][:, W // 2] == 2).mean() > 0.9


def test_support_respects_geometric_occlusion():
    scene, cams, out = _wall_scene(True)
    sup = compute_support(out.depth, cams.intrinsics, cams.extrinsics, normals=out.normals)
    wall0 = out.prim_id[0] == 0
    # some wall pixels of view 0 are hidden behind the box from view 1 -> support 1
    behind = wall0 & (sup.support[0] == 1)
    in_frustum_1 = sup.support[0] >= 1
    assert behind.sum() > 10
    # cross-check one such pixel by explicit reprojection
    r, c = np.argwhere(behind & (np.arange(W)[None] > W // 3))[0]
    X = G.np_unproject_depth(out.depth[0], cams.intrinsics[0], cams.extrinsics[0])[r, c]
    uv, z = G.project(torch.from_numpy(X), torch.from_numpy(cams.intrinsics[1]), torch.from_numpy(cams.extrinsics[1]))
    u, v = int(uv[0]), int(uv[1])
    if 0 <= u < W and 0 <= v < H:
        assert out.depth[1][v, u] < z.item() - 0.05  # something nearer occupies that pixel


def test_masks_reduce_support_and_bits_are_consistent():
    scene, cams, out = _wall_scene(False)
    masks = np.zeros((2, H, W), bool)
    masks[1] = True  # hide all of view 1
    sup = compute_support(out.depth, cams.intrinsics, cams.extrinsics, masks, normals=out.normals)
    assert sup.support[0].max() == 1  # view 1 no longer contributes
    assert (sup.support[1] <= 1).all()  # view-1 pixels: only view 0 can support them now
    assert not sup.self_visible[1].any()
    popcount = np.vectorize(lambda b: bin(int(b)).count("1"))(sup.support_bits)
    assert np.array_equal(popcount, sup.support)
    masks[0] = True
    sup2 = compute_support(out.depth, cams.intrinsics, cams.extrinsics, masks)
    assert (sup2.support == 0).all()


def test_footprint_mask_shapes():
    fp = np.zeros((20, 20), bool)
    fp[5:8, 4:6] = True
    fp[12, 15] = True
    bbox = footprint_to_mask(fp, "bbox", 0)
    assert bbox[5:13, 4:16].all() and bbox.sum() == 8 * 12
    sil = footprint_to_mask(fp, "silhouette", 1)
    assert sil[fp].all() and sil.sum() > fp.sum()
    with pytest.raises(ValueError):
        footprint_to_mask(fp, "circle", 0)


def test_region_footprints():
    wp = np.zeros((1, 3, 3, 3), np.float32)
    wp[0, 1, 1] = [1, 0, 0]
    fp = region_footprints(wp, np.array([1.0, 0, 0]), 0.5)
    assert fp.sum() == 1 and fp[0, 1, 1]


def test_sampled_regions_hit_target_support(small_sample):
    """Core guarantee: no pixel inside a region gets more support than its target, and
    a substantial share reaches the target exactly."""
    s = small_sample
    cfg = OcclusionConfig(regions_per_condition=3, target_supports=(0, 1, 2), min_visible_views=3,
                          min_region_pixels=20, max_image_frac=0.3)
    rng = np.random.default_rng(0)
    regions, masks, rmap = sample_regions(s.world_points, s.render.prim_id, s.scene, cfg, rng)
    assert len(regions) >= 2
    sup = compute_support(s.render.depth, s.cams.intrinsics, s.cams.extrinsics, masks, normals=s.render.normals)
    for reg in regions:
        px = rmap == reg.region_id
        assert px.any()
        assert sup.support[px].max() <= reg.target_support, (reg.region_id, reg.target_support)
        if reg.target_support > 0:
            assert (sup.support[px] == reg.target_support).mean() > 0.2
        # kept views are untouched inside the region, masked views are covered
        for j in reg.kept_views:
            assert not masks[j][px[j]].any()
        for j in reg.masked_views:
            assert masks[j][px[j]].all()
        # regions do not overlap
    for a in regions:
        for b in regions:
            if a.region_id < b.region_id:
                assert np.linalg.norm(a.center - b.center) > a.radius + b.radius


def test_region_sampling_is_seeded(small_sample):
    s = small_sample
    cfg = OcclusionConfig(regions_per_condition=2, min_visible_views=3, min_region_pixels=20, max_image_frac=0.3,
                          target_supports=(0, 1))
    r1 = sample_regions(s.world_points, s.render.prim_id, s.scene, cfg, np.random.default_rng(7))
    r2 = sample_regions(s.world_points, s.render.prim_id, s.scene, cfg, np.random.default_rng(7))
    assert np.array_equal(r1[1], r2[1])


@pytest.mark.parametrize("fill", ["gray", "black", "mean", "noise"])
def test_apply_fill_only_touches_masked_pixels(fill):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (2, 10, 12, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 12), bool)
    masks[0, 2:5, 3:7] = True
    out = apply_fill(imgs, masks, fill, rng)
    assert np.array_equal(out[~masks], imgs[~masks])
    if fill == "gray":
        assert (out[masks] == 127).all()
    if fill == "black":
        assert (out[masks] == 0).all()
    assert out is not imgs


def test_apply_fill_unknown():
    with pytest.raises(ValueError):
        apply_fill(np.zeros((1, 2, 2, 3), np.uint8), np.ones((1, 2, 2), bool), "blur", np.random.default_rng())
