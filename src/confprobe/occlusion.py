"""Controlled 3D occlusion.

An *occlusion region* is a ball in world space. Its footprint in view j is the
set of pixels whose GT surface point lies inside the ball. To give a region a
target support k, we pick k of the views that see it, keep them untouched, and
hide the region's footprint in every other view that sees it. Support is then
re-measured exactly with :func:`confprobe.support.compute_support`, so the stored
per-pixel support is always ground truth (target_support is only the intent).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

from .scene import KIND_ROOM, Scene


@dataclass
class OcclusionConfig:
    regions_per_condition: int = 4
    target_supports: tuple[int, ...] = (0, 1, 2, 3)
    region_radius: tuple[float, float] = (0.15, 0.35)
    min_visible_views: int = 4  # a region must be seen by at least this many views
    min_region_pixels: int = 150  # footprint size for a view to count as "seeing" the region
    max_image_frac: float = 0.12  # skip regions that cover too much of any image
    object_prob: float = 0.7  # prob. of centering a region on an object (vs. the room)
    min_region_gap: float = 0.05  # regions must not overlap (center distance > r1 + r2 + gap)
    mask_shape: str = "bbox"  # "bbox" (rectangle around footprint) or "silhouette"
    mask_dilation: int = 3  # pixels; guards against support leaks at footprint borders
    fill: str = "gray"  # "gray" | "black" | "mean" | "noise"
    max_tries: int = 400


@dataclass
class OcclusionRegion:
    region_id: int
    center: np.ndarray
    radius: float
    target_support: int
    visible_views: list[int]
    kept_views: list[int]
    masked_views: list[int]
    footprint_pixels: list[int]  # per view (len S)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = [float(x) for x in self.center]
        return d


def region_footprints(world_points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """(S, H, W) bool: pixels whose GT surface point is inside the ball."""
    return np.linalg.norm(world_points - center[None, None, None], axis=-1) < radius


def footprint_to_mask(fp: np.ndarray, shape: str, dilation: int) -> np.ndarray:
    """Convert a single-view footprint (H, W) into the image mask actually applied."""
    if not fp.any():
        return fp.copy()
    if shape == "silhouette":
        m = fp
    elif shape == "bbox":
        rows = np.where(fp.any(1))[0]
        cols = np.where(fp.any(0))[0]
        m = np.zeros_like(fp)
        m[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = True
    else:
        raise ValueError(f"unknown mask_shape {shape!r}")
    if dilation > 0:
        m = ndimage.binary_dilation(m, iterations=dilation)
    return m


def sample_regions(
    world_points: np.ndarray,
    prim_id: np.ndarray,
    scene: Scene | None,
    cfg: OcclusionConfig,
    rng: np.random.Generator,
) -> tuple[list[OcclusionRegion], np.ndarray, np.ndarray]:
    """Sample non-overlapping regions and build masks that realize their target supports.

    Args:
        world_points: (S, H, W, 3) GT world points of the views in the condition.
        prim_id: (S, H, W) primitive index per pixel (0 = room).

    Returns:
        regions, masks (S, H, W) bool, region_id map (S, H, W) int16 (-1 = none).
    """
    S, H, W, _ = world_points.shape
    masks = np.zeros((S, H, W), dtype=bool)
    region_map = np.full((S, H, W), -1, dtype=np.int16)
    regions: list[OcclusionRegion] = []
    obj_pix = np.argwhere(prim_id != 0)
    room_pix = np.argwhere(prim_id == 0)
    targets = list(cfg.target_supports)

    tries = 0
    while len(regions) < cfg.regions_per_condition and tries < cfg.max_tries:
        tries += 1
        target = targets[len(regions) % len(targets)]
        pool = obj_pix if (rng.uniform() < cfg.object_prob and len(obj_pix)) else room_pix
        if len(pool) == 0:
            pool = obj_pix
        v, r_, c_ = pool[rng.integers(len(pool))]
        center = world_points[v, r_, c_].astype(np.float32)
        radius = float(rng.uniform(*cfg.region_radius))
        if any(np.linalg.norm(center - g.center) < radius + g.radius + cfg.min_region_gap for g in regions):
            continue
        fp = region_footprints(world_points, center, radius)
        counts = fp.reshape(S, -1).sum(1)
        if counts.max() > cfg.max_image_frac * H * W:
            continue
        visible = [int(j) for j in np.where(counts >= cfg.min_region_pixels)[0]]
        if len(visible) < max(cfg.min_visible_views, target + 1):
            continue
        # views that see the region only marginally are masked as well (no partial leaks)
        touching = [int(j) for j in np.where(counts > 0)[0]]
        # kept views are drawn proportionally to how much of the region they see, so the
        # realized support of region pixels actually reaches the target
        if target > 0:
            w = counts[visible].astype(np.float64)
            kept = sorted(int(j) for j in rng.choice(visible, size=target, replace=False, p=w / w.sum()))
        else:
            kept = []
        masked = [j for j in touching if j not in kept]
        # masks of this region must not cover the footprint of a kept view of an existing region
        new_masks = np.zeros_like(masks)
        for j in masked:
            new_masks[j] = footprint_to_mask(fp[j], cfg.mask_shape, cfg.mask_dilation)
        conflict = False
        for g in regions:
            gfp = region_map == g.region_id
            for j in g.kept_views:
                if (new_masks[j] & gfp[j]).any():
                    conflict = True
        if conflict:
            continue
        # and the new region's kept views must not already be masked by older regions
        if any((masks[j] & fp[j]).any() for j in kept):
            continue
        rid = len(regions)
        masks |= new_masks
        region_map[fp & (region_map < 0)] = rid
        regions.append(
            OcclusionRegion(
                region_id=rid,
                center=center,
                radius=radius,
                target_support=target,
                visible_views=visible,
                kept_views=kept,
                masked_views=masked,
                footprint_pixels=[int(x) for x in counts],
            )
        )
    return regions, masks, region_map


def apply_fill(images: np.ndarray, masks: np.ndarray, fill: str, rng: np.random.Generator) -> np.ndarray:
    """Hide masked pixels. images: (S, H, W, 3) uint8."""
    out = images.copy()
    if not masks.any():
        return out
    if fill == "gray":
        out[masks] = 127
    elif fill == "black":
        out[masks] = 0
    elif fill == "mean":
        for j in range(out.shape[0]):
            out[j][masks[j]] = images[j].reshape(-1, 3).mean(0).astype(np.uint8)
    elif fill == "noise":
        out[masks] = rng.integers(0, 256, size=(int(masks.sum()), 3), dtype=np.uint8)
    else:
        raise ValueError(f"unknown fill {fill!r}")
    return out
