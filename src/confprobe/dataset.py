"""Synthetic dataset: scenes, rendered views, and experimental *conditions*.

Layout on disk (one directory per scene):

    <root>/<scene_id>/
        scene.json            scene + camera parameters (enough to re-render)
        render.npz            rgb, depth, normals, prim_id, face_id, K, E
        conditions/<name>.npz images (masked), masks, support maps, region map
        conditions/<name>.json condition metadata (view indices, regions)

A *condition* is a choice of input views plus an occlusion pattern:
  * ``clean``          all views, nothing masked (natural support only)
  * ``masked_<k>``     all views, occlusion regions with controlled support
  * ``subset<n>_<k>``  n of the views, nothing masked (support via view selection)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .occlusion import OcclusionConfig, OcclusionRegion, apply_fill, sample_regions
from .renderer import RenderOutput, render
from .scene import CameraConfig, Cameras, Scene, SceneConfig, random_cameras, random_scene
from .support import compute_support


@dataclass
class ConditionConfig:
    include_clean: bool = True
    n_masked: int = 2
    view_subsets: tuple[int, ...] = ()  # e.g. (2, 4): unmasked runs on random view subsets
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    support_rel_tol: float = 0.01
    support_abs_tol: float = 0.005
    support_min_cos: float = 0.1  # grazing-angle cutoff for visibility (needs GT normals)


@dataclass
class SceneSample:
    scene_id: str
    scene: Scene
    cams: Cameras
    render: RenderOutput

    @property
    def world_points(self) -> np.ndarray:
        if not hasattr(self, "_wp"):
            self._wp = self.render.world_points(self.cams)
        return self._wp

    def save(self, root: str | Path) -> Path:
        d = Path(root) / self.scene_id
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "scene_id": self.scene_id,
            "scene": self.scene.to_dict(),
            "height": self.cams.height,
            "width": self.cams.width,
        }
        (d / "scene.json").write_text(json.dumps(meta))
        np.savez_compressed(
            d / "render.npz",
            rgb=self.render.rgb,
            depth=self.render.depth,
            normals=self.render.normals.astype(np.float16),
            prim_id=self.render.prim_id,
            face_id=self.render.face_id,
            intrinsics=self.cams.intrinsics,
            extrinsics=self.cams.extrinsics,
        )
        return d

    @staticmethod
    def load(root: str | Path, scene_id: str) -> "SceneSample":
        d = Path(root) / scene_id
        meta = json.loads((d / "scene.json").read_text())
        z = np.load(d / "render.npz")
        cams = Cameras(z["extrinsics"], z["intrinsics"], int(meta["height"]), int(meta["width"]))
        r = RenderOutput(
            rgb=z["rgb"],
            depth=z["depth"],
            normals=z["normals"].astype(np.float32),
            prim_id=z["prim_id"],
            face_id=z["face_id"],
        )
        return SceneSample(scene_id, Scene.from_dict(meta["scene"]), cams, r)

    @staticmethod
    def exists(root: str | Path, scene_id: str) -> bool:
        d = Path(root) / scene_id
        return (d / "scene.json").exists() and (d / "render.npz").exists()


@dataclass
class ConditionSample:
    name: str
    scene_id: str
    view_indices: list[int]  # indices into the scene's views; first = VGGT reference frame
    images: np.ndarray  # (S', H, W, 3) uint8, what the model sees
    masks: np.ndarray  # (S', H, W) bool
    support: np.ndarray  # (S', H, W) uint8
    n_visible: np.ndarray  # (S', H, W) uint8
    support_bits: np.ndarray  # (S', H, W) int64
    region_map: np.ndarray  # (S', H, W) int16
    regions: list[dict]
    fill: str = "none"

    @property
    def kind(self) -> str:
        return self.name.split("_")[0].rstrip("0123456789") or self.name

    def self_visible_map(self) -> np.ndarray:
        """True where the pixel is not masked in its own view."""
        return ~self.masks

    @property
    def target_support_map(self) -> np.ndarray:
        out = np.full(self.region_map.shape, -1, dtype=np.int8)
        for r in self.regions:
            out[self.region_map == r["region_id"]] = r["target_support"]
        return out

    def gt(self, sample: SceneSample) -> dict:
        idx = self.view_indices
        return {
            "depth": sample.render.depth[idx],
            "intrinsics": sample.cams.intrinsics[idx],
            "extrinsics": sample.cams.extrinsics[idx],
            "world_points": sample.world_points[idx],
            "prim_id": sample.render.prim_id[idx],
        }

    def save(self, root: str | Path) -> None:
        d = Path(root) / self.scene_id / "conditions"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / f"{self.name}.npz",
            images=self.images,
            masks=self.masks,
            support=self.support,
            n_visible=self.n_visible,
            support_bits=self.support_bits,
            region_map=self.region_map,
        )
        meta = {"name": self.name, "scene_id": self.scene_id, "view_indices": self.view_indices,
                "regions": self.regions, "fill": self.fill}
        (d / f"{self.name}.json").write_text(json.dumps(meta))

    @staticmethod
    def load(root: str | Path, scene_id: str, name: str) -> "ConditionSample":
        d = Path(root) / scene_id / "conditions"
        meta = json.loads((d / f"{name}.json").read_text())
        z = np.load(d / f"{name}.npz")
        return ConditionSample(
            name=name, scene_id=scene_id, view_indices=meta["view_indices"],
            images=z["images"], masks=z["masks"], support=z["support"], n_visible=z["n_visible"],
            support_bits=z["support_bits"], region_map=z["region_map"], regions=meta["regions"],
            fill=meta.get("fill", "none"),
        )

    @staticmethod
    def list_names(root: str | Path, scene_id: str) -> list[str]:
        d = Path(root) / scene_id / "conditions"
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def scene_id_for(seed: int) -> str:
    return f"scene_{seed:05d}"


def generate_scene(
    seed: int,
    scene_cfg: SceneConfig | None = None,
    cam_cfg: CameraConfig | None = None,
    ssaa: int = 2,
    device: str = "cpu",
) -> SceneSample:
    scene = random_scene(seed, scene_cfg)
    cams = random_cameras(seed, scene, cam_cfg)
    out = render(scene, cams, ssaa=ssaa, device=device)
    return SceneSample(scene_id_for(seed), scene, cams, out)


def make_condition(
    sample: SceneSample,
    name: str,
    view_indices: list[int],
    occlusion: OcclusionConfig | None,
    rng: np.random.Generator,
    cfg: ConditionConfig,
    device: str = "cpu",
) -> ConditionSample:
    idx = list(view_indices)
    wp = sample.world_points[idx]
    depth = sample.render.depth[idx]
    K = sample.cams.intrinsics[idx]
    E = sample.cams.extrinsics[idx]
    S, H, W = depth.shape
    if occlusion is not None:
        regions, masks, region_map = sample_regions(wp, sample.render.prim_id[idx], sample.scene, occlusion, rng)
        fill = occlusion.fill
    else:
        regions, masks, region_map = [], np.zeros((S, H, W), bool), np.full((S, H, W), -1, np.int16)
        fill = "none"
    sup = compute_support(
        depth, K, E, masks, rel_tol=cfg.support_rel_tol, abs_tol=cfg.support_abs_tol, device=device,
        normals=sample.render.normals[idx], min_cos=cfg.support_min_cos,
    )
    images = apply_fill(sample.render.rgb[idx], masks, fill, rng) if fill != "none" else sample.render.rgb[idx].copy()
    return ConditionSample(
        name=name,
        scene_id=sample.scene_id,
        view_indices=idx,
        images=images,
        masks=masks,
        support=sup.support,
        n_visible=sup.n_visible,
        support_bits=sup.support_bits,
        region_map=region_map,
        regions=[r.to_dict() for r in regions],
        fill=fill,
    )


def build_conditions(sample: SceneSample, cfg: ConditionConfig, seed: int, device: str = "cpu") -> list[ConditionSample]:
    rng = np.random.default_rng(seed + 104729)
    S = len(sample.cams)
    all_views = list(range(S))
    conds = []
    if cfg.include_clean:
        conds.append(make_condition(sample, "clean", all_views, None, rng, cfg, device))
    for k in range(cfg.n_masked):
        c = make_condition(sample, f"masked_{k}", all_views, cfg.occlusion, rng, cfg, device)
        conds.append(c)
    for n in cfg.view_subsets:
        if n >= S:
            continue
        # keep view order (and hence the reference frame convention) but random members
        start = int(rng.integers(0, S))
        sub = sorted(rng.choice([v for v in all_views if v != start], size=n - 1, replace=False).tolist())
        conds.append(make_condition(sample, f"subset{n}_0", [start] + sub, None, rng, cfg, device))
    return conds
