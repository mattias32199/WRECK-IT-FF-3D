"""Shared fixtures: a small rendered scene and fake VGGT models."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from confprobe.config import load_config  # noqa: E402
from confprobe.dataset import ConditionConfig, build_conditions, generate_scene  # noqa: E402
from confprobe.geometry import extri_intri_to_pose_encoding  # noqa: E402
from confprobe.metrics import relative_extrinsics, world_to_cam0  # noqa: E402
from confprobe.occlusion import OcclusionConfig  # noqa: E402
from confprobe.scene import CameraConfig, SceneConfig  # noqa: E402

SMALL_H, SMALL_W = 112, 154  # multiples of 14


def small_overrides(tmp: Path) -> dict:
    return {
        "n_scenes": 2,
        "camera.n_views": 4,
        "camera.height": SMALL_H,
        "camera.width": SMALL_W,
        "render.ssaa": 1,
        "conditions.n_masked": 1,
        "conditions.view_subsets": [2],
        "conditions.occlusion.regions_per_condition": 3,
        "conditions.occlusion.target_supports": [0, 1, 2],
        "conditions.occlusion.min_visible_views": 3,
        "conditions.occlusion.min_region_pixels": 20,
        "conditions.occlusion.max_image_frac": 0.3,
        "eval.pixel_stride": 2,
        "output.data_dir": str(tmp / "data"),
        "output.results_dir": str(tmp / "results"),
    }


@pytest.fixture(scope="session")
def small_cam_cfg() -> CameraConfig:
    return CameraConfig(n_views=4, height=SMALL_H, width=SMALL_W)


@pytest.fixture(scope="session")
def small_sample(small_cam_cfg):
    return generate_scene(3, SceneConfig(), small_cam_cfg, ssaa=1)


@pytest.fixture(scope="session")
def small_cond_cfg() -> ConditionConfig:
    occ = OcclusionConfig(
        regions_per_condition=3, target_supports=(0, 1, 2), min_visible_views=3,
        min_region_pixels=20, max_image_frac=0.3,
    )
    return ConditionConfig(n_masked=1, view_subsets=(2,), occlusion=occ)


@pytest.fixture(scope="session")
def small_conditions(small_sample, small_cond_cfg):
    return build_conditions(small_sample, small_cond_cfg, seed=3)


@pytest.fixture()
def small_cfg(tmp_path):
    return load_config(None, small_overrides(tmp_path))


# --------------------------------------------------------------------------- #
# Fake models with VGGT's output contract
# --------------------------------------------------------------------------- #
class ConstantVGGT(torch.nn.Module):
    """Outputs fixed, known predictions regardless of the input (shape contract only)."""

    def __init__(self, depth_value: float = 2.0, extrinsics: torch.Tensor | None = None):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.depth_value = depth_value
        self.extrinsics = extrinsics
        self.last_input = None

    def forward(self, images):
        self.last_input = images
        if images.ndim == 4:
            images = images[None]
        B, S, _, H, W = images.shape
        dev = images.device
        E = self.extrinsics if self.extrinsics is not None else torch.eye(3, 4).expand(S, 3, 4)
        K = torch.tensor([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]]).expand(S, 3, 3)
        pe = extri_intri_to_pose_encoding(E.float(), K, (H, W))[None].to(dev)
        return {
            "pose_enc": pe.expand(B, S, 9),
            "depth": torch.full((B, S, H, W, 1), self.depth_value, device=dev),
            "depth_conf": 1 + images.mean(2),  # depends on input so tests can check plumbing
            "world_points": torch.zeros(B, S, H, W, 3, device=dev),
            "world_points_conf": torch.ones(B, S, H, W, device=dev),
            "images": images,
        }


def image_key(images_uint8: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(images_uint8).tobytes()).hexdigest()


class OracleVGGT(torch.nn.Module):
    """Returns GT geometry (scaled) with noise that grows as support shrinks, and a
    confidence that tracks support. Registered conditions are looked up by image hash.

    Used to check that the metrics/pipeline report the *expected direction* of effects.
    """

    def __init__(self, scale: float = 0.37, seed: int = 0):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.scale = scale
        self.registry: dict[str, tuple] = {}
        self.rng = np.random.default_rng(seed)

    def register(self, cond, gt):
        self.registry[image_key(cond.images)] = (cond, gt)

    def forward(self, images):
        x = images if images.ndim == 4 else images[0]
        u8 = (x.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        cond, gt = self.registry[image_key(u8)]
        S, H, W = gt["depth"].shape
        sup = cond.support.astype(np.float32)
        sigma = 0.01 + 0.25 / (1.0 + sup) ** 2  # support 0 -> 0.26, 1 -> 0.07, 2 -> 0.04
        noise = self.rng.normal(size=(S, H, W)).astype(np.float32) * sigma
        depth = gt["depth"] * self.scale * (1 + noise)
        conf = 1 + np.exp(2.0 - 8.0 * sigma + 0.05 * self.rng.normal(size=sigma.shape)).astype(np.float32)
        X0 = world_to_cam0(gt["world_points"], gt["extrinsics"][0]) * self.scale
        X0 = X0 * (1 + noise[..., None])
        E_rel = relative_extrinsics(gt["extrinsics"]).copy()
        E_rel[:, :, 3] *= self.scale
        pe = extri_intri_to_pose_encoding(torch.from_numpy(E_rel), torch.from_numpy(gt["intrinsics"]), (H, W))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None]  # noqa: E731
        return {
            "pose_enc": pe.float()[None],
            "depth": t(depth)[..., None],
            "depth_conf": t(conf),
            "world_points": t(X0),
            "world_points_conf": t(conf),
        }
