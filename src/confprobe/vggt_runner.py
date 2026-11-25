"""Thin, testable wrapper around VGGT inference.

    runner = VGGTRunner.from_pretrained(device="cuda")
    pred = runner.run(images_uint8)          # (S, H, W, 3), H and W multiples of 14
    pred.depth, pred.depth_conf, pred.world_points, pred.extrinsic, ...

Images are fed at their native resolution (no resize/crop), so predictions are
pixel-aligned with the ground truth. The synthetic renderer produces 518-wide,
14-divisible images for exactly this reason.

Any ``torch.nn.Module`` with VGGT's output contract can be injected through
``VGGTRunner(model=...)``, which is how the tests run without weights.
"""
from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from . import geometry

PATCH_SIZE = 14
DEFAULT_WEIGHTS = "facebook/VGGT-1B"
WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass
class VGGTPrediction:
    depth: np.ndarray  # (S, H, W)  z-depth in VGGT's normalized units
    depth_conf: np.ndarray  # (S, H, W) >= 1 (VGGT uses 1 + exp(x))
    world_points: np.ndarray  # (S, H, W, 3) in the frame of view 0
    world_points_conf: np.ndarray  # (S, H, W)
    extrinsic: np.ndarray  # (S, 3, 4) world(=cam0)->cam, OpenCV
    intrinsic: np.ndarray  # (S, 3, 3)
    pose_enc: np.ndarray  # (S, 9)

    def save(self, path: str | Path, half: bool = True) -> None:
        arrs = {f.name: getattr(self, f.name) for f in fields(self)}
        if half:
            for k in ("depth", "depth_conf", "world_points", "world_points_conf"):
                arrs[k] = arrs[k].astype(np.float16)
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path: str | Path) -> "VGGTPrediction":
        z = np.load(path)
        return VGGTPrediction(**{f.name: z[f.name].astype(np.float32) for f in fields(VGGTPrediction)})

    def depth_points(self) -> np.ndarray:
        """World points obtained by unprojecting the depth head with the predicted cameras
        (VGGT reports this is usually more accurate than the point head)."""
        out = [
            geometry.np_unproject_depth(self.depth[i], self.intrinsic[i], self.extrinsic[i])
            for i in range(self.depth.shape[0])
        ]
        return np.stack(out)


def _pose_decoder():
    try:  # use the official implementation when vggt is installed
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        return pose_encoding_to_extri_intri
    except Exception:  # pragma: no cover - exercised when vggt is absent
        return geometry.pose_encoding_to_extri_intri


def default_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


class VGGTRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device | None = None,
        amp_dtype: torch.dtype | None | str = "auto",
        strict_size: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.amp_dtype = default_amp_dtype(self.device) if amp_dtype == "auto" else amp_dtype
        self.strict_size = strict_size
        self._decode_pose = _pose_decoder()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_WEIGHTS,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        **kw,
    ) -> "VGGTRunner":
        """Load VGGT-1B.

        * ``checkpoint``: a local ``model.pt`` state dict (use on clusters without internet).
        * otherwise ``weights`` is passed to ``VGGT.from_pretrained`` (HF hub id or local dir).
        """
        from vggt.models.vggt import VGGT  # imported lazily so tests run without vggt

        if checkpoint is not None:
            model = VGGT()
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
        else:
            try:
                model = VGGT.from_pretrained(weights)
            except Exception as e:  # older vggt versions without the HF mixin
                warnings.warn(f"VGGT.from_pretrained failed ({e}); falling back to torch.hub download")
                model = VGGT()
                model.load_state_dict(torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu"))
        return cls(model, device=device, **kw)

    # ------------------------------------------------------------------ #
    def preprocess(self, images) -> torch.Tensor:
        """uint8 (S, H, W, 3) array / float (S, 3, H, W) tensor in [0,1] -> float tensor on device."""
        if isinstance(images, np.ndarray):
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"expected (S, H, W, 3) array, got {images.shape}")
            if images.dtype == np.uint8:
                t = torch.from_numpy(images).float() / 255.0
            else:
                t = torch.from_numpy(images.astype(np.float32))
            t = t.permute(0, 3, 1, 2)
        elif isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"expected (S, 3, H, W) tensor, got {tuple(images.shape)}")
            t = images.float()
            if images.dtype == torch.uint8:
                t = t / 255.0
        else:
            raise TypeError(type(images))
        if t.min() < 0 or t.max() > 1:
            raise ValueError("image values must be in [0, 1]")
        H, W = t.shape[-2:]
        if H % PATCH_SIZE or W % PATCH_SIZE:
            raise ValueError(f"H and W must be multiples of {PATCH_SIZE}, got {(H, W)}")
        if self.strict_size and max(H, W) != 518:
            warnings.warn(f"VGGT was trained with a 518-px long side; got {(H, W)}")
        return t.contiguous().to(self.device)

    def _is_preprocessed(self, x) -> bool:
        return (
            isinstance(x, torch.Tensor)
            and x.dtype == torch.float32
            and x.device == self.device
            and x.ndim == 4
            and x.shape[1] == 3
        )

    @torch.no_grad()
    def forward_raw(self, images) -> dict:
        """Raw VGGT output dict (with batch dim). Accepts anything `preprocess` accepts."""
        x = images if self._is_preprocessed(images) else self.preprocess(images)
        ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with ctx:
            out = self.model(x)
        return out

    @torch.no_grad()
    def run(self, images) -> VGGTPrediction:
        x = self.preprocess(images)
        out = self.forward_raw(x)
        S, _, H, W = x.shape

        def take(key, expected_tail):
            v = out[key].float()
            if v.shape[0] == 1 and v.ndim == len(expected_tail) + 2:
                v = v[0]  # drop batch dim
            if tuple(v.shape[1:]) != expected_tail or v.shape[0] != S:
                raise RuntimeError(f"unexpected shape for {key}: {tuple(v.shape)}")
            return v

        depth = take("depth", (H, W, 1))[..., 0]
        depth_conf = take("depth_conf", (H, W))
        wp = take("world_points", (H, W, 3))
        wp_conf = take("world_points_conf", (H, W))
        pose_enc = take("pose_enc", (9,))
        extr, intr = self._decode_pose(pose_enc[None], (H, W))
        return VGGTPrediction(
            depth=depth.cpu().numpy(),
            depth_conf=depth_conf.cpu().numpy(),
            world_points=wp.cpu().numpy(),
            world_points_conf=wp_conf.cpu().numpy(),
            extrinsic=extr[0].float().cpu().numpy(),
            intrinsic=intr[0].float().cpu().numpy(),
            pose_enc=pose_enc.cpu().numpy(),
        )
