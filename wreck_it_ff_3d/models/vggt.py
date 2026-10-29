"""VGGTRunner loads and runs VGGT. Scene in, Prediction out.

    Input: scene
    Output: prediction
"""

from __future__ import annotations

import contextlib
import time
from typing import Optional

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from lib.models.prediction import Prediction
from lib.data.scene import Scene

DEFAULT_MODEL = "facebook/VGGT-1B"


def pick_dtype(device: torch.device) -> torch.dtype:
    """bf16 on Ampere+ (A4500, H100); fp16 on older GPUs (Titan V has no bf16); fp32 on CPU."""
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


class VGGTRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        model_id:   Hugging Face repo, downloaded on first use.
        checkpoint: path to a local `model.pt` instead (for offline compute nodes, e.g. Bridges-2).
        dtype:      autocast dtype for the transformer; heads always run in fp32 (VGGT does this itself).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or pick_dtype(self.device)
        if checkpoint:
            self.model = VGGT()
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self.model_name = checkpoint
        else:
            self.model = VGGT.from_pretrained(model_id)
            self.model_name = model_id
        self.model = self.model.to(self.device).eval()

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    @torch.no_grad()
    def forward_raw(self, images: torch.Tensor) -> dict:
        """(S, 3, H, W) in [0, 1] -> VGGT's raw output dict (batch dim of 1, on device)."""
        with self._autocast():
            return self.model(images.to(self.device)[None])

    def run(self, scene: Scene) -> Prediction:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        out = self.forward_raw(scene.images)
        extr, intr = pose_encoding_to_extri_intri(out["pose_enc"], scene.hw)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0

        np_ = lambda t: t[0].float().cpu().numpy()  # drop batch dim
        images = (scene.images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        meta = {
            "scene": scene.name,
            "image_paths": scene.image_paths,
            "model": self.model_name,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "seconds": round(seconds, 3),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2)
            if self.device.type == "cuda" else None,
        }
        return Prediction(
            extrinsics=np_(extr),
            intrinsics=np_(intr),
            depth=np_(out["depth"])[..., 0],
            depth_conf=np_(out["depth_conf"]),
            world_points=np_(out["world_points"]),
            world_points_conf=np_(out["world_points_conf"]),
            images=images,
            meta=meta,
        )
