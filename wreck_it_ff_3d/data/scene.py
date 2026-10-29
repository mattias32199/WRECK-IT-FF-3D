"""`Scene`: the input side of every experiment.

A Scene is a set of views already in VGGT's pixel grid, optionally with ground
truth in the *same* grid. Dataset loaders (folder now, DTU/ETH3D later) only
read raw files and call `Scene.build`, which does all the resizing/cropping,
so no other code ever has to think about resolution.

Conventions (identical to VGGT's outputs, so GT and predictions compare directly):
    images      (S, 3, H, W) float32 in [0, 1]
    depth       (S, H, W)    float32, z-depth along the optical axis; 0 = unknown
    valid       (S, H, W)    bool, where GT depth is trustworthy
    intrinsics  (S, 3, 3)    pixel units, pixel centers at integer coordinates
    extrinsics  (S, 3, 4)    [R | t], camera-from-world, OpenCV (x right, y down, z forward)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from lib.data.preprocess import ImageTransform, load_image, preprocess_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Scene:
    name: str
    images: torch.Tensor
    image_paths: list[str]
    transforms: list[ImageTransform] = field(repr=False)
    depth: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None

    @property
    def num_views(self) -> int:
        return self.images.shape[0]

    @property
    def hw(self) -> tuple[int, int]:
        return tuple(self.images.shape[-2:])

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_cameras(self) -> bool:
        return self.intrinsics is not None and self.extrinsics is not None

    def subset(self, idx: Sequence[int]) -> "Scene":
        """Select views (order matters: VGGT treats view idx[0] as the reference/world frame)."""
        idx = list(idx)
        pick = lambda a: None if a is None else a[idx]
        return Scene(
            name=self.name,
            images=self.images[idx],
            image_paths=[self.image_paths[i] for i in idx],
            transforms=[self.transforms[i] for i in idx],
            depth=pick(self.depth),
            valid=pick(self.valid),
            intrinsics=pick(self.intrinsics),
            extrinsics=pick(self.extrinsics),
        )

    # ---- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        name: str,
        image_paths: Sequence[str],
        depths: Optional[Sequence[np.ndarray]] = None,
        intrinsics: Optional[Sequence[np.ndarray]] = None,
        extrinsics: Optional[Sequence[np.ndarray]] = None,
        valid: Optional[Sequence[np.ndarray]] = None,
    ) -> "Scene":
        """Preprocess images and map raw (original-resolution) GT into VGGT's grid.

        Extrinsics need no change: resizing/cropping an image does not move the camera.
        """
        if len(image_paths) == 0:
            raise ValueError("A scene needs at least one image")

        imgs, tfs = zip(*(preprocess_image(load_image(p)) for p in image_paths))
        shapes = {tuple(x.shape) for x in imgs}
        if len(shapes) > 1:
            # VGGT would pad mismatched images with white; we refuse instead, because
            # padding would also have to be tracked in the GT.
            raise ValueError(f"All views in a scene must share a size after preprocessing, got {shapes}")

        d = v = K = E = None
        if depths is not None:
            d = np.stack([tf.apply_to_map(np.asarray(x, np.float32)) for tf, x in zip(tfs, depths)])
            if valid is not None:
                v = np.stack([tf.apply_to_map(np.asarray(x, bool)) for tf, x in zip(tfs, valid)])
            else:
                v = np.isfinite(d) & (d > 0)
            d = np.where(v, d, 0.0).astype(np.float32)
        if intrinsics is not None:
            K = np.stack([tf.apply_to_K(k) for tf, k in zip(tfs, intrinsics)]).astype(np.float32)
        if extrinsics is not None:
            E = np.stack([np.asarray(e, np.float64)[:3, :4] for e in extrinsics]).astype(np.float32)

        return cls(name, torch.stack(imgs), [str(p) for p in image_paths], list(tfs), d, v, K, E)

    @classmethod
    def from_folder(cls, folder: str | Path, max_views: Optional[int] = None, stride: int = 1) -> "Scene":
        """Images only (no GT), sorted by filename. Good for smoke tests and campus captures."""
        folder = Path(folder)
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)[::stride]
        if max_views is not None:
            paths = paths[:max_views]
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        return cls.build(folder.name, paths)
