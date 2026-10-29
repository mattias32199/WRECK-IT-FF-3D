"""Image preprocessing that matches VGGT exactly AND remembers what it did.

VGGT's own `load_and_preprocess_images(mode="crop")` resizes every image to
width 518, rounds the height to a multiple of 14, and center-crops the height
if it exceeds 518. It returns only the pixels, so any ground truth (depth maps,
intrinsics) would silently drift out of alignment.

Here the same steps are recorded in an `ImageTransform`, which is then applied
to ground truth so that everything downstream lives in one pixel grid: the one
VGGT sees. `tests/test_preprocess.py` checks the pixels are bit-identical to
VGGT's function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

TARGET = 518  # VGGT's training resolution (width)
PATCH = 14    # ViT patch size; H and W must be multiples of this


@dataclass(frozen=True)
class ImageTransform:
    """Original image -> VGGT input: resize to (new_w, new_h), then crop rows [top, top + out_h)."""

    orig_w: int
    orig_h: int
    new_w: int
    new_h: int
    top: int     # rows removed from the top by the center crop (0 if no crop)
    out_h: int   # final height (== new_h unless cropped)

    @property
    def out_w(self) -> int:
        return self.new_w

    @property
    def sx(self) -> float:
        return self.new_w / self.orig_w

    @property
    def sy(self) -> float:
        return self.new_h / self.orig_h

    @classmethod
    def for_size(cls, orig_w: int, orig_h: int) -> "ImageTransform":
        new_w = TARGET
        new_h = round(orig_h * (new_w / orig_w) / PATCH) * PATCH
        top = (new_h - TARGET) // 2 if new_h > TARGET else 0
        out_h = min(new_h, TARGET)
        return cls(orig_w, orig_h, new_w, new_h, top, out_h)

    # ---- applying the transform to ground truth -------------------------

    def apply_to_K(self, K: np.ndarray) -> np.ndarray:
        """Adjust a 3x3 (or ...x3x3) intrinsic matrix.

        Convention: integer pixel indices are pixel centers (OpenCV). A resize by
        s maps index u -> (u + 0.5) * s - 0.5, so the principal point gets the same
        half-pixel shift; the crop then subtracts `top` from cy.
        """
        K = np.array(K, dtype=np.float64, copy=True)
        K[..., 0, 0] *= self.sx
        K[..., 1, 1] *= self.sy
        K[..., 0, 1] *= self.sx  # skew (normally 0)
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * self.sx - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * self.sy - 0.5 - self.top
        return K

    def apply_to_map(self, m: np.ndarray) -> np.ndarray:
        """Resize + crop a per-pixel map (depth, validity mask, ...) of shape (H, W).

        Uses nearest-neighbour sampling so depth values are never blended across
        object boundaries (bilinear would invent depths that exist nowhere).
        """
        assert m.shape == (self.orig_h, self.orig_w), f"map {m.shape} != image {(self.orig_h, self.orig_w)}"
        dtype = m.dtype
        t = torch.from_numpy(np.ascontiguousarray(m).astype(np.float32))[None, None]
        t = F.interpolate(t, size=(self.new_h, self.new_w), mode="nearest-exact")
        out = t[0, 0, self.top : self.top + self.out_h].numpy()
        return out.astype(dtype)


def load_image(path: str) -> Image.Image:
    """Open an image the way VGGT does: blend alpha onto white, convert to RGB."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def preprocess_image(img: Image.Image) -> tuple[torch.Tensor, ImageTransform]:
    """PIL image -> (3, H, W) float tensor in [0, 1], plus the transform used."""
    tf = ImageTransform.for_size(*img.size)
    x = TF.ToTensor()(img.resize((tf.new_w, tf.new_h), Image.Resampling.BICUBIC))
    x = x[:, tf.top : tf.top + tf.out_h, :]
    return x, tf
