"""Dataclass for model-agnostic prediction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lib.geometry import unproject_depth


@dataclass
class Prediction:
    extrinsics: np.ndarray         # (S, 3, 4) camera-from-world, OpenCV
    intrinsics: np.ndarray         # (S, 3, 3)
    depth: np.ndarray              # (S, H, W)
    depth_conf: np.ndarray         # (S, H, W), >= 1 (activation is 1 + exp(x))
    world_points: np.ndarray       # (S, H, W, 3) from the point-map head
    world_points_conf: np.ndarray  # (S, H, W)
    images: np.ndarray             # (S, H, W, 3) uint8, what the model saw
    meta: dict = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        return self.depth.shape[0]

    def points_from_depth(self) -> np.ndarray:
        """World points by unprojecting predicted depth with predicted cameras.

        The VGGT paper reports this is usually more accurate than the point-map
        head, so it is worth comparing both.
        """
        return unproject_depth(self.depth, self.intrinsics, self.extrinsics)

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in asdict(self).items() if k != "meta"}
        np.savez_compressed(path, **arrays, meta=np.array(json.dumps(self.meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Prediction":
        with np.load(Path(path).with_suffix(".npz")) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        return cls(**arrays, meta=meta)
