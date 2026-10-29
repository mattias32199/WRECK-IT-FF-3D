"""Quick-look visualisation: image / depth / confidence panels and .ply export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from lib.models.prediction import Prediction


def save_panels(pred: Prediction, path: str | Path, max_views: int = 6) -> Path:
    """One row per view: input image, predicted depth, depth confidence (log scale)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(pred.num_views, max_views)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n), squeeze=False)
    for i in range(n):
        panels = [
            (pred.images[i], None, f"view {i}"),
            (pred.depth[i], "turbo", "depth"),
            (np.log(pred.depth_conf[i]), "viridis", "log depth_conf"),
        ]
        for ax, (img, cmap, title) in zip(axes[i], panels):
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap:
                fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def conf_mask(conf: np.ndarray, keep_top: float = 0.5) -> np.ndarray:
    """Keep the `keep_top` fraction of pixels with the highest confidence (scale-free)."""
    return conf >= np.quantile(conf, 1.0 - keep_top)


def save_ply(points: np.ndarray, colors: np.ndarray, path: str | Path, mask: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian .ply. points (..., 3) float, colors (..., 3) uint8, mask (...) bool."""
    pts = points.reshape(-1, 3).astype("<f4")
    cols = colors.reshape(-1, 3).astype(np.uint8)
    if mask is not None:
        m = mask.reshape(-1)
        pts, cols = pts[m], cols[m]
    finite = np.isfinite(pts).all(1)
    pts, cols = pts[finite], cols[finite]

    vertex = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts.T
    vertex["red"], vertex["green"], vertex["blue"] = cols.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
    return path
