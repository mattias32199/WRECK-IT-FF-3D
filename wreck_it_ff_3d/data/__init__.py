"""Inputs: Scene (views + optional GT) and VGGT-matched preprocessing. Dataset loaders go here."""

from wreck_it_ff_3d.data.preprocess import ImageTransform
from wreck_it_ff_3d.data.scene import Scene

__all__ = ["Scene", "ImageTransform"]
