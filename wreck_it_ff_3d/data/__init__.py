"""Inputs: Scene (views + optional GT) and VGGT-matched preprocessing. Dataset loaders go here."""

from lib.data.preprocess import ImageTransform
from lib.data.scene import Scene

__all__ = ["Scene", "ImageTransform"]
