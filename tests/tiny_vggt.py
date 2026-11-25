"""A randomly initialized, tiny VGGT built from the official vggt modules.

Same forward/output contract as VGGT-1B, small enough for CPU unit tests.
Returns None if the `vggt` package is not installed.
"""
from __future__ import annotations

import inspect

import torch


def build_tiny_vggt(embed_dim: int = 64, depth: int = 4, seed: int = 0):
    try:
        from vggt.heads.camera_head import CameraHead
        from vggt.heads.dpt_head import DPTHead
        from vggt.models.aggregator import Aggregator
        from vggt.models.vggt import VGGT
    except Exception:
        return None
    torch.manual_seed(seed)
    model = VGGT.__new__(VGGT)
    torch.nn.Module.__init__(model)
    agg_kw = dict(embed_dim=embed_dim, depth=depth, num_heads=4, patch_embed="conv")
    layers = list(range(depth))
    if "cached_layer_indices" in inspect.signature(Aggregator.__init__).parameters:
        agg_kw["cached_layer_indices"] = tuple(layers)
    model.aggregator = Aggregator(**agg_kw)
    dpt_kw = dict(features=32, out_channels=[16, 32, 64, 64], intermediate_layer_idx=layers)
    model.camera_head = CameraHead(dim_in=2 * embed_dim, trunk_depth=1, num_heads=4)
    model.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", **dpt_kw)
    model.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", **dpt_kw)
    model.track_head = None
    return model.eval()
