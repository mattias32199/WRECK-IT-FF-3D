"""Hooks: read what is inside VGGT during a run (tokens, attention).  [not implemented yet]

Facts about the current VGGT code that shape this layer (checked against
facebookresearch/vggt @ a288dd0):

  * `model.aggregator.frame_blocks[i]` and `.global_blocks[i]`, i = 0..23,
    alternate frame -> global. Frame blocks see tokens as (B*S, P, C), global
    blocks as (B, S*P, C), with C = 1024.
  * Each view has P = 5 + (H/14)(W/14) tokens: index 0 is the camera token,
    1-4 are registers, patches start at `aggregator.patch_start_idx` (= 5).
  * The aggregator only *keeps* layers {4, 11, 17, 23} (the ones the DPT
    heads use); the rest of its returned list is None. So reading all layers
    needs forward hooks on the blocks, not the return value.
  * Attention uses fused SDPA, so attention weights never exist. Capturing
    them means recomputing softmax(QK^T) for chosen blocks/heads/queries
    (the full global matrix is ~8 GB per layer at 8 views).

Planned API (context managers around `runner.model`, see models/vggt.py):

    with TokenRecorder(model, layers=[0, 6, 12, 18, 23], kind="global") as rec:
        pred = runner.run(scene)
    rec.tokens[12]   # (S, P, 1024) on CPU

    with AttentionRecorder(model, block=("global", 12), heads=[3, 7], queries=idx) as att:
        ...
"""

from __future__ import annotations


class TokenRecorder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("analysis/hooks.py")
