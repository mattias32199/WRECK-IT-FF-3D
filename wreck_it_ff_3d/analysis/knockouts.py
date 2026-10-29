"""Knockouts: switch off attention blocks or heads mid-forward-pass.  [not implemented yet]

Planned knockouts (all context managers that restore the model on exit):

  CrossViewOff(model, blocks)
      Run global block i on tokens reshaped (B, S*P, C) -> (B*S, P, C), so each
      token only attends within its own view, with the block's own weights.
      Everything else (MLP, layer scale, residual) is unchanged.

  WithinFrameOff(model, blocks)
      Replace frame block i with the identity (only the residual path remains).

  HeadAblation(model, block=("global", i), heads=[...])
      Zero the chosen heads' outputs before the attention output projection.

Required sanity test before trusting any result: an intervention that touches
no blocks must reproduce the baseline outputs exactly.
"""

from __future__ import annotations


class CrossViewOff:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("analysis/knockouts.py")
