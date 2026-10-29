"""Metrics: compare a Prediction to a Scene's ground truth.  [next, not implemented yet]

Planned contents (each small and unit-tested):

    align_depth(pred_depth, gt_depth, valid) -> scale
        Per-scene median ratio. VGGT's scale is arbitrary, so every depth
        metric is computed after this.

    align_cameras(pred_E, gt_E) -> Sim3
        Umeyama on camera centers. Both trajectories must be expressed relative
        to the same reference view first (VGGT's world frame = view 0).

    depth_metrics(pred, gt, valid) -> {"abs_rel", "delta_1.25"}
    pose_metrics(pred_E, gt_E)     -> {"RRA@15", "RTA@15", "AUC@30"}   (pairwise, as in the VGGT paper)
    ranking_metrics(conf, err)     -> {"spearman", "auroc"}            (Exp 1)

First milestone ("done when"): DTU numbers land near those in the VGGT paper.
"""

from __future__ import annotations

from lib.models.prediction import Prediction
from lib.data.scene import Scene


def evaluate(pred: Prediction, scene: Scene) -> dict:
    raise NotImplementedError("eval/metrics.py: build after models/vggt.py is verified on your GPU.")
