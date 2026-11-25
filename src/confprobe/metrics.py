"""Errors, alignment and confidence-quality metrics.

Two stages:
  1. :func:`evaluate_condition` turns (GT, condition, prediction) into a per-pixel
     table (pandas DataFrame) plus per-view pose errors.
  2. :func:`summarize` reduces any per-pixel table (one condition or many) into
     scalar metrics: Spearman(conf, error), AUROC for high-error detection,
     AUROC for "is the model guessing" (low support), AUSE, and stats per
     support level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .dataset import ConditionSample
from .vggt_runner import VGGTPrediction


@dataclass
class EvalConfig:
    pixel_stride: int = 2  # keep every n-th pixel (rows and cols) in the stored table
    align_on: str = "supported"  # pixels used for scale alignment: "supported" | "all"
    align_min_support: int = 2
    err_threshold: float = 0.05  # abs-rel threshold for "high error"
    err_quantile: float = 0.8  # alternative: top (1-q) errors within a condition are "high"
    max_spearman_samples: int = 2_000_000


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def median_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Scale s minimizing robustly |s * pred - gt| via the median ratio."""
    p, g = pred[valid], gt[valid]
    ok = (p > 1e-8) & np.isfinite(p) & np.isfinite(g)
    if ok.sum() == 0:
        raise ValueError("no valid pixels for scale alignment")
    return float(np.median(g[ok] / p[ok]))


def relative_extrinsics(E: np.ndarray) -> np.ndarray:
    """Re-express (S, 3, 4) world->cam extrinsics in the frame of camera 0 (VGGT's convention)."""
    R0, t0 = E[0, :, :3], E[0, :, 3]
    out = np.empty_like(E)
    for i in range(E.shape[0]):
        Ri, ti = E[i, :, :3], E[i, :, 3]
        R = Ri @ R0.T
        out[i, :, :3] = R
        out[i, :, 3] = ti - R @ t0
    return out


def world_to_cam0(points: np.ndarray, E0: np.ndarray) -> np.ndarray:
    return points @ E0[:, :3].T + E0[:, 3]


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    R = np.swapaxes(Ra, -1, -2) @ Rb
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.clip((a * b).sum(-1) / np.maximum(na * nb, 1e-12), -1, 1)
    return np.degrees(np.arccos(cos))


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map values to (0, 1] by rank — makes confidence comparable across scenes."""
    flat = x.reshape(-1)
    return (rankdata(flat, method="average") / flat.size).reshape(x.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Per-condition evaluation
# --------------------------------------------------------------------------- #
def evaluate_condition(
    cond: ConditionSample,
    gt: dict,
    pred: VGGTPrediction,
    cfg: EvalConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (pixel_table, pose_table, info).

    gt: dict from ``ConditionSample.gt(sample)`` with depth, intrinsics, extrinsics,
        world_points, prim_id for the condition's views.
    """
    cfg = cfg or EvalConfig()
    gd = gt["depth"]
    S, H, W = gd.shape
    if pred.depth.shape != (S, H, W):
        raise ValueError(f"prediction shape {pred.depth.shape} != GT {(S, H, W)}")

    if cfg.align_on == "supported":
        align_mask = (cond.support >= cfg.align_min_support) & cond.self_visible_map()
        if align_mask.sum() < 100:
            align_mask = cond.self_visible_map()
    elif cfg.align_on == "all":
        align_mask = np.ones_like(gd, dtype=bool)
    else:
        raise ValueError(cfg.align_on)

    # ---- depth head --------------------------------------------------- #
    s_depth = median_scale(pred.depth, gd, align_mask)
    d_pred = pred.depth * s_depth
    abs_rel = np.abs(d_pred - gd) / gd

    # ---- point head (frame of view 0) --------------------------------- #
    E_gt = gt["extrinsics"]
    X_gt = world_to_cam0(gt["world_points"], E_gt[0])
    n_gt = np.linalg.norm(X_gt, axis=-1)
    n_pr = np.linalg.norm(pred.world_points, axis=-1)
    s_pts = median_scale(n_pr, n_gt, align_mask)
    pt_err = np.linalg.norm(pred.world_points * s_pts - X_gt, axis=-1) / gd

    # ---- poses -------------------------------------------------------- #
    E_rel = relative_extrinsics(E_gt)
    rot_err = rotation_error_deg(pred.extrinsic[:, :, :3], E_rel[:, :, :3])
    # camera centers in cam0 frame, scaled with the point-map scale
    c_gt = -np.einsum("sji,sj->si", E_rel[:, :, :3], E_rel[:, :, 3])
    c_pr = -np.einsum("sji,sj->si", pred.extrinsic[:, :, :3], pred.extrinsic[:, :, 3]) * s_pts
    trans_ang = np.where(np.arange(S) == 0, 0.0, angle_deg(c_pr, c_gt))
    trans_err = np.linalg.norm(c_pr - c_gt, axis=-1)
    pose_df = pd.DataFrame(
        {
            "scene_id": cond.scene_id,
            "condition": cond.name,
            "view": np.arange(S),
            "src_view": cond.view_indices,
            "rot_err_deg": rot_err,
            "trans_ang_err_deg": trans_ang,
            "center_err": trans_err,
            "frac_masked": cond.masks.reshape(S, -1).mean(1),
        }
    )

    # ---- confidence normalizations (on full-resolution maps) --------- #
    dconf_pct = percentile_rank(pred.depth_conf)
    pconf_pct = percentile_rank(pred.world_points_conf)

    # ---- per-pixel table (strided) ------------------------------------ #
    st = cfg.pixel_stride
    vv, yy, xx = np.meshgrid(np.arange(S), np.arange(0, H, st), np.arange(0, W, st), indexing="ij")
    sl = (slice(None), slice(0, H, st), slice(0, W, st))
    tsm = cond.target_support_map
    df = pd.DataFrame(
        {
            "scene_id": cond.scene_id,
            "condition": cond.name,
            "kind": cond.kind,
            "n_views": S,
            "view": vv.ravel().astype(np.int16),
            "y": yy.ravel().astype(np.int16),
            "x": xx.ravel().astype(np.int16),
            "gt_depth": gd[sl].ravel(),
            "pred_depth": d_pred[sl].ravel(),
            "abs_rel": abs_rel[sl].ravel(),
            "point_err": pt_err[sl].ravel(),
            "depth_conf": pred.depth_conf[sl].ravel(),
            "point_conf": pred.world_points_conf[sl].ravel(),
            "depth_conf_pct": dconf_pct[sl].ravel(),
            "point_conf_pct": pconf_pct[sl].ravel(),
            "support": cond.support[sl].ravel(),
            "n_visible": cond.n_visible[sl].ravel(),
            "self_visible": cond.self_visible_map()[sl].ravel(),
            "region_id": cond.region_map[sl].ravel(),
            "target_support": tsm[sl].ravel(),
            "is_object": (gt["prim_id"][sl] != 0).ravel(),
            "used_for_align": align_mask[sl].ravel(),
        }
    )
    info = {
        "scene_id": cond.scene_id,
        "condition": cond.name,
        "scale_depth": s_depth,
        "scale_points": s_pts,
        "n_regions": len(cond.regions),
        "frac_masked": float(cond.masks.mean()),
    }
    return df, pose_df, info


# --------------------------------------------------------------------------- #
# Confidence-quality metrics
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` for predicting boolean `labels` (Mann-Whitney U). NaN if undefined."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray, max_n: int | None = None, seed: int = 0) -> float:
    if len(a) < 3:
        return float("nan")
    if max_n is not None and len(a) > max_n:
        idx = np.random.default_rng(seed).choice(len(a), size=max_n, replace=False)
        a, b = a[idx], b[idx]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def sparsification(err: np.ndarray, conf: np.ndarray, n_steps: int = 20):
    """Sparsification curves. Removes the least-confident fraction f of pixels and
    reports the mean error of the rest; the oracle removes the highest-error pixels.

    Returns fractions, curve, oracle, AUSE (area between curve and oracle, normalized by
    the full-set mean error so it is comparable across conditions).
    """
    n = len(err)
    fr = np.linspace(0, 0.95, n_steps)
    order_conf = np.argsort(-conf, kind="stable")  # most confident first
    order_orac = np.argsort(err, kind="stable")  # lowest error first
    cs_conf = np.cumsum(err[order_conf])
    cs_orac = np.cumsum(err[order_orac])
    keep = np.maximum(1, np.round((1 - fr) * n).astype(int))
    curve = cs_conf[keep - 1] / keep
    oracle = cs_orac[keep - 1] / keep
    base = err.mean() if err.mean() > 0 else 1.0
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ause = float(trapz((curve - oracle) / base, fr))
    return fr, curve, oracle, ause


def confidence_metrics(err: np.ndarray, conf: np.ndarray, support: np.ndarray, cfg: EvalConfig) -> dict:
    out = {"n": int(len(err))}
    if len(err) == 0:
        return out
    out["spearman_conf_err"] = spearman(conf, err, cfg.max_spearman_samples)
    out["auroc_err_thresh"] = auroc(-conf, err > cfg.err_threshold)
    out["frac_high_err"] = float((err > cfg.err_threshold).mean())
    q = np.quantile(err, cfg.err_quantile)
    out["auroc_err_quantile"] = auroc(-conf, err > q)
    out["ause"] = sparsification(err, conf)[3]
    # "is the model guessing?" — does low confidence identify low-support pixels
    out["auroc_support0_vs_2plus"] = auroc(-conf[support != 1], support[support != 1] == 0)
    out["auroc_support_le1_vs_2plus"] = auroc(-conf, support <= 1)
    out["mean_err"] = float(err.mean())
    out["median_err"] = float(np.median(err))
    return out


SUPPORT_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 255)]


def support_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}+"


def per_support_table(df: pd.DataFrame, err_col: str, conf_col: str) -> pd.DataFrame:
    rows = []
    for lo, hi in SUPPORT_BINS:
        for sv in (True, False):
            sub = df[(df.support >= lo) & (df.support <= hi) & (df.self_visible == sv)]
            if len(sub) == 0:
                continue
            e, c = sub[err_col].to_numpy(), sub[conf_col].to_numpy()
            rows.append(
                {
                    "support": support_label(lo, hi),
                    "self_visible": sv,
                    "n": len(sub),
                    f"{conf_col}_mean": float(c.mean()),
                    f"{conf_col}_median": float(np.median(c)),
                    f"{err_col}_mean": float(e.mean()),
                    f"{err_col}_median": float(np.median(e)),
                    "spearman_within_bin": spearman(c, e, 500_000),
                }
            )
    return pd.DataFrame(rows)


PAIRINGS = {  # (error column, confidence columns to test)
    "depth": ("abs_rel", ["depth_conf", "depth_conf_pct"]),
    "points": ("point_err", ["point_conf", "point_conf_pct"]),
}


def summarize(df: pd.DataFrame, cfg: EvalConfig | None = None) -> dict:
    """Scalar metrics for any subset of the per-pixel table."""
    cfg = cfg or EvalConfig()
    out: dict = {"n_pixels": int(len(df))}
    sup = df.support.to_numpy()
    for head, (err_col, conf_cols) in PAIRINGS.items():
        err = df[err_col].to_numpy()
        for cc in conf_cols:
            out[f"{head}/{cc}"] = confidence_metrics(err, df[cc].to_numpy(), sup, cfg)
        out[f"{head}/by_support"] = per_support_table(df, err_col, conf_cols[1]).to_dict(orient="records")
    # region-only view (pixels whose support was directly manipulated)
    reg = df[df.region_id >= 0]
    if len(reg):
        out["regions/depth/depth_conf_pct"] = confidence_metrics(
            reg.abs_rel.to_numpy(), reg.depth_conf_pct.to_numpy(), reg.support.to_numpy(), cfg
        )
    return out


def paired_region_deltas(df: pd.DataFrame, baseline: str = "clean") -> pd.DataFrame:
    """For every masked-region pixel, compare against the same pixel in the baseline run.

    Returns one row per (scene, condition, region) with mean confidence percentile /
    error in both runs and the deltas. Requires the same pixel stride for both.
    """
    keys = ["scene_id", "view", "y", "x"]
    base = df[df.condition == baseline][keys + ["depth_conf_pct", "abs_rel", "depth_conf"]]
    reg = df[(df.region_id >= 0) & (df.condition != baseline) & (df.kind == "masked")]
    if len(reg) == 0 or len(base) == 0:
        return pd.DataFrame()
    m = reg.merge(base, on=keys, suffixes=("", "_base"))
    g = m.groupby(["scene_id", "condition", "region_id", "target_support", "self_visible"])
    res = g.agg(
        n=("abs_rel", "size"),
        support_mean=("support", "mean"),
        conf_pct=("depth_conf_pct", "mean"),
        conf_pct_base=("depth_conf_pct_base", "mean"),
        log_conf=("depth_conf", lambda x: float(np.log(x).mean())),
        log_conf_base=("depth_conf_base", lambda x: float(np.log(x).mean())),
        abs_rel=("abs_rel", "mean"),
        abs_rel_base=("abs_rel_base", "mean"),
    ).reset_index()
    res["d_conf_pct"] = res.conf_pct - res.conf_pct_base
    res["d_log_conf"] = res.log_conf - res.log_conf_base
    res["d_abs_rel"] = res.abs_rel - res.abs_rel_base
    return res
