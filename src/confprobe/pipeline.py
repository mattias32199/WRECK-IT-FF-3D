"""End-to-end Experiment 1 pipeline: generate -> run VGGT -> evaluate -> store -> aggregate.

Results layout:
    <results_dir>/
        config.yaml
        pixels/<scene_id>__<condition>.parquet   per-pixel table (strided)
        poses/<scene_id>__<condition>.parquet    per-view pose errors
        info/<scene_id>__<condition>.json        alignment scales etc.
        preds/<scene_id>__<condition>.npz        raw predictions (optional)
        summary.json                             aggregated metrics (after `aggregate`)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .dataset import ConditionSample, SceneSample, build_conditions, generate_scene, scene_id_for
from .metrics import evaluate_condition, paired_region_deltas, summarize
from .vggt_runner import VGGTRunner

log = logging.getLogger("confprobe")


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_one(cfg: ExperimentConfig, seed: int) -> tuple[SceneSample, list[ConditionSample]]:
    root = Path(cfg.output.data_dir)
    sid = scene_id_for(seed)
    if SceneSample.exists(root, sid) and not cfg.output.overwrite:
        sample = SceneSample.load(root, sid)
        names = ConditionSample.list_names(root, sid)
        if names:
            return sample, [ConditionSample.load(root, sid, n) for n in names]
    else:
        sample = generate_scene(seed, cfg.scene, cfg.camera, ssaa=cfg.render.ssaa, device=cfg.render.device)
        sample.save(root)
    conds = build_conditions(sample, cfg.conditions, seed, device=cfg.render.device)
    for c in conds:
        c.save(root)
    return sample, conds


def generate_dataset(cfg: ExperimentConfig) -> list[str]:
    ids = []
    for seed in cfg.seeds:
        t0 = time.time()
        sample, conds = generate_one(cfg, seed)
        ids.append(sample.scene_id)
        log.info("generated %s (%d conditions) in %.1fs", sample.scene_id, len(conds), time.time() - t0)
    return ids


# --------------------------------------------------------------------------- #
# Running + evaluation
# --------------------------------------------------------------------------- #
def _stem(scene_id: str, cond: str) -> str:
    return f"{scene_id}__{cond}"


def run_condition(runner: VGGTRunner, sample: SceneSample, cond: ConditionSample, cfg: ExperimentConfig):
    pred = runner.run(cond.images)
    pix, poses, info = evaluate_condition(cond, cond.gt(sample), pred, cfg.eval)
    return pred, pix, poses, info


def run_experiment(cfg: ExperimentConfig, runner: VGGTRunner) -> Path:
    out = Path(cfg.output.results_dir)
    for sub in ("pixels", "poses", "info", "preds"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    for seed in cfg.seeds:
        sample, conds = generate_one(cfg, seed)
        for cond in conds:
            stem = _stem(sample.scene_id, cond.name)
            if (out / "pixels" / f"{stem}.parquet").exists() and not cfg.output.overwrite:
                continue
            t0 = time.time()
            pred, pix, poses, info = run_condition(runner, sample, cond, cfg)
            info["runtime_s"] = time.time() - t0
            pix.to_parquet(out / "pixels" / f"{stem}.parquet", index=False)
            poses.to_parquet(out / "poses" / f"{stem}.parquet", index=False)
            (out / "info" / f"{stem}.json").write_text(json.dumps(info))
            if cfg.output.save_raw_predictions:
                pred.save(out / "preds" / f"{stem}.npz")
            log.info("%s: %.1fs, mean abs_rel %.4f", stem, info["runtime_s"], float(pix.abs_rel.mean()))
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
CATEGORICAL_COLS = ("scene_id", "condition", "kind")


def pixel_sample_mask(df: pd.DataFrame, frac: float) -> np.ndarray:
    """Deterministic subsample keyed on (view, y, x) only, so the *same* pixels are kept
    in every condition and scene (paired clean-vs-masked comparisons stay intact)."""
    if frac >= 1.0:
        return np.ones(len(df), dtype=bool)
    h = (
        df["view"].to_numpy(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        ^ df["y"].to_numpy(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        ^ df["x"].to_numpy(np.uint64) * np.uint64(0x94D049BB133111EB)
    )
    h ^= h >> np.uint64(31)
    h *= np.uint64(0xD6E8FEB86659FD93)
    h ^= h >> np.uint64(32)
    return (h % np.uint64(1_000_000)) < np.uint64(int(frac * 1_000_000))


def _read_with_shared_categories(files: list[Path], columns=None, frac: float = 1.0) -> pd.DataFrame:
    """Concatenate parquet tables keeping string columns categorical (with one shared
    category set), optionally subsampling each file first, so 50+ scenes of per-pixel
    rows fit in Colab-sized RAM."""
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=columns)
        if frac < 1.0:
            df = df[pixel_sample_mask(df, frac)].reset_index(drop=True)
        frames.append(df)
    for col in CATEGORICAL_COLS:
        if all(col in f for f in frames):
            cats = sorted(set().union(*(set(f[col].astype(str).unique()) for f in frames)))
            for f in frames:
                f[col] = pd.Categorical(f[col].astype(str), categories=cats)
    return pd.concat(frames, ignore_index=True)


def load_tables(
    results_dir: str | Path, columns: list[str] | None = None, frac: float = 1.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all per-pixel and pose tables.

    `columns` loads a subset of columns and `frac` a deterministic subset of pixels
    (see :func:`pixel_sample_mask`); both save RAM.
    """
    out = Path(results_dir)
    pix_files = sorted((out / "pixels").glob("*.parquet"))
    if not pix_files:
        raise FileNotFoundError(f"no results in {out / 'pixels'}")
    pix = _read_with_shared_categories(pix_files, columns, frac)
    pose_files = sorted((out / "poses").glob("*.parquet"))
    poses = pd.concat([pd.read_parquet(f) for f in pose_files], ignore_index=True) if pose_files else pd.DataFrame()
    return pix, poses


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def aggregate(cfg: ExperimentConfig, pix: pd.DataFrame | None = None, poses: pd.DataFrame | None = None) -> dict:
    """Write summary.json (+ region_deltas.csv). Loads a `cfg.eval.aggregate_frac` pixel
    subsample unless tables are passed in."""
    out = Path(cfg.output.results_dir)
    if pix is None:
        pix, poses = load_tables(out, frac=cfg.eval.aggregate_frac)
        log.info("aggregating %d pixel rows (frac=%.2f)", len(pix), cfg.eval.aggregate_frac)
    poses = pd.DataFrame() if poses is None else poses
    summary = {
        "n_scenes": int(pix.scene_id.nunique()),
        "pixel_frac": cfg.eval.aggregate_frac,
        "n_pixel_rows": int(len(pix)),
        "all": summarize(pix, cfg.eval),
        "by_kind": {k: summarize(g, cfg.eval) for k, g in pix.groupby("kind", observed=True)},
    }
    if len(poses):
        summary["poses"] = (
            poses[poses.view > 0]
            .groupby(poses.condition.astype(str).str.split("_").str[0])
            .agg(rot_err_deg=("rot_err_deg", "mean"), trans_ang_err_deg=("trans_ang_err_deg", "mean"))
            .to_dict(orient="index")
        )
    deltas = paired_region_deltas(pix)
    if len(deltas):
        deltas.to_csv(out / "region_deltas.csv", index=False)
        summary["region_deltas_by_target"] = (
            deltas.groupby(["target_support", "self_visible"])[["d_conf_pct", "d_log_conf", "d_abs_rel", "support_mean"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
    summary = _jsonable(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
