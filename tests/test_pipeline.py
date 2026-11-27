"""End-to-end pipeline with an oracle model whose behaviour is known."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from confprobe.config import ExperimentConfig, load_config
from confprobe.dataset import ConditionSample, SceneSample
from confprobe.pipeline import aggregate, generate_dataset, generate_one, load_tables, run_experiment
from confprobe.visualize import make_report_figures, save_condition_preview
from confprobe.metrics import paired_region_deltas
from confprobe.vggt_runner import VGGTRunner
from conftest import OracleVGGT, small_overrides


def _oracle_runner(cfg):
    model = OracleVGGT()
    for seed in cfg.seeds:
        sample, conds = generate_one(cfg, seed)
        for c in conds:
            model.register(c, c.gt(sample))
    return VGGTRunner(model, device="cpu", strict_size=False)


# --------------------------------------------------------------------------- #
def test_config_loading(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"n_scenes": 3, "camera": {"n_views": 5},
                                 "conditions": {"occlusion": {"target_supports": [0, 2]}}}))
    cfg = load_config(p, {"conditions.occlusion.fill": "noise"})
    assert cfg.n_scenes == 3 and cfg.camera.n_views == 5
    assert cfg.conditions.occlusion.target_supports == (0, 2)  # lists become tuples
    assert cfg.conditions.occlusion.fill == "noise"
    assert cfg.camera.width == 518  # defaults kept
    with pytest.raises(KeyError):
        load_config(None, {"camera.not_a_field": 1})


def test_shipped_configs_parse():
    for name in ("configs/default.yaml", "configs/smoke.yaml"):
        cfg = load_config(name)
        assert cfg.camera.height % 14 == 0 and cfg.camera.width % 14 == 0


def test_generate_dataset_and_cache(small_cfg):
    ids = generate_dataset(small_cfg)
    assert len(ids) == 2
    for sid in ids:
        assert SceneSample.exists(small_cfg.output.data_dir, sid)
        assert "clean" in ConditionSample.list_names(small_cfg.output.data_dir, sid)
    # second call loads from disk: identical arrays
    s1, c1 = generate_one(small_cfg, small_cfg.seeds[0])
    s2, c2 = generate_one(small_cfg, small_cfg.seeds[0])
    assert np.array_equal(c1[1].images, c2[1].images)


def test_end_to_end_with_oracle(small_cfg):
    runner = _oracle_runner(small_cfg)
    out = run_experiment(small_cfg, runner)
    pix, poses = load_tables(out)
    n_conds = sum(len(ConditionSample.list_names(small_cfg.output.data_dir, s)) for s in pix.scene_id.unique())
    assert pix.groupby(["scene_id", "condition"]).ngroups == n_conds
    assert len(poses) > 0 and (out / "config.yaml").exists()

    summary = aggregate(small_cfg)
    assert (out / "summary.json").exists()
    m = summary["all"]["depth/depth_conf"]
    # the oracle's confidence tracks its error -> the metrics must say so
    assert m["spearman_conf_err"] < -0.3
    assert m["auroc_err_quantile"] > 0.7
    assert m["auroc_support_le1_vs_2plus"] > 0.8
    # oracle geometry is exact up to noise -> poses near perfect, scale recovered
    assert poses.rot_err_deg.max() < 0.5
    infos = [json.loads(p.read_text()) for p in (out / "info").glob("*.json")]
    assert all(abs(i["scale_depth"] - 1 / 0.37) / (1 / 0.37) < 0.02 for i in infos)
    # masked regions lose confidence relative to the clean run
    deltas = paired_region_deltas(pix)
    assert len(deltas) > 0
    zero = deltas[deltas.target_support == 0]
    if len(zero):
        assert zero.d_conf_pct.mean() < 0

    figs = make_report_figures(pix, deltas, out / "figures")
    assert all(f.exists() and f.stat().st_size > 0 for f in figs)


def test_resume_skips_finished_conditions(small_cfg):
    small_cfg.n_scenes = 1
    runner = _oracle_runner(small_cfg)
    out = run_experiment(small_cfg, runner)
    files = sorted((out / "pixels").glob("*.parquet"))
    mtimes = [f.stat().st_mtime_ns for f in files]

    class Boom(OracleVGGT):
        def forward(self, images):
            raise AssertionError("should not be called when results exist")

    run_experiment(small_cfg, VGGTRunner(Boom(), device="cpu", strict_size=False))
    assert [f.stat().st_mtime_ns for f in files] == mtimes


def test_raw_prediction_saving(small_cfg):
    small_cfg.n_scenes = 1
    small_cfg.output.save_raw_predictions = True
    out = run_experiment(small_cfg, _oracle_runner(small_cfg))
    assert len(list((out / "preds").glob("*.npz"))) == len(list((out / "pixels").glob("*.parquet")))


def test_condition_preview(tmp_path, small_sample, small_conditions):
    p = save_condition_preview(small_sample, small_conditions[1], tmp_path / "prev.png")
    assert p.exists() and p.stat().st_size > 1000


def test_pixel_sample_mask_is_deterministic_and_paired():
    from confprobe.pipeline import pixel_sample_mask

    yy, xx = np.meshgrid(np.arange(0, 392, 2), np.arange(0, 518, 2), indexing="ij")
    df = pd.DataFrame({"view": 3, "y": yy.ravel(), "x": xx.ravel()})
    m = pixel_sample_mask(df, 0.25)
    assert abs(m.mean() - 0.25) < 0.01
    # same pixels regardless of row order / which condition the table came from
    shuffled = df.sample(frac=1.0, random_state=0)
    assert np.array_equal(pixel_sample_mask(shuffled, 0.25), m[shuffled.index.to_numpy()])
    assert pixel_sample_mask(df, 1.0).all()
    # no obvious spatial structure: both halves of the image are sampled evenly
    assert abs(m[df.x.to_numpy() < 259].mean() - m[df.x.to_numpy() >= 259].mean()) < 0.02


def test_aggregate_with_subsample_keeps_pairs(small_cfg):
    small_cfg.n_scenes = 1
    run_experiment(small_cfg, _oracle_runner(small_cfg))
    small_cfg.eval.aggregate_frac = 1.0
    full = aggregate(small_cfg)
    small_cfg.eval.aggregate_frac = 0.3
    sub = aggregate(small_cfg)
    assert sub["n_pixel_rows"] < 0.4 * full["n_pixel_rows"]
    assert sub["pixel_frac"] == 0.3
    # paired deltas still exist and summary metrics stay close to the full-data values
    assert "region_deltas_by_target" in sub
    a, b = full["all"]["depth/depth_conf"], sub["all"]["depth/depth_conf"]
    assert abs(a["spearman_conf_err"] - b["spearman_conf_err"]) < 0.1
