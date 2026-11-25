"""Run Experiment 1 end to end: (generate if needed) -> VGGT -> per-pixel tables -> summary.

    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/smoke.yaml --checkpoint /path/to/model.pt
    python scripts/run_experiment.py --config configs/default.yaml --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confprobe.config import load_config  # noqa: E402
from confprobe.pipeline import aggregate, run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="local VGGT model.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-scenes", type=int, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    over = {}
    if args.checkpoint:
        over["model.checkpoint"] = args.checkpoint
    if args.device:
        over["model.device"] = args.device
    if args.n_scenes:
        over["n_scenes"] = args.n_scenes
    cfg = load_config(args.config, over)

    if not args.aggregate_only:
        from confprobe.vggt_runner import VGGTRunner

        runner = VGGTRunner.from_pretrained(cfg.model.weights, checkpoint=cfg.model.checkpoint, device=cfg.model.device)
        run_experiment(cfg, runner)
    summary = aggregate(cfg)
    a = summary["all"]
    print(json.dumps({k: a[k] for k in ("depth/depth_conf", "points/point_conf") if k in a}, indent=2))
    print(f"full summary: {Path(cfg.output.results_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
