"""Make figures + summary from a finished results directory.

    python scripts/analyze.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confprobe.config import load_config  # noqa: E402
from confprobe.metrics import paired_region_deltas  # noqa: E402
from confprobe.pipeline import aggregate, load_tables  # noqa: E402
from confprobe.visualize import make_report_figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pix, poses = load_tables(cfg.output.results_dir, frac=cfg.eval.aggregate_frac)  # load once
    print(f"loaded {len(pix):,} pixel rows (aggregate_frac={cfg.eval.aggregate_frac})")
    aggregate(cfg, pix, poses)
    print(Path(cfg.output.results_dir) / "summary.json")
    paths = make_report_figures(pix, paired_region_deltas(pix), Path(cfg.output.results_dir) / "figures")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
