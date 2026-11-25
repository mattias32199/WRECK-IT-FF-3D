"""Render synthetic scenes and build their experimental conditions.

    python scripts/generate_data.py --config configs/default.yaml
    python scripts/generate_data.py --config configs/default.yaml --preview 3   # also write preview PNGs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confprobe.config import load_config  # noqa: E402
from confprobe.pipeline import generate_one  # noqa: E402
from confprobe.visualize import save_condition_preview  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-scenes", type=int, default=None)
    ap.add_argument("--preview", type=int, default=0, help="write preview PNGs for the first N scenes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    over = {"n_scenes": args.n_scenes} if args.n_scenes else {}
    cfg = load_config(args.config, over)
    for i, seed in enumerate(cfg.seeds):
        sample, conds = generate_one(cfg, seed)
        logging.info("%s: %s", sample.scene_id, ", ".join(f"{c.name}({len(c.regions)} regions)" for c in conds))
        if i < args.preview:
            for c in conds:
                p = Path(cfg.output.data_dir) / sample.scene_id / f"preview_{c.name}.png"
                save_condition_preview(sample, c, p)


if __name__ == "__main__":
    main()
