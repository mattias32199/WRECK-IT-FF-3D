# Experiment 1 — Does VGGT's confidence reflect geometric support?

Controlled synthetic scenes → VGGT → per-pixel (support, error, confidence) tables → metrics.

```
synthetic data            vggt                  results + metrics
──────────────            ────                  ─────────────────
scene.py      (room + textured objects)
renderer.py   (torch ray caster, exact GT)
occlusion.py  (3D regions → per-view masks) ──► vggt_runner.py ──► metrics.py  (align, errors, Spearman/AUROC/AUSE)
support.py    (exact per-pixel support)                            pipeline.py (parquet tables, resume, aggregate)
dataset.py    (conditions, save/load)                              visualize.py (figures)
```

## Core idea

**Geometric support** of a pixel = the number of input views in which its ground-truth
3D point is visible *and not masked*. It is computed exactly from GT depth for every
pixel (`support.py`), so it is ground truth, not just the intended value.

We control it with **occlusion regions**: balls in 3D. For a region with target support *k*,
we keep *k* of the views that see it and hide its footprint in all the other views
(`occlusion.py`). Guarantee (tested): no pixel in a region ever gets support above its target.

| support | meaning |
|---|---|
| 0 | hidden in every view: pure guess from context/priors |
| 1 | seen in exactly one view: monocular |
| 2+ | multi-view: can be triangulated |

Each pixel also stores `self_visible`, which records whether the pixel is visible in its own view.
That separates the case "hidden here but seen elsewhere" (the model must transfer it across views)
from "visible here".

### Conditions per scene
* `clean`: all views, nothing masked. This is the paired baseline for every masked pixel.
* `masked_k`: all views, with 4 regions whose targets are 0/1/2/3.
* `subsetN_k`: N views, nothing masked. Support then comes from view selection, the proposal's
  second occlusion method.

## Install

```bash
pip install -e ".[test]"
pip install --no-deps git+https://github.com/facebookresearch/vggt.git   # + huggingface_hub einops safetensors
```
(`--no-deps` is used because vggt pins `numpy<2`, but it runs fine on numpy 2.)

## Run

```bash
python scripts/generate_data.py --config configs/default.yaml --preview 3   # render + preview PNGs
python scripts/run_experiment.py --config configs/smoke.yaml                 # 2 tiny scenes, sanity
python scripts/run_experiment.py --config configs/default.yaml               # full run (resumable)
python scripts/analyze.py --config configs/default.yaml                      # summary.json + figures/
```
Offline cluster (PSC): download `model.pt` from `facebook/VGGT-1B` once and pass `--checkpoint`.

## Tests

```bash
pytest                    # 73 fast tests, CPU only, no weights needed (~10 s)
pytest -m slow            # real VGGT-1B on a GPU (HF download or VGGT_CHECKPOINT=/path/model.pt)
```

| file | covers |
|---|---|
| `test_geometry.py` | camera conventions, projection round-trips, pose encoding (vs. official vggt) |
| `test_renderer.py` | analytic depths (sphere/box), surface points, multi-view color consistency, determinism |
| `test_occlusion.py` | support with/without occluders and masks, target-support guarantee, fills |
| `test_dataset.py` | conditions, GT slicing, save/load round-trip |
| `test_vggt.py` | preprocessing/validation, output conversion, pose decoding; tiny *real-architecture* VGGT (random weights) for the output contract; slow tests with real weights |
| `test_metrics.py` | AUROC/Spearman/AUSE vs. known values, scale alignment, zero error on perfect predictions |
| `test_pipeline.py` | end-to-end with an *oracle* model whose confidence tracks support. Checks that metrics report the right direction, plus resume and figures |

## Outputs

`results/<exp>/pixels/*.parquet` has one row per (strided) pixel:
`support, n_visible, self_visible, region_id, target_support, abs_rel, point_err,
depth_conf, point_conf, *_conf_pct, gt_depth, is_object, …`.
`summary.json` has, overall and per condition kind:
* `spearman_conf_err` (Spearman(conf, error); negative means good)
* `auroc_err_*`: does low confidence flag high-error pixels
* `auroc_support_le1_vs_2plus`, `auroc_support0_vs_2plus`: does low confidence flag guessing
* `ause`: sparsification error
* per-support tables

`region_deltas.csv` holds the paired masked-vs-clean confidence change per region.

## Design choices worth knowing
* **No resizing.** Images render at 518×392 (multiples of 14), so VGGT output is pixel-aligned with GT.
* **Scale alignment** uses a median ratio over pixels with support ≥ 2 that are visible in their own view.
  This keeps the guessed regions from biasing the scale.
* **Confidence percentile** is computed within each run, because raw VGGT confidence is `1+exp(x)` and its scale varies by scene.
  Spearman and AUROC are computed for both raw and percentile.
* **`bbox` masks** are the default. `silhouette` masks leak the object outline through the mask shape.
* **Visibility** uses a depth test (bilinear or nearest-of-4) plus a grazing-angle cutoff (`support_min_cos`).
  The cutoff removes aliasing at silhouettes.
* Data is cached. Delete `data_dir` (or set `overwrite: true`) after changing scene/occlusion settings.
