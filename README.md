# WRECK-IT-FF-3D

## Setup
```bash
python3 -m venv ~/venvs/vggt && source ~/venvs/vggt/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA / GPU
pip install --no-deps git+https://github.com/facebookresearch/vggt.git            # upstream VGGT (pins an old torch)
pip install -e ".[test]"
pytest tests/                                                                      # no GPU needed
```

## Pipeline

### 1. Dataset
Views/objects for prediction and experiments.
- [DTU](https://ieeexplore.ieee.org/document/6909453)

### Video → VGGT → Isaac Sim
```bash
# 1. In the VGGT environment: video -> scene.usd (mesh with static collider + per-frame cameras)
python -m wreck_it_ff_3d.pipelines.video_to_usd my_video.mp4 --out out/my_scene

# 2. With Isaac Sim's own Python: load it with physics and drop a test cube
<isaac-sim>/python.sh isaac/load_scene.py out/my_scene/scene.usd
```

Steps (each a separate module, so they can be swapped or tuned):

| Step | Module | What it does |
|---|---|---|
| Frames | `data/video.py` | `--frames` (64) frames spread evenly by camera motion (optical flow), so turns get more frames; sharpest of 5 candidates per bin |
| VGGT | `models/vggt.py` | depth, confidence and cameras for all frames in one pass |
| Fusion | `recon/fusion.py` | drop low-confidence (`--keep-conf`) and depth-edge (`--edge-jump`) pixels, optionally multi-view-inconsistent ones (`--consistency`), TSDF-fuse, clean and decimate (`--max-faces`) |
| Align | `recon/align.py` | Z-up, floor plane at z = 0, first camera looking along +X |
| Export | `export/usd.py` | USD mesh with vertex colors and triangle-mesh collider, one USD camera per frame |

Things to know:
- **Scale is arbitrary.** VGGT doesn't know real size. Pass `--camera-height 1.4` (how high the phone was held, in meters) and the pipeline scales the scene so the cameras sit that high above the detected floor. Or pass `--scale <meters per unit>` if you know a real distance.
- **Memory:** 64 frames use about 10.3 GB, and 96 run out of memory on a 16 GB card. More frames, and motion spacing (the default), make the poses at turns noticeably better.
- **Portrait videos** lose the top and bottom of each frame, because VGGT's preprocessing crops the height to 518 px. The CLI prints how much is lost.
- To re-tune fusion without re-running VGGT: `--from-prediction out/my_scene/prediction.npz`.
- Check the result: in Isaac Sim, look through `/World/Reconstruction/Cameras/view_000`. The view should line up with `frames/` frame 0.

Isaac Sim ships its own Python, so it can't share the VGGT environment. The pipeline only hands it `scene.usd`. Install a recent Isaac Sim (5.x for RTX 50-series GPUs) as the workstation build or the `nvcr.io/nvidia/isaac-sim` container. In the container, run `./python.sh /workspace/isaac/load_scene.py ... --headless`, which prints whether the test cube came to rest on the scene.

## File Tree
```
├── README.md
├── pyproject.toml
├── isaac
│   └── load_scene.py            // runs inside Isaac Sim
├── tests
│   └── test_video_to_usd.py     // synthetic scene, no GPU
└── wreck_it_ff_3d
    ├── __init__.py
    ├── analysis // stubs, placeholders
    │   ├── __init__.py
    │   ├── hooks.py
    │   └── knockouts.py
    ├── data
    │   ├── __init__.py
    │   ├── preprocess.py
    │   ├── scene.py
    │   └── video.py
    ├── eval
    │   ├── __init__.py
    │   └── metrics.py
    ├── export
    │   ├── __init__.py
    │   └── usd.py
    ├── geometry_utils.py
    ├── models
    │   ├── __init__.py
    │   ├── prediction.py
    │   └── vggt.py
    ├── pipelines
    │   ├── __init__.py
    │   └── video_to_usd.py
    ├── recon
    │   ├── __init__.py
    │   ├── align.py
    │   └── fusion.py
    └── vis_utils.py
```
