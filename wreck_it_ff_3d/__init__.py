"""wreck_it_ff_3d: reusable code for running and analysing feed-forward 3D reconstruction models.

Folders follow the pipeline; a file sits in a folder if it belongs to that
stage, and at the top level if every stage uses it.

    data/       what goes in        scene.py, preprocess.py, video.py   (implemented)
    models/     running a model     vggt.py, prediction.py              (implemented)
    eval/       scoring vs. GT      metrics.py                          (stub, next)
    analysis/   inside the model    hooks.py, knockouts.py              (stubs)
    geometry_utils.py  camera math used by all of the above
    vis_utils.py       plotting and .ply export
    recon/      prediction -> fused mesh, gravity alignment
    export/     mesh -> USD (for Isaac Sim)
    pipelines/  end-to-end CLIs (video_to_usd.py)

Adding another model (DA3, DUSt3R) means adding models/da3.py that returns the
same `Prediction`; data/, eval/ and geometry_utils.py are reused unchanged.
"""

from wreck_it_ff_3d.data.scene import Scene
from wreck_it_ff_3d.models.prediction import Prediction

__all__ = ["Scene", "Prediction", "VGGTRunner"]


def __getattr__(name):
    # Import lazily so Scene/Prediction work without the upstream VGGT package installed.
    if name == "VGGTRunner":
        from wreck_it_ff_3d.models.vggt import VGGTRunner
        return VGGTRunner
    raise AttributeError(name)
