"""lib: reusable code for running and analysing feed-forward 3D reconstruction models.

Folders follow the pipeline; a file sits in a folder if it belongs to that
stage, and at the top level if every stage uses it.

    data/       what goes in        scene.py, preprocess.py             (implemented)
    models/     running a model     vggt.py, prediction.py              (implemented)
    eval/       scoring vs. GT      metrics.py                          (stub, next)
    analysis/   inside the model    hooks.py, knockouts.py              (stubs)
    geometry.py camera math used by all of the above
    viz.py      plotting

Adding another model (DA3, DUSt3R) means adding models/da3.py that returns the
same `Prediction`; data/, eval/ and geometry.py are reused unchanged.
"""

from lib.data.scene import Scene
from lib.models.prediction import Prediction

__all__ = ["Scene", "Prediction", "VGGTRunner"]


def __getattr__(name):
    # Import lazily so Scene/Prediction work without the upstream VGGT package installed.
    if name == "VGGTRunner":
        from lib.models.vggt import VGGTRunner
        return VGGTRunner
    raise AttributeError(name)
