"""Running models: one file per model (vggt.py), all returning a Prediction (prediction.py).

Inside this folder, `from vggt...` still means the upstream VGGT package
(absolute imports), never models/vggt.py. Don't run files in this folder
directly as scripts: that would put this folder on sys.path and make
`vggt` resolve to models/vggt.py.
"""

from lib.models.prediction import Prediction

__all__ = ["Prediction"]
