"""Video -> a few sharp, well-spread frames on disk, ready for `Scene.build`.

VGGT runs on all views at once, so the number of frames is limited (~64 on a
16 GB GPU). A video has thousands, so we pick a subset: split the video into
`num_frames` bins and keep the sharpest of a few candidates in each bin
(variance of the Laplacian), which avoids motion-blurred frames.

Bins are equal in *camera motion* by default (spacing="motion"), not time:
walking straight changes the view slowly, turning a corner changes it fast.
With equal-time bins, frames at a turn end up 20-25 degrees apart with little
overlap, and VGGT mis-places them (IMG_1736: doubled walls after each turn).
Motion is measured as the median optical-flow magnitude between frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sharpness(frame_bgr: np.ndarray, max_side: int = 640) -> float:
    """Variance of the Laplacian on a downscaled grayscale copy (higher = sharper)."""
    import cv2

    h, w = frame_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def count_frames(video: str | Path) -> int:
    """Exact frame count by decoding headers (container metadata is often wrong)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def motion_profile(video: str | Path, rate_hz: float = 20.0, width: int = 256) -> tuple[int, np.ndarray, np.ndarray]:
    """(total frames, sampled frame indices, cumulative motion at those indices).

    Flow is measured between frames ~1/rate_hz s apart, small enough that fast
    turns stay within what the flow pyramid can track (otherwise fast motion is
    under-measured). Motion between consecutive samples = median optical-flow
    magnitude (pixels at `width`), plus a small floor so standing still still gets frames.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / rate_hz))
    prev, idx, mot, i = None, [], [], 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]
                g = cv2.cvtColor(cv2.resize(f, (width, round(width * h / w)), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY)
                if prev is None:
                    mot.append(0.0)
                else:
                    flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 5, 21, 3, 5, 1.2, 0)
                    mot.append(float(np.median(np.linalg.norm(flow, axis=2))))
                idx.append(i)
                prev = g
        i += 1
    cap.release()
    mot = np.asarray(mot)
    if len(mot) > 1:
        mot[1:] += 0.05 * mot[1:].mean()
    return i, np.asarray(idx), np.cumsum(mot)


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    num_frames: int = 32,
    candidates_per_frame: int = 5,
    edge_margin: float = 0.2,
    spacing: str = "motion",
) -> list[Path]:
    """Write up to `num_frames` frames to `out_dir/frame_XXXXX.png`, in time order.

    The number in the file name is the frame index in the video. Returns the paths
    written, in order. Rotation metadata (phone videos) is applied by OpenCV.
    Candidates come from the middle (1 - 2 * edge_margin) of each bin, so two
    neighbouring bins can't both pick near-identical frames at their shared edge.
    spacing: "motion" (equal optical-flow motion per bin) or "time" (equal duration).
    """
    import cv2

    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    if spacing not in ("motion", "time"):
        raise ValueError(f"spacing must be 'motion' or 'time', got {spacing!r}")
    if spacing == "motion":
        total, idx, cum = motion_profile(video)
    else:
        total = count_frames(video)
    if total == 0:
        raise ValueError(f"Could not decode any frames from {video}")

    # Bin edges (frame indices), then candidate frame indices grouped by bin.
    num_frames = min(num_frames, total)
    if spacing == "motion" and len(idx) > 1 and cum[-1] > 0:
        edges = np.interp(np.linspace(0, cum[-1], num_frames + 1), cum, idx)
        edges[0], edges[-1] = 0, total
    else:
        edges = np.linspace(0, total, num_frames + 1)
    bins: list[list[int]] = []
    used: set[int] = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo, hi = int(np.ceil(lo)), max(int(np.ceil(hi)) - 1, int(np.ceil(lo)))
        cut = int(edge_margin * (hi - lo))
        lo, hi = lo + cut, hi - cut
        k = min(candidates_per_frame, hi - lo + 1)
        cands = sorted(set(np.linspace(lo, hi, k).round().astype(int).tolist()) - used)
        if cands:  # very short bins (fast turns) can collide with their neighbour
            used.update(cands)
            bins.append(cands)
    bin_of = {i: b for b, idx in enumerate(bins) for i in idx}
    last_of_bin = {idx[-1]: b for b, idx in enumerate(bins)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    best: tuple[float, int, np.ndarray] | None = None  # (sharpness, index, frame) in the current bin
    written: list[Path] = []
    for i in range(total):
        if not cap.grab():
            break
        if i not in bin_of:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        s = sharpness(frame)
        if best is None or s > best[0]:
            best = (s, i, frame)
        if i in last_of_bin and best is not None:
            path = out_dir / f"frame_{best[1]:05d}.png"
            cv2.imwrite(str(path), best[2])
            written.append(path)
            best = None
    cap.release()
    if not written:
        raise ValueError(f"No frames extracted from {video}")
    return written
