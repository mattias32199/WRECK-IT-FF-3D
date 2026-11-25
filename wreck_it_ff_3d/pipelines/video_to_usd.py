"""Video -> VGGT -> fused, gravity-aligned mesh -> USD scene for Isaac Sim.

    python -m wreck_it_ff_3d.pipelines.video_to_usd path/to/video.mp4 --out out/kitchen
    # re-run only the geometry steps on a saved prediction (no GPU needed):
    python -m wreck_it_ff_3d.pipelines.video_to_usd --from-prediction out/kitchen/prediction.npz --out out/kitchen

Writes to --out:
    frames/           the sampled video frames VGGT saw
    prediction.npz    raw VGGT output (Prediction.save)
    panels.png        image / depth / confidence for a few views
    mesh.ply          the final mesh (same frame as the USD), for MeshLab etc.
    scene.usd         load in Isaac Sim with isaac/load_scene.py
    summary.json      settings and diagnostics

Units: VGGT's scale is arbitrary, so 1 USD unit is NOT 1 m unless you pass
--camera-height (how high the camera was held, in meters; the scale is set so the
cameras' median height above the detected floor matches it) or --scale
(meters per VGGT unit).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from wreck_it_ff_3d.models.prediction import Prediction


def prediction_to_usd(
    pred: Prediction,
    out: Path,
    keep_conf: float = 0.8,
    voxel_frac: float = 1 / 80,
    max_faces: int = 300_000,
    max_rel_jump: Optional[float] = 0.05,
    consistency: Optional[dict] = None,
    scale: float = 1.0,
    camera_height: Optional[float] = None,
    level: bool = True,
    cameras: bool = True,
    collision: bool = True,
) -> dict:
    """Geometry half of the pipeline (no torch): fuse, clean, align, export."""
    from wreck_it_ff_3d.export.usd import write_scene_usd
    from wreck_it_ff_3d.recon.align import gravity_align, opencv_to_zup_transform, world_from_camera
    from wreck_it_ff_3d.recon.fusion import clean_mesh, fuse_tsdf

    t = time.perf_counter()
    print("[fuse]  TSDF fusion of VGGT depth maps")
    mesh, fuse_info = fuse_tsdf(pred, keep_conf=keep_conf, voxel_frac=voxel_frac,
                                max_rel_jump=max_rel_jump, consistency=consistency)
    mesh = clean_mesh(mesh, max_faces=max_faces)
    if mesh.num_faces == 0:
        raise RuntimeError("Fused mesh is empty: try a higher --keep-conf or a larger --voxel-frac")
    print(f"        {mesh.num_faces:,} faces, voxel {fuse_info['voxel']:.4g} (VGGT units)")

    print("[align] Z-up, floor at z=0" if level else "[align] Z-up only (--no-level)")
    if level:
        T, align_info = gravity_align(mesh.vertices, pred.extrinsics)
        print("        floor plane " + ("found" if align_info["floor_found"]
                                        else "NOT found, using camera up vectors"))
    else:
        T, align_info = opencv_to_zup_transform(), {}

    cam_z = np.array([(T @ M)[2, 3] for M in world_from_camera(pred.extrinsics)])
    align_info["camera_height_median_vggt_units"] = round(float(np.median(cam_z)), 5)
    if camera_height is not None:
        if not level or np.median(cam_z) <= 0:
            raise ValueError("--camera-height needs a detected floor below the cameras (don't use --no-level)")
        scale = camera_height / float(np.median(cam_z))
        print(f"[scale] cameras {np.median(cam_z):.4g} units above floor -> {camera_height} m: "
              f"scale {scale:.4g} m/unit")
    mesh = mesh.transformed(T, scale)

    cams = None
    if cameras:
        H, W = pred.depth.shape[1:]
        cams = []
        for M, K in zip(world_from_camera(pred.extrinsics), pred.intrinsics):
            M = T @ M
            M[:3, 3] *= scale
            cams.append((M, K, (H, W)))

    print("[usd]   writing scene.usd")
    usd_path = write_scene_usd(out / "scene.usd", mesh, cams, collision=collision)
    mesh.save_ply(out / "mesh.ply")

    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    return {
        "usd": str(usd_path),
        "faces": int(mesh.num_faces),
        "vertices": int(len(mesh.vertices)),
        "bbox_min": lo.round(4).tolist(),
        "bbox_max": hi.round(4).tolist(),
        "scale_m_per_vggt_unit": scale,
        "units_are_meters": scale != 1.0,
        "camera_height_m": camera_height,
        "fusion": fuse_info,
        "align": align_info,
        "seconds_geometry": round(time.perf_counter() - t, 2),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", nargs="?", help="input video (mp4, mov, ...)")
    p.add_argument("--from-prediction", help="skip frames+VGGT and start from a saved prediction.npz")
    p.add_argument("--out", help="output folder (default: out/<video name>)")
    p.add_argument("--frames", type=int, default=64, help="frames sampled from the video (64 fits in 16 GB; 96 does not)")
    p.add_argument("--spacing", choices=["motion", "time"], default="motion",
                   help="spread frames evenly by camera motion (more frames at turns) or by time")
    p.add_argument("--model-id", default="facebook/VGGT-1B")
    p.add_argument("--checkpoint", help="local model.pt instead of downloading --model-id")
    p.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    p.add_argument("--keep-conf", type=float, default=0.8, help="fraction of pixels kept by depth confidence")
    p.add_argument("--voxel-frac", type=float, default=1 / 80,
                   help="TSDF voxel size / median depth (smaller = more detail but more streaks)")
    p.add_argument("--edge-jump", type=float, default=0.05,
                   help="drop pixels at depth jumps larger than this fraction of depth (0 = off)")
    p.add_argument("--consistency", action="store_true",
                   help="keep only depth that neighbouring views agree with (removes ghost walls, "
                        "but also removes regions where poses are poor, e.g. corners)")
    p.add_argument("--max-faces", type=int, default=300_000, help="decimate the mesh to this many triangles")
    p.add_argument("--scale", type=float, default=1.0, help="meters per VGGT unit (default 1: units arbitrary)")
    p.add_argument("--camera-height", type=float,
                   help="camera height above the floor while filming, in m (sets --scale; ~1.4 for a hand-held phone)")
    p.add_argument("--no-level", action="store_true", help="skip floor detection; only re-axis to Z-up")
    p.add_argument("--no-cameras", action="store_true", help="don't write per-frame cameras into the USD")
    p.add_argument("--no-collision", action="store_true", help="visual mesh only")
    args = p.parse_args(argv)
    if bool(args.video) == bool(args.from_prediction):
        p.error("give either a video or --from-prediction")
    if args.camera_height is not None and args.scale != 1.0:
        p.error("give --camera-height or --scale, not both")

    summary: dict = {}
    if args.from_prediction:
        out = Path(args.out or Path(args.from_prediction).parent)
        pred = Prediction.load(args.from_prediction)
        summary["prediction"] = str(args.from_prediction)
    else:
        from wreck_it_ff_3d.data.scene import Scene
        from wreck_it_ff_3d.data.video import extract_frames
        from wreck_it_ff_3d.models.vggt import VGGTRunner
        from wreck_it_ff_3d.vis_utils import save_panels

        video = Path(args.video)
        out = Path(args.out or Path("out") / video.stem)
        print(f"[video] sampling {args.frames} frames from {video} (spacing: {args.spacing})")
        stale = list((out / "frames").glob("frame_*.png"))
        for f in stale:  # frames from an earlier run would be confusing next to the new ones
            f.unlink()
        paths = extract_frames(video, out / "frames", num_frames=args.frames, spacing=args.spacing)
        scene = Scene.build(video.stem, paths)
        tf = scene.transforms[0]
        print(f"        {len(paths)} frames, {tf.orig_w}x{tf.orig_h} -> VGGT input {tf.out_w}x{tf.out_h}")
        if tf.out_h < tf.new_h:
            lost = 1 - tf.out_h / tf.new_h
            print(f"        WARNING: portrait video, VGGT's center crop drops {lost:.0%} of each frame's height")

        print(f"[vggt]  running {args.checkpoint or args.model_id}")
        runner = VGGTRunner(model_id=args.model_id, checkpoint=args.checkpoint, device=args.device)
        pred = runner.run(scene)
        del runner
        print(f"        {pred.meta['seconds']} s, peak {pred.meta['peak_mem_gb']} GB on {pred.meta['device']}")
        pred.save(out / "prediction.npz")
        save_panels(pred, out / "panels.png")
        summary.update(video=str(video), frames=[str(x) for x in paths], vggt=pred.meta)

    summary.update(prediction_to_usd(
        pred, out,
        keep_conf=args.keep_conf, voxel_frac=args.voxel_frac, max_faces=args.max_faces,
        max_rel_jump=args.edge_jump or None,
        consistency=dict(neighbors=3, rel_tol=0.03, min_agree=2) if args.consistency else None,
        scale=args.scale, camera_height=args.camera_height, level=not args.no_level, cameras=not args.no_cameras,
        collision=not args.no_collision,
    ))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done]  {out / 'scene.usd'}")
    print(f"        open in Isaac Sim:  python isaac/load_scene.py {out / 'scene.usd'}   (in the Isaac Sim env; python.sh for a workstation install)")


if __name__ == "__main__":
    main()
