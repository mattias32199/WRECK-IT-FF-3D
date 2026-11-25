"""Open a reconstructed scene.usd in Isaac Sim with physics, and drop a test cube on it.

Run with Isaac Sim's own Python, NOT the VGGT environment:

    <isaac-sim>/python.sh isaac/load_scene.py out/kitchen/scene.usd
    <isaac-sim>/python.sh isaac/load_scene.py out/kitchen/scene.usd --headless --steps 300

The reconstruction is referenced (not copied) under /World/Reconstruction, so
re-running the pipeline updates it. --save writes this wrapper stage so you can
open it later from the Isaac Sim GUI (File > Open).

Uses only SimulationApp + plain USD/PhysX APIs, which are stable across Isaac Sim 4.x/5.x.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("usd", help="scene.usd from wreck_it_ff_3d.pipelines.video_to_usd")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=300, help="frames to simulate when --headless")
parser.add_argument("--no-drop", action="store_true", help="don't add the falling test cube")
parser.add_argument("--paused", action="store_true", help="don't start the simulation automatically")
parser.add_argument("--save", help="write the wrapper stage (with physics scene + light) to this .usd")
args = parser.parse_args()
usd_path = os.path.abspath(args.usd)
if not os.path.exists(usd_path):
    raise SystemExit(f"not found: {usd_path}")

from isaacsim import SimulationApp  # noqa: E402  (must start before any omni/pxr import)

app = SimulationApp({"headless": args.headless})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
physics.CreateGravityMagnitudeAttr(9.81)
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

recon = stage.DefinePrim("/World/Reconstruction", "Xform")
recon.GetReferences().AddReference(usd_path)

bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(recon)
box = bounds.ComputeAlignedRange()
lo, hi = box.GetMin(), box.GetMax()
diag = (hi - lo).GetLength()
center = (lo + hi) / 2
print(f"[load] {usd_path}")
print(f"[load] scene bounds min {tuple(round(x, 3) for x in lo)} max {tuple(round(x, 3) for x in hi)}")

cube_path = "/World/TestCube"
if not args.no_drop:
    # Drop from where the first frame was filmed: there is floor under the camera path,
    # whereas the bounding-box center of an L-shaped or multi-alley scene can be empty.
    cam = stage.GetPrimAtPath("/World/Reconstruction/Cameras/view_000")
    if cam.IsValid():
        cam_pos = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        size = 0.25 * max(cam_pos[2], 1e-3)  # ~35 cm for a phone held at ~1.4 m
        start = Gf.Vec3d(cam_pos[0], cam_pos[1], cam_pos[2])
    else:
        size = 0.01 * diag
        start = Gf.Vec3d(center[0], center[1], hi[2] + size)
    print(f"[load] test cube: size {size:.3f}, dropped from {tuple(round(x, 3) for x in start)}")
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
    UsdGeom.XformCommonAPI(cube).SetTranslate(start)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)

# Frame the scene in the viewport (API moved between versions; skip if unavailable).
eye = [center[0] - 0.8 * diag, center[1] - 0.8 * diag, center[2] + 0.6 * diag]
target = [center[0], center[1], center[2]]
for mod in ("isaacsim.core.utils.viewports", "omni.isaac.core.utils.viewports"):
    try:
        __import__(mod, fromlist=["set_camera_view"]).set_camera_view(eye=eye, target=target)
        break
    except Exception:
        continue

if args.save:
    stage.GetRootLayer().Export(os.path.abspath(args.save))
    print(f"[load] wrapper stage saved to {args.save}")

if not args.paused:
    omni.timeline.get_timeline_interface().play()


def cube_height():
    """Cube z from PhysX (works whether or not results are written back to USD)."""
    try:
        from omni.physx import get_physx_interface

        pose = get_physx_interface().get_rigidbody_transformation(cube_path)
        if pose.get("ret_val"):
            return float(pose["position"][2])
    except Exception:
        pass
    return None


step = 0
while app.is_running():
    app.update()
    step += 1
    if args.headless and step >= args.steps:
        break

if args.headless and not args.no_drop and not args.paused:
    z = cube_height()
    if z is None:
        print("[check] could not read the cube pose; check the viewport")
    elif z < lo[2] - diag:
        print(f"[check] cube fell through the scene (z = {z:.3f}): collision is not working")
    else:
        print(f"[check] cube resting at z = {z:.3f} (expected about {size / 2:.3f} if it landed on the floor at z = 0)")

app.close()
