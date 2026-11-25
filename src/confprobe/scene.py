"""Procedural synthetic scenes: an enclosed room with textured primitive objects.

Every surface carries a *solid* (3D, world-space) procedural texture, so a given
3D point has the same albedo in every view, and lighting is view-independent
(Lambertian, directional lights, no shadows). That makes the rendered views
photometrically consistent, which is what a multi-view model expects.

The room is closed, so every camera ray hits a surface and every pixel has
valid ground-truth depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from .geometry import intrinsics_from_fov, look_at

KIND_ROOM, KIND_BOX, KIND_SPHERE = 0, 1, 2
KIND_NAMES = {KIND_ROOM: "room", KIND_BOX: "box", KIND_SPHERE: "sphere"}
NUM_FACES = {KIND_ROOM: 6, KIND_BOX: 6, KIND_SPHERE: 1}


@dataclass
class Texture:
    """Parameters of a solid procedural texture.

    albedo(p) = lerp(A[f], B[f], sigmoid(sharpness * pattern(p))) + fine_amp * fine(p)
    where pattern/fine are sums of random 3D plane waves.
    """

    colors_a: np.ndarray  # (F, 3) per-face primary color
    colors_b: np.ndarray  # (F, 3) per-face secondary color
    freqs: np.ndarray  # (K, 3) wave vectors (rad / scene unit)
    phases: np.ndarray  # (K,)
    amps: np.ndarray  # (K,)
    sharpness: float
    fine_freqs: np.ndarray  # (3, Kf, 3) per-channel fine detail
    fine_phases: np.ndarray  # (3, Kf)
    fine_amp: float


@dataclass
class Primitive:
    kind: int
    center: np.ndarray  # (3,)
    half_extents: np.ndarray  # (3,) for room/box; [r, r, r] for sphere
    rotation: np.ndarray  # (3, 3) local -> world
    texture: Texture
    name: str = ""

    @property
    def radius(self) -> float:
        return float(self.half_extents[0])


@dataclass
class Scene:
    primitives: list[Primitive]
    light_dirs: np.ndarray  # (L, 3) unit vectors pointing *towards* the light
    light_strengths: np.ndarray  # (L,)
    ambient: float
    seed: int = 0

    @property
    def room(self) -> Primitive:
        return self.primitives[0]

    def to_dict(self) -> dict:
        def conv(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, list):
                return [conv(v) for v in x]
            return x

        return conv(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        prims = []
        for p in d["primitives"]:
            t = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in p["texture"].items()}
            prims.append(
                Primitive(
                    kind=int(p["kind"]),
                    center=np.asarray(p["center"], dtype=np.float32),
                    half_extents=np.asarray(p["half_extents"], dtype=np.float32),
                    rotation=np.asarray(p["rotation"], dtype=np.float32),
                    texture=Texture(**t),
                    name=p.get("name", ""),
                )
            )
        return Scene(
            primitives=prims,
            light_dirs=np.asarray(d["light_dirs"], dtype=np.float32),
            light_strengths=np.asarray(d["light_strengths"], dtype=np.float32),
            ambient=float(d["ambient"]),
            seed=int(d.get("seed", 0)),
        )


@dataclass
class Cameras:
    extrinsics: np.ndarray  # (S, 3, 4) world -> cam, OpenCV
    intrinsics: np.ndarray  # (S, 3, 3)
    height: int
    width: int

    def __len__(self) -> int:
        return self.extrinsics.shape[0]

    def subset(self, idx) -> "Cameras":
        idx = list(idx)
        return Cameras(self.extrinsics[idx], self.intrinsics[idx], self.height, self.width)


# --------------------------------------------------------------------------- #
# Random generation
# --------------------------------------------------------------------------- #
@dataclass
class SceneConfig:
    # room: x/y half-size and height
    room_half_size: tuple[float, float] = (4.5, 6.0)
    room_height: tuple[float, float] = (2.8, 3.6)
    # objects placed within this radius of the room center
    n_objects: tuple[int, int] = (6, 12)
    object_radius: float = 1.6
    object_size: tuple[float, float] = (0.15, 0.55)  # half extent / sphere radius
    sphere_prob: float = 0.35
    floating_prob: float = 0.25  # fraction of objects not resting on the floor
    # texture
    n_waves: int = 16
    wave_freq: tuple[float, float] = (1.5, 40.0)  # rad / m, log-uniform
    fine_waves: int = 6
    fine_amp: float = 0.08
    sharpness: tuple[float, float] = (2.0, 10.0)
    ambient: float = 0.35


@dataclass
class CameraConfig:
    n_views: int = 8
    height: int = 392  # multiple of 14 so VGGT sees pixels 1:1
    width: int = 518
    fov_x_deg: float = 65.0
    radius: tuple[float, float] = (2.4, 3.4)  # distance of cameras from room center
    cam_height: tuple[float, float] = (1.0, 2.0)
    arc_deg: float = 120.0  # total angular span of the camera arc (smaller = more overlap)
    angle_jitter_deg: float = 4.0
    target_jitter: float = 0.35
    target_height: tuple[float, float] = (0.4, 0.9)
    shuffle: bool = False  # randomize the view order (view 0 defines VGGT's world frame)


def _random_rotation_z(rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _random_color(rng: np.random.Generator, lo=0.1, hi=0.95) -> np.ndarray:
    # random hue with moderate saturation; avoids pure white/black
    h = rng.uniform(0, 1)
    s = rng.uniform(0.25, 0.9)
    v = rng.uniform(lo + 0.2, hi)
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return np.array(rgb, dtype=np.float32)


def random_texture(rng: np.random.Generator, n_faces: int, cfg: SceneConfig) -> Texture:
    K = cfg.n_waves
    dirs = rng.normal(size=(K, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = np.exp(rng.uniform(math.log(cfg.wave_freq[0]), math.log(cfg.wave_freq[1]), size=(K, 1)))
    freqs = (dirs * mags).astype(np.float32)
    amps = (1.0 / np.sqrt(mags[:, 0])).astype(np.float32)  # pinkish spectrum
    Kf = cfg.fine_waves
    fdirs = rng.normal(size=(3, Kf, 3))
    fdirs /= np.linalg.norm(fdirs, axis=-1, keepdims=True)
    fmags = rng.uniform(cfg.wave_freq[1] * 0.5, cfg.wave_freq[1] * 1.5, size=(3, Kf, 1))
    return Texture(
        colors_a=np.stack([_random_color(rng) for _ in range(n_faces)]),
        colors_b=np.stack([_random_color(rng) for _ in range(n_faces)]),
        freqs=freqs,
        phases=rng.uniform(0, 2 * math.pi, size=K).astype(np.float32),
        amps=amps,
        sharpness=float(rng.uniform(*cfg.sharpness)),
        fine_freqs=(fdirs * fmags).astype(np.float32),
        fine_phases=rng.uniform(0, 2 * math.pi, size=(3, Kf)).astype(np.float32),
        fine_amp=cfg.fine_amp,
    )


def random_scene(seed: int, cfg: SceneConfig | None = None) -> Scene:
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(seed)
    hx, hy = rng.uniform(*cfg.room_half_size, size=2)
    hz = rng.uniform(*cfg.room_height) / 2
    room = Primitive(
        kind=KIND_ROOM,
        center=np.array([0.0, 0.0, hz], dtype=np.float32),
        half_extents=np.array([hx, hy, hz], dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        texture=random_texture(rng, 6, cfg),
        name="room",
    )
    prims = [room]
    n_obj = int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1))
    placed: list[tuple[np.ndarray, float]] = []
    attempts = 0
    while len(prims) - 1 < n_obj and attempts < 500:
        attempts += 1
        r = cfg.object_radius * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        xy = np.array([r * math.cos(a), r * math.sin(a)])
        is_sphere = rng.uniform() < cfg.sphere_prob
        if is_sphere:
            rad = rng.uniform(*cfg.object_size)
            he = np.array([rad, rad, rad], dtype=np.float32)
            bound = rad
        else:
            he = rng.uniform(cfg.object_size[0], cfg.object_size[1], size=3).astype(np.float32)
            bound = float(np.linalg.norm(he[:2]))
        floating = rng.uniform() < cfg.floating_prob
        z = he[2] + (rng.uniform(0.3, 1.2) if floating else 0.0)
        center = np.array([xy[0], xy[1], z], dtype=np.float32)
        # avoid heavy interpenetration in xy (keeps scenes readable)
        if any(np.linalg.norm(center[:2] - c[:2]) < 0.8 * (bound + b) for c, b in placed):
            continue
        placed.append((center, bound))
        if is_sphere:
            rot = np.eye(3, dtype=np.float32)
        else:
            rot = _random_rotation(rng) if floating else _random_rotation_z(rng)
        prims.append(
            Primitive(
                kind=KIND_SPHERE if is_sphere else KIND_BOX,
                center=center,
                half_extents=he,
                rotation=rot,
                texture=random_texture(rng, NUM_FACES[KIND_SPHERE if is_sphere else KIND_BOX], cfg),
                name=f"obj{len(prims)}",
            )
        )
    light_dirs = np.array([[0.4, 0.3, 1.0], [-0.6, -0.5, 0.4], [0.2, -0.9, -0.3]], dtype=np.float32)
    light_dirs /= np.linalg.norm(light_dirs, axis=1, keepdims=True)
    return Scene(
        primitives=prims,
        light_dirs=light_dirs,
        light_strengths=np.array([0.55, 0.3, 0.15], dtype=np.float32),
        ambient=cfg.ambient,
        seed=seed,
    )


def random_cameras(seed: int, scene: Scene, cfg: CameraConfig | None = None) -> Cameras:
    cfg = cfg or CameraConfig()
    rng = np.random.default_rng(seed + 7919)
    S = cfg.n_views
    room = scene.room
    hx, hy, hz2 = room.half_extents
    base = rng.uniform(0, 360)
    if S == 1:
        angles = np.array([base])
    else:
        angles = base + np.linspace(-cfg.arc_deg / 2, cfg.arc_deg / 2, S)
    angles = angles + rng.uniform(-cfg.angle_jitter_deg, cfg.angle_jitter_deg, size=S)
    K = intrinsics_from_fov(cfg.width, cfg.height, cfg.fov_x_deg).numpy()
    Es, Ks = [], []
    for ang in angles:
        for _ in range(50):
            r = rng.uniform(*cfg.radius)
            a = math.radians(ang)
            eye = np.array([r * math.cos(a), r * math.sin(a), rng.uniform(*cfg.cam_height)], dtype=np.float32)
            # keep the camera inside the room with a margin
            eye[0] = np.clip(eye[0], -hx + 0.3, hx - 0.3)
            eye[1] = np.clip(eye[1], -hy + 0.3, hy - 0.3)
            eye[2] = np.clip(eye[2], 0.3, 2 * hz2 - 0.3)
            if not point_inside_any_object(scene, eye, margin=0.25):
                break
        target = np.array(
            [
                rng.uniform(-cfg.target_jitter, cfg.target_jitter),
                rng.uniform(-cfg.target_jitter, cfg.target_jitter),
                rng.uniform(*cfg.target_height),
            ],
            dtype=np.float32,
        )
        Es.append(look_at(torch.from_numpy(eye), torch.from_numpy(target)).numpy())
        Ks.append(K.copy())
    order = rng.permutation(S) if cfg.shuffle else np.arange(S)
    return Cameras(np.stack(Es)[order].astype(np.float32), np.stack(Ks)[order].astype(np.float32), cfg.height, cfg.width)


def point_inside_any_object(scene: Scene, p: np.ndarray, margin: float = 0.05) -> bool:
    for prim in scene.primitives[1:]:
        local = prim.rotation.T @ (p - prim.center)
        if prim.kind == KIND_SPHERE:
            if np.linalg.norm(local) < prim.radius + margin:
                return True
        elif np.all(np.abs(local) < prim.half_extents + margin):
            return True
    return False
