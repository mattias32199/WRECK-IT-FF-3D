"""Experiment configuration: nested dataclasses loadable from YAML."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .dataset import ConditionConfig
from .metrics import EvalConfig
from .occlusion import OcclusionConfig
from .scene import CameraConfig, SceneConfig


@dataclass
class RenderConfig:
    ssaa: int = 2
    device: str = "cpu"  # rendering/support device ("cuda" speeds up generation)


@dataclass
class ModelConfig:
    weights: str = "facebook/VGGT-1B"
    checkpoint: str | None = None  # local model.pt (offline clusters)
    device: str | None = None  # default: cuda if available


@dataclass
class OutputConfig:
    data_dir: str = "data/synthetic"
    results_dir: str = "results/exp1"
    save_raw_predictions: bool = False
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    name: str = "exp1_confidence_probe"
    n_scenes: int = 50
    seed_offset: int = 0
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    conditions: ConditionConfig = field(default_factory=ConditionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_offset, self.seed_offset + self.n_scenes))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, data):
    """Recursively build dataclass `cls` from a dict, converting lists to tuples where typed so."""
    if data is None:
        return cls()
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs = {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        t = hints[f.name]
        if dataclasses.is_dataclass(t):
            kwargs[f.name] = _build(t, v)
        elif typing.get_origin(t) is tuple and isinstance(v, list):
            kwargs[f.name] = tuple(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> ExperimentConfig:
    data = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for k, v in (overrides or {}).items():  # dotted keys: "conditions.occlusion.fill": "noise"
        node = data
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return _build(ExperimentConfig, data)
