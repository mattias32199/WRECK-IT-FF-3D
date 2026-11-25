"""Static figures for Experiment 1 (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .metrics import SUPPORT_BINS, sparsification, support_label  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
# sequential blue steps for discrete support levels 0..4+
SUPPORT_CMAP = ListedColormap(["#efefef", "#b6d5f8", "#84b9f3", "#338ced", "#1757ac", "#0e3d7e"])


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)


def _bin_col(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    for lo, hi in SUPPORT_BINS:
        out[(df.support >= lo) & (df.support <= hi)] = support_label(lo, hi)
    return out


# --------------------------------------------------------------------------- #
def save_condition_preview(sample, cond, path: str | Path) -> Path:
    """Rows: model input, per-pixel support (0..4+), GT depth."""
    S = cond.images.shape[0]
    fig, axes = plt.subplots(3, S, figsize=(2.2 * S, 5.4), squeeze=False)
    gd = sample.render.depth[cond.view_indices]
    for j in range(S):
        axes[0, j].imshow(cond.images[j])
        axes[0, j].set_title(f"view {cond.view_indices[j]}", fontsize=8, color=TEXT2)
        im = axes[1, j].imshow(np.minimum(cond.support[j], 5), cmap=SUPPORT_CMAP, vmin=-0.5, vmax=5.5)
        axes[2, j].imshow(gd[j], cmap="Greys")
        for i in range(3):
            axes[i, j].axis("off")
    cb = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, ticks=range(6))
    cb.ax.set_yticklabels(["0", "1", "2", "3", "4", "5+"], fontsize=7)
    cb.set_label("support (views)", fontsize=8, color=TEXT2)
    fig.suptitle(f"{cond.scene_id} / {cond.name}: input · support · GT depth", fontsize=10, color=TEXT)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_conf_by_support(df: pd.DataFrame, conf_col: str = "depth_conf_pct", ax=None):
    ax = ax or plt.gca()
    b = _bin_col(df)
    labels = [support_label(lo, hi) for lo, hi in SUPPORT_BINS]
    groups = [(True, "visible in own view", SERIES[0]), (False, "masked in own view", SERIES[1])]
    w = 0.36
    for k, (sv, name, col) in enumerate(groups):
        data = [df.loc[(b == l) & (df.self_visible == sv), conf_col].to_numpy() for l in labels]
        pos = np.arange(len(labels)) + (k - 0.5) * w
        keep = [i for i, d in enumerate(data) if len(d)]
        if not keep:
            continue
        bp = ax.boxplot(
            [data[i] for i in keep], positions=pos[keep], widths=w * 0.85, patch_artist=True,
            showfliers=False, medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=col), capprops=dict(color=col),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=col, edgecolor=col)
        ax.plot([], [], color=col, linewidth=6, label=name)
    ax.set_xticks(range(len(labels)), labels)
    _style(ax, f"Confidence vs geometric support", "support (# input views that see the point unmasked)",
           "depth-conf percentile" if conf_col.endswith("pct") else conf_col)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    return ax


def plot_error_by_support(df: pd.DataFrame, err_col: str = "abs_rel", ax=None):
    ax = ax or plt.gca()
    b = _bin_col(df)
    labels = [support_label(lo, hi) for lo, hi in SUPPORT_BINS]
    w = 0.36
    for k, (sv, name, col) in enumerate([(True, "visible in own view", SERIES[0]), (False, "masked in own view", SERIES[1])]):
        med = [df.loc[(b == l) & (df.self_visible == sv), err_col].median() for l in labels]
        x = np.arange(len(labels)) + (k - 0.5) * w
        ax.bar(x, med, width=w - 0.04, color=col, label=name)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels)
    _style(ax, "Depth error vs geometric support", "support", f"median {err_col} (log)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    return ax


def plot_sparsification(df: pd.DataFrame, err_col="abs_rel", conf_col="depth_conf", ax=None):
    ax = ax or plt.gca()
    fr, curve, oracle, ause = sparsification(df[err_col].to_numpy(), df[conf_col].to_numpy())
    ax.plot(fr, curve, color=SERIES[0], linewidth=2, label=f"by VGGT confidence (AUSE {ause:.3f})")
    ax.plot(fr, oracle, color=TEXT2, linewidth=2, linestyle="--", label="oracle (by true error)")
    _style(ax, "Sparsification", "fraction of pixels removed", f"mean {err_col} of remaining")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    return ax


def plot_region_deltas(deltas: pd.DataFrame, ax=None):
    """Change in confidence percentile of masked regions vs the clean run, by target support."""
    ax = ax or plt.gca()
    rng = np.random.default_rng(0)
    for k, (sv, name, col) in enumerate([(True, "visible in own view", SERIES[0]), (False, "masked in own view", SERIES[1])]):
        sub = deltas[deltas.self_visible == sv]
        for t, g in sub.groupby("target_support"):
            x = t + (k - 0.5) * 0.3 + rng.uniform(-0.06, 0.06, len(g))
            ax.scatter(x, g.d_conf_pct, s=12, color=col, alpha=0.45, linewidths=0)
            ax.plot([t + (k - 0.5) * 0.3 - 0.12, t + (k - 0.5) * 0.3 + 0.12], [g.d_conf_pct.mean()] * 2, color=col, linewidth=2.5)
        ax.plot([], [], color=col, linewidth=2.5, label=f"{name} (bar = mean)")
    ax.axhline(0, color=TEXT2, linewidth=1)
    ax.set_xticks(sorted(deltas.target_support.unique()))
    _style(ax, "Region confidence change vs clean run", "target support of region", "Δ conf percentile")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    return ax


def make_report_figures(pix: pd.DataFrame, deltas: pd.DataFrame | None, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in [
        ("conf_by_support.png", lambda ax: plot_conf_by_support(pix, ax=ax)),
        ("error_by_support.png", lambda ax: plot_error_by_support(pix, ax=ax)),
        ("sparsification.png", lambda ax: plot_sparsification(pix, ax=ax)),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        fn(ax)
        fig.savefig(out_dir / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_dir / name)
    if deltas is not None and len(deltas):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        plot_region_deltas(deltas, ax=ax)
        fig.savefig(out_dir / "region_deltas.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_dir / "region_deltas.png")
    return paths
