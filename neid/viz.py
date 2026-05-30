from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .analysis import (
    build_feature_matrix,
    normalize_matrix,
    shannon_entropy,
    spectrum_1d,
)

from .analysis import (
    CrossSpectralResult,
    WindowProfileResult,
    STFTResult,
)
from .tracks import PairwiseMatrixResult, TrackComparisonResult

from .io import ensure_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Core utilities (UNCHANGED)
# ---------------------------------------------------------------------

def _plt():
    """Lazy matplotlib import with clear error."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: python -m pip install matplotlib"
        ) from exc
    return plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def out_dir() -> Path:
    out = project_root() / "out"
    ensure_dir(out)
    return out


# ---------------------------------------------------------------------
# Phase 1 plots (UNCHANGED)
# ---------------------------------------------------------------------

def plot_time_series(x: np.ndarray) -> Path:
    plt = _plt()
    path = out_dir() / "time_series.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, lw=1.2)
    ax.set_title("Time Series")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_power_spectrum(x: np.ndarray) -> Path:
    plt = _plt()
    path = out_dir() / "power_spectrum.png"

    freqs, power = spectrum_1d(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, power, lw=1.2)
    ax.set_yscale("log")

    ax.set_title("Power Spectrum (log scale)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_entropy_over_windows(x: np.ndarray) -> Path:
    plt = _plt()
    path = out_dir() / "entropy_vs_window.png"

    window_sizes = np.unique(
        np.logspace(1, np.log10(len(x) // 4), num=20).astype(int)
    )

    entropies: list[float] = []
    for w in window_sizes:
        chunks = [x[i:i + w] for i in range(0, len(x) - w + 1, w)]
        if not chunks:
            continue
        counts = np.array([np.histogram(c, bins=32)[0] for c in chunks])
        e = np.mean(shannon_entropy(counts, axis=1))
        entropies.append(e)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window_sizes[:len(entropies)], entropies, marker="o")

    ax.set_xscale("log")
    ax.set_title("Shannon Entropy vs Window Size")
    ax.set_xlabel("Window size (log)")
    ax.set_ylabel("Entropy (bits)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_feature_heatmap() -> Path:
    plt = _plt()
    path = out_dir() / "feature_heatmap.png"

    rng = np.random.default_rng(0)
    rows = [
        {
            "mean": float(np.mean(rng.normal(size=200))),
            "std": float(np.std(rng.normal(size=200))),
            "entropy": float(
                shannon_entropy(
                    np.histogram(rng.normal(size=200), bins=32)[0]
                )
            ),
        }
        for _ in range(20)
    ]

    fm = build_feature_matrix(rows)
    Xn = normalize_matrix(fm.X, method="zscore", axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Xn, aspect="auto", cmap="viridis")

    ax.set_title("Normalized Feature Matrix")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample")

    ax.set_xticks(range(len(fm.feature_names)))
    ax.set_xticklabels(fm.feature_names, rotation=45, ha="right")

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------
# Phase 2: Pairwise + localized plots
# ---------------------------------------------------------------------

def plot_coherence(result: CrossSpectralResult) -> Path:
    plt = _plt()
    path = out_dir() / "coherence.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.frequencies, result.coherence, lw=1.2)

    ax.set_title("Coherence vs Frequency")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Coherence")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_cross_spectrum(result: CrossSpectralResult) -> Path:
    plt = _plt()
    path = out_dir() / "cross_spectrum.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.frequencies, np.abs(result.cross_spectrum), lw=1.2)

    ax.set_title("Cross-Spectral Magnitude")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("|Pxy|")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_window_profile(profile: WindowProfileResult) -> Path:
    plt = _plt()
    path = out_dir() / f"{profile.metric_name}.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(profile.centers, profile.values, lw=1.3)

    ax.set_title(profile.metric_name.replace("_", " ").title())
    ax.set_xlabel("Genomic position")
    ax.set_ylabel(profile.metric_name)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_pairwise_matrix(matrix: PairwiseMatrixResult) -> Path:
    plt = _plt()
    path = out_dir() / f"{matrix.metric_name}_matrix.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix.matrix, cmap="viridis", aspect="auto")

    ax.set_title(matrix.metric_name.replace("_", " ").title())

    ax.set_xticks(range(len(matrix.labels)))
    ax.set_xticklabels(matrix.labels, rotation=45, ha="right")

    ax.set_yticks(range(len(matrix.labels)))
    ax.set_yticklabels(matrix.labels)

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_stft(result: STFTResult) -> Path:
    plt = _plt()
    path = out_dir() / "stft.png"

    fig, ax = plt.subplots(figsize=(9, 4))

    mesh = ax.pcolormesh(
        result.positions,
        result.frequencies,
        result.power,
        shading="auto",
        cmap="magma"
    )

    ax.set_title("Localized Spectrum (STFT)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")

    fig.colorbar(mesh, ax=ax, label="Power")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_pairwise_summary(result: TrackComparisonResult) -> None:
    """
    Generate a full set of comparison plots for two tracks.
    """
    plot_coherence(result.spectral)
    plot_cross_spectrum(result.spectral)

    plot_window_profile(result.local_mutual_information)
    plot_window_profile(result.local_coherence)

    plot_window_profile(result.local_entropy_a)
    plot_window_profile(result.local_entropy_b)
