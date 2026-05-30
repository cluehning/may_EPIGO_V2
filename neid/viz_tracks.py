from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt

from pathlib import Path
from .io import load_all_gz_tracks


from .tracks import (
    PairwiseMatrixResult,
    Track,
    TrackComparisonConfig,
    TrackComparisonResult,
    compare_tracks,
    pairwise_comparison_matrices,
)
from .viz import (
    plot_time_series,
    plot_power_spectrum,
    plot_pairwise_summary,
    plot_pairwise_matrix,
)

try:
    # Keep compatibility with the existing March-style loaders if present.
    from .io import load_track  # type: ignore
except Exception:  # pragma: no cover
    load_track = None


def load_tracks_from_data() -> list[Track]:
    data_dir = Path(__file__).resolve().parents[1] / "data"

    raw_tracks = load_all_gz_tracks(data_dir)

    tracks = [
        Track.from_array(values, name=name)
        for name, values in raw_tracks.items()
    ]

    return tracks


def run_full_analysis():
    """
    Automatically:
    - loads all .gz tracks
    - runs single + pairwise + batch analysis
    - produces plots
    """
    tracks = load_tracks_from_data()

    if not tracks:
        print("No tracks found in data/")
        return

    print(f"Loaded {len(tracks)} tracks")

    # ----------------------------
    # single-track plots
    # ----------------------------
    for t in tracks:
        print(f"Analyzing {t.name}")
        plot_time_series(t.values)
        plot_power_spectrum(t.values)

    # ----------------------------
    # pairwise (only if >=2 tracks)
    # ----------------------------
    if len(tracks) >= 2:
        result = tracks[0].compare_to(tracks[1])
        plot_pairwise_summary(result)

    # ----------------------------
    # multi-track matrices
    # ----------------------------
    matrices = pairwise_comparison_matrices(tracks)

    for m in matrices.values():
        plot_pairwise_matrix(m)

    print("Analysis complete.")


def analyze_single_track(
    track: Track,
    *,
    fs: float = 1.0,
    show: bool = False,
):
    """
    Existing-style one-track exploration, extended with localized spectrum.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plot_track_signal(track, ax=axes[0])
    plot_spectrum(track, fs=fs, ax=axes[1])
    stft_result = track.localized_spectrum(fs=fs)
    plot_localized_spectrum(stft_result, ax=axes[2])
    fig.tight_layout()
    if show:
        plt.show()
    return {
        "track": track,
        "stft": stft_result,
        "figure": fig,
        "axes": axes,
    }


def analyze_track_pair(
    track_a: Track,
    track_b: Track,
    *,
    config: Optional[TrackComparisonConfig] = None,
    show: bool = False,
) -> TrackComparisonResult:
    """
    Phase 2 pairwise comparison entry point.
    """
    result = compare_tracks(track_a, track_b, config=config)
    fig, _ = plot_pairwise_comparison_summary(result)
    if show:
        plt.show()
    result.metadata["figure"] = fig
    return result


def analyze_track_batch(
    tracks: Sequence[Track],
    *,
    config: Optional[TrackComparisonConfig] = None,
    include_mean_coherence: bool = True,
    show: bool = False,
) -> Dict[str, PairwiseMatrixResult]:
    """
    Phase 2 multi-track batch comparison entry point.
    """
    matrices = pairwise_comparison_matrices(
        tracks,
        config=config,
        include_mean_coherence=include_mean_coherence,
    )
    figs = {}
    for key, matrix_result in matrices.items():
        fig, _ = plot_metric_heatmap(matrix_result)
        figs[key] = fig
    if show:
        plt.show()
    return matrices

if __name__ == "__main__":
    run_full_analysis()
