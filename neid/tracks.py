from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .analysis import (
    CrossSpectralResult,
    PairwiseInfoResult,
    STFTResult,
    SignalSpectrum,
    WindowProfileResult,
    align_signals,
    cross_spectral_analysis,
    joint_information_analysis,
    localized_spectrum_stft,
    power_spectrum,
    shannon_entropy,
    sliding_window_coherence_summary,
    sliding_window_entropy,
    sliding_window_mutual_information,
    sliding_window_spectral_energy,
)


@dataclass
class Track:
    """
    EPIGO track abstraction.

    The goal is to treat a biological track as a structured object rather than
    passing raw arrays through the codebase. This keeps the Phase 2 extensions
    compatible with the documented EPIGO philosophy.
    """
    values: np.ndarray
    name: str = "track"
    coordinates: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates).reshape(-1)
            if len(self.coordinates) != len(self.values):
                raise ValueError("coordinates must have the same length as values")

    @classmethod
    def from_array(
        cls,
        values: Sequence[float],
        *,
        name: str = "track",
        coordinates: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Track":
        return cls(
            values=np.asarray(values, dtype=float),
            name=name,
            coordinates=None if coordinates is None else np.asarray(coordinates),
            metadata=dict(metadata or {}),
        )

    @property
    def n(self) -> int:
        return int(len(self.values))

    def entropy(
        self,
        *,
        bins: int = 32,
        quantile_bins: bool = False,
        epsilon: float = 1e-12,
        base: float = 2.0,
    ) -> float:
        value, _ = shannon_entropy(
            self.values,
            bins=bins,
            quantile_bins=quantile_bins,
            epsilon=epsilon,
            base=base,
        )
        return value

    def spectrum(
        self,
        *,
        fs: float = 1.0,
        method: str = "welch",
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> SignalSpectrum:
        return power_spectrum(
            self.values,
            fs=fs,
            method=method,
            nperseg=nperseg,
            noverlap=noverlap,
        )

    def localized_spectrum(
        self,
        *,
        fs: float = 1.0,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
    ) -> STFTResult:
        return localized_spectrum_stft(
            self.values,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )

    def compare_to(
        self,
        other: "Track",
        config: Optional["TrackComparisonConfig"] = None,
    ) -> "TrackComparisonResult":
        return compare_tracks(self, other, config=config)


@dataclass
class TrackComparisonConfig:
    """
    Configuration for Phase 2 pairwise/batch comparisons.
    """
    bins: int = 32
    quantile_bins: bool = False
    epsilon: float = 1e-12
    base: float = 2.0
    normalize_mutual_information: Optional[str] = None

    fs: float = 1.0
    spectrum_method: str = "welch"
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    detrend: str = "constant"
    window: str = "hann"

    window_size: int = 1024
    step_size: int = 256
    include_partial_windows: bool = False
    spectral_energy_band: Optional[Tuple[float, float]] = None
    coherence_summary: str = "mean"

    stft_nperseg: int = 256
    stft_noverlap: Optional[int] = None


@dataclass
class TrackComparisonResult:
    """
    High-level comparison result for two tracks.

    Designed to be easy to inspect, print, visualize, and reuse downstream.
    """
    track_a_name: str
    track_b_name: str
    aligned_length: int
    entropy_a: float
    entropy_b: float
    entropy_difference: float
    pairwise_information: PairwiseInfoResult
    spectral: CrossSpectralResult
    local_entropy_a: WindowProfileResult
    local_entropy_b: WindowProfileResult
    local_mutual_information: WindowProfileResult
    local_spectral_energy_a: WindowProfileResult
    local_spectral_energy_b: WindowProfileResult
    local_coherence: WindowProfileResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "track_a": self.track_a_name,
            "track_b": self.track_b_name,
            "aligned_length": self.aligned_length,
            "entropy_a": self.entropy_a,
            "entropy_b": self.entropy_b,
            "entropy_difference": self.entropy_difference,
            "joint_entropy": self.pairwise_information.joint_entropy,
            "mutual_information": self.pairwise_information.mutual_information,
            "normalized_mutual_information": self.pairwise_information.normalized_mutual_information,
            "js_divergence": self.pairwise_information.js_divergence,
            "js_distance": self.pairwise_information.js_distance,
            "kl_xy": self.pairwise_information.kl_xy,
            "kl_yx": self.pairwise_information.kl_yx,
            "max_coherence": self.spectral.max_coherence,
            "frequency_at_max_coherence": self.spectral.frequency_at_max_coherence,
            "lag_at_max_abs_correlation": self.spectral.lag_at_max_abs_correlation,
            "max_abs_correlation": self.spectral.max_abs_correlation,
        }


@dataclass
class PairwiseMatrixResult:
    """
    Symmetric pairwise metric matrix for multi-track batch mode.
    """
    metric_name: str
    labels: List[str]
    matrix: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def compare_tracks(
    track_a: Track,
    track_b: Track,
    *,
    config: Optional[TrackComparisonConfig] = None,
) -> TrackComparisonResult:
    cfg = config or TrackComparisonConfig()

    entropy_a = track_a.entropy(
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
    )
    entropy_b = track_b.entropy(
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
    )

    pairwise_information = joint_information_analysis(
        track_a.values,
        track_b.values,
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
        normalize_mutual_information=cfg.normalize_mutual_information,
    )

    spectral = cross_spectral_analysis(
        track_a.values,
        track_b.values,
        fs=cfg.fs,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        window=cfg.window,
        detrend=cfg.detrend,
    )

    local_entropy_a = sliding_window_entropy(
        track_a.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
        include_partial=cfg.include_partial_windows,
    )
    local_entropy_b = sliding_window_entropy(
        track_b.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
        include_partial=cfg.include_partial_windows,
    )
    local_mi = sliding_window_mutual_information(
        track_a.values,
        track_b.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        bins=cfg.bins,
        quantile_bins=cfg.quantile_bins,
        epsilon=cfg.epsilon,
        base=cfg.base,
        normalize_mutual_information=cfg.normalize_mutual_information,
        include_partial=cfg.include_partial_windows,
    )
    local_energy_a = sliding_window_spectral_energy(
        track_a.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        fs=cfg.fs,
        band=cfg.spectral_energy_band,
        nperseg=cfg.nperseg,
        include_partial=cfg.include_partial_windows,
    )
    local_energy_b = sliding_window_spectral_energy(
        track_b.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        fs=cfg.fs,
        band=cfg.spectral_energy_band,
        nperseg=cfg.nperseg,
        include_partial=cfg.include_partial_windows,
    )
    local_coh = sliding_window_coherence_summary(
        track_a.values,
        track_b.values,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        fs=cfg.fs,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        include_partial=cfg.include_partial_windows,
        summary=cfg.coherence_summary,
    )

    xa, ya, align_meta = align_signals(track_a.values, track_b.values)

    return TrackComparisonResult(
        track_a_name=track_a.name,
        track_b_name=track_b.name,
        aligned_length=len(xa),
        entropy_a=entropy_a,
        entropy_b=entropy_b,
        entropy_difference=entropy_a - entropy_b,
        pairwise_information=pairwise_information,
        spectral=spectral,
        local_entropy_a=local_entropy_a,
        local_entropy_b=local_entropy_b,
        local_mutual_information=local_mi,
        local_spectral_energy_a=local_energy_a,
        local_spectral_energy_b=local_energy_b,
        local_coherence=local_coh,
        metadata={
            "config": cfg,
            "alignment": align_meta,
            "track_a_metadata": dict(track_a.metadata),
            "track_b_metadata": dict(track_b.metadata),
        },
    )


def _matrix_from_tracks(
    tracks: Sequence[Track],
    *,
    metric_name: str,
    metric_fn,
) -> PairwiseMatrixResult:
    n = len(tracks)
    labels = [t.name for t in tracks]
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        matrix[i, i] = metric_fn(tracks[i], tracks[i])
        for j in range(i + 1, n):
            value = metric_fn(tracks[i], tracks[j])
            matrix[i, j] = value
            matrix[j, i] = value

    return PairwiseMatrixResult(
        metric_name=metric_name,
        labels=labels,
        matrix=matrix,
    )


def pairwise_comparison_matrices(
    tracks: Sequence[Track],
    *,
    config: Optional[TrackComparisonConfig] = None,
    include_mean_coherence: bool = True,
) -> Dict[str, PairwiseMatrixResult]:
    """
    Compute symmetric pairwise matrices for more than two tracks.

    Metrics included
    ----------------
    - mutual information (or normalized MI if configured),
    - Jensen-Shannon distance,
    - entropy difference magnitude,
    - mean coherence summary (optional).
    """
    cfg = config or TrackComparisonConfig()

    def mi_metric(a: Track, b: Track) -> float:
        info = joint_information_analysis(
            a.values,
            b.values,
            bins=cfg.bins,
            quantile_bins=cfg.quantile_bins,
            epsilon=cfg.epsilon,
            base=cfg.base,
            normalize_mutual_information=cfg.normalize_mutual_information,
        )
        if cfg.normalize_mutual_information is not None and info.normalized_mutual_information is not None:
            return float(info.normalized_mutual_information)
        return float(info.mutual_information)

    def js_metric(a: Track, b: Track) -> float:
        info = joint_information_analysis(
            a.values,
            b.values,
            bins=cfg.bins,
            quantile_bins=cfg.quantile_bins,
            epsilon=cfg.epsilon,
            base=cfg.base,
        )
        return float(info.js_distance)

    def entropy_diff_metric(a: Track, b: Track) -> float:
        ha = a.entropy(
            bins=cfg.bins,
            quantile_bins=cfg.quantile_bins,
            epsilon=cfg.epsilon,
            base=cfg.base,
        )
        hb = b.entropy(
            bins=cfg.bins,
            quantile_bins=cfg.quantile_bins,
            epsilon=cfg.epsilon,
            base=cfg.base,
        )
        return abs(float(ha - hb))

    results: Dict[str, PairwiseMatrixResult] = {
        "mutual_information": _matrix_from_tracks(tracks, metric_name="mutual_information", metric_fn=mi_metric),
        "jensen_shannon_distance": _matrix_from_tracks(tracks, metric_name="jensen_shannon_distance", metric_fn=js_metric),
        "entropy_difference": _matrix_from_tracks(tracks, metric_name="entropy_difference", metric_fn=entropy_diff_metric),
    }

    if include_mean_coherence:
        def coherence_metric(a: Track, b: Track) -> float:
            spectral = cross_spectral_analysis(
                a.values,
                b.values,
                fs=cfg.fs,
                nperseg=cfg.nperseg,
                noverlap=cfg.noverlap,
                window=cfg.window,
                detrend=cfg.detrend,
            )
            if spectral.coherence.size == 0:
                return np.nan
            return float(np.nanmean(spectral.coherence))

        results["mean_coherence"] = _matrix_from_tracks(
            tracks,
            metric_name="mean_coherence",
            metric_fn=coherence_metric,
        )

    return results
