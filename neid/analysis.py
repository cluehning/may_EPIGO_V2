from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal


@dataclass
class FeatureMatrix:
    """
    Simple container for feature matrices.
    """
    X: np.ndarray
    feature_names: List[str]


def build_feature_matrix(rows: List[Mapping[str, Any]]) -> FeatureMatrix:
    """
    Convert list of dicts → matrix.

    Example:
        [{"mean": 1, "std": 2}, {"mean": 3, "std": 4}]
    """
    if not rows:
        raise ValueError("No rows provided")

    feature_names = sorted(rows[0].keys())

    X = np.array([
        [float(row[f]) for f in feature_names]
        for row in rows
    ])

    return FeatureMatrix(X=X, feature_names=feature_names)


def normalize_matrix(
    X: np.ndarray,
    *,
    method: str = "zscore",
    axis: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize feature matrix.

    Supported methods:
    - 'zscore'
    - 'minmax'
    """
    X = np.asarray(X, dtype=float)

    if method == "zscore":
        mean = np.mean(X, axis=axis, keepdims=True)
        std = np.std(X, axis=axis, keepdims=True)
        std = np.maximum(std, eps)
        return (X - mean) / std

    if method == "minmax":
        lo = np.min(X, axis=axis, keepdims=True)
        hi = np.max(X, axis=axis, keepdims=True)
        denom = np.maximum(hi - lo, eps)
        return (X - lo) / denom

    raise ValueError(f"Unknown normalization method: {method}")

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]

_LOG_BASES = {
    "e": np.e,
    "2": 2.0,
    "10": 10.0,
}


@dataclass
class ProbabilityEstimate:
    """
    Discrete probability estimate derived from a real-valued signal.

    This object stores a histogram/discretization-based empirical distribution.
    It is useful because Phase 2 comparisons (entropy, mutual information,
    Jensen-Shannon divergence) are defined on discrete probability mass functions.

    Assumptions
    -----------
    - The input signal can be meaningfully discretized into bins.
    - Missing and non-finite values have already been filtered or are removable.
    - This is an empirical approximation, not a parametric distribution model.

    Numerical caveats
    -----------------
    - Results depend on the chosen binning strategy.
    - Very small sample sizes can produce unstable estimates.
    - Epsilon smoothing is applied to avoid log(0) and zero-probability issues.
    """
    probabilities: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    method: str
    epsilon: float
    sample_count: int
    dropped_count: int
    is_constant: bool
    data_min: float
    data_max: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseInfoResult:
    """
    Joint information-theoretic comparison for two discretized signals.

    All quantities are computed from empirical histogram-based estimates of the
    marginal and joint distributions. This keeps the method explicit and
    interpretable for exploratory signal analysis.

    Numerical caveats
    -----------------
    - Mutual information is estimated from discretized variables, not from a
      continuous estimator.
    - Values can shift with the number of bins and binning mode.
    """
    entropy_x: float
    entropy_y: float
    joint_entropy: float
    conditional_entropy_x_given_y: float
    conditional_entropy_y_given_x: float
    mutual_information: float
    normalized_mutual_information: Optional[float]
    kl_xy: float
    kl_yx: float
    js_divergence: float
    js_distance: float
    sample_count: int
    x_bin_edges: np.ndarray
    y_bin_edges: np.ndarray
    method: str
    epsilon: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalSpectrum:
    """
    One-track spectral summary.

    The spectrum is useful for describing spatial organization across scales.
    """
    frequencies: np.ndarray
    power: np.ndarray
    method: str
    fs: float
    nperseg: Optional[int]
    noverlap: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossSpectralResult:
    """
    Pairwise spectral relationship analysis between two tracks.

    Contains cross spectral density, magnitude-squared coherence, and spatial
    cross-correlation summaries so that both frequency-domain and coordinate-domain
    relationships remain inspectable.
    """
    frequencies: np.ndarray
    cross_spectrum: np.ndarray
    coherence: np.ndarray
    lags: np.ndarray
    cross_correlation: np.ndarray
    lag_at_max_abs_correlation: Optional[int]
    max_abs_correlation: Optional[float]
    max_coherence: Optional[float]
    frequency_at_max_coherence: Optional[float]
    fs: float
    nperseg: Optional[int]
    noverlap: Optional[int]
    window: str
    detrend: Union[str, bool]
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowProfileResult:
    """
    Genomic-position-dependent profile of a single scalar metric.

    This supports localized analysis such as sliding-window entropy or
    sliding-window mutual information.
    """
    starts: np.ndarray
    stops: np.ndarray
    centers: np.ndarray
    values: np.ndarray
    metric_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STFTResult:
    """
    Localized frequency representation from a short-time Fourier transform.

    STFT is useful when a track is not stationary across the genomic axis and
    one wants to inspect how dominant scales evolve by position.

    Numerical caveats
    -----------------
    - Time/frequency resolution is controlled by the window length.
    - Short windows improve localization in coordinate but reduce frequency
      resolution, and vice versa.
    """
    frequencies: np.ndarray
    positions: np.ndarray
    stft: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    fs: float
    nperseg: int
    noverlap: int
    window: str
    detrend: Union[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_1d_float_array(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(min(len(a) for a in arrays), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr[: len(mask)])
    return mask


def align_signals(
    x: ArrayLike,
    y: ArrayLike,
    *,
    drop_nonfinite: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Align two one-dimensional signals by trimming to common minimum length.

    This explicit policy avoids silent misalignment for unequal-length inputs.
    """
    xa = _as_1d_float_array(x)
    ya = _as_1d_float_array(y)
    original_lengths = (len(xa), len(ya))
    n = min(len(xa), len(ya))
    xa = xa[:n]
    ya = ya[:n]

    if drop_nonfinite:
        mask = _finite_mask(xa, ya)
        dropped = int(n - mask.sum())
        xa = xa[mask]
        ya = ya[mask]
    else:
        dropped = 0

    meta = {
        "original_lengths": original_lengths,
        "aligned_length_before_mask": n,
        "aligned_length_after_mask": len(xa),
        "dropped_nonfinite": dropped,
        "alignment_policy": "trim_to_common_min_length",
    }
    return xa, ya, meta


def clean_signal(values: ArrayLike) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = _as_1d_float_array(values)
    mask = np.isfinite(arr)
    meta = {
        "original_length": len(arr),
        "clean_length": int(mask.sum()),
        "dropped_nonfinite": int((~mask).sum()),
    }
    return arr[mask], meta


def zscore_signal(values: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """
    Z-score normalize a signal.

    Useful for localized spectral/coherence analysis when relative fluctuations
    matter more than absolute magnitude.
    """
    arr, _ = clean_signal(values)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < eps:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _safe_edges_for_constant(value: float) -> np.ndarray:
    delta = max(abs(value) * 1e-6, 1e-6)
    return np.array([value - delta, value + delta], dtype=float)


def compute_bin_edges(
    values: ArrayLike,
    *,
    bins: Union[int, Sequence[float]] = 32,
    bin_edges: Optional[Sequence[float]] = None,
    quantile_bins: bool = False,
    data_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute histogram bin edges for empirical discretization.

    Supports:
    - fixed number of bins,
    - explicitly supplied edges,
    - quantile-based binning for approximately balanced occupancy.
    """
    arr, _ = clean_signal(values)
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional array of length >= 2.")
        return edges

    if np.allclose(arr, arr[0]):
        return _safe_edges_for_constant(float(arr[0]))

    if isinstance(bins, Sequence) and not isinstance(bins, (str, bytes)):
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Custom bins sequence must contain at least two edges.")
        return edges

    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be >= 1")

    lo = float(arr.min()) if data_range is None else float(data_range[0])
    hi = float(arr.max()) if data_range is None else float(data_range[1])

    if np.isclose(lo, hi):
        return _safe_edges_for_constant(lo)

    if quantile_bins:
        q = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(arr, q)
        edges = np.unique(edges)
        if edges.size < 2:
            return _safe_edges_for_constant(float(arr[0]))
        return edges

    return np.linspace(lo, hi, bins + 1, dtype=float)


def estimate_probability(
    values: ArrayLike,
    *,
    bins: Union[int, Sequence[float]] = 32,
    bin_edges: Optional[Sequence[float]] = None,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
    data_range: Optional[Tuple[float, float]] = None,
) -> ProbabilityEstimate:
    """
    Estimate a discrete empirical distribution by histogram binning.

    This is the basic building block for entropy and divergence-based EPIGO
    comparisons. Raw signal normalization and probability normalization are kept
    separate by design.

    Numerical caveats
    -----------------
    - Histogram estimates are sensitive to bin width/placement.
    - Quantile bins reduce occupancy sparsity but sacrifice equal-width bins.
    - Smoothing is applied after counting, then renormalized.
    """
    arr = _as_1d_float_array(values)
    finite_mask = np.isfinite(arr)
    clean = arr[finite_mask]
    dropped_count = int((~finite_mask).sum())

    if clean.size == 0:
        edges = np.array([0.0, 1.0], dtype=float)
        counts = np.zeros(1, dtype=float)
        probs = np.array([1.0], dtype=float)
        return ProbabilityEstimate(
            probabilities=probs,
            counts=counts,
            bin_edges=edges,
            method="histogram",
            epsilon=epsilon,
            sample_count=0,
            dropped_count=dropped_count,
            is_constant=False,
            data_min=np.nan,
            data_max=np.nan,
            metadata={"empty_input": True},
        )

    edges = compute_bin_edges(
        clean,
        bins=bins,
        bin_edges=bin_edges,
        quantile_bins=quantile_bins,
        data_range=data_range,
    )

    counts, edges = np.histogram(clean, bins=edges)
    counts = counts.astype(float)

    smoothed = counts + float(epsilon)
    probs = smoothed / smoothed.sum()

    is_constant = bool(np.allclose(clean, clean[0]))

    return ProbabilityEstimate(
        probabilities=probs,
        counts=counts,
        bin_edges=edges,
        method="histogram_quantile" if quantile_bins else "histogram",
        epsilon=float(epsilon),
        sample_count=int(clean.size),
        dropped_count=dropped_count,
        is_constant=is_constant,
        data_min=float(clean.min()),
        data_max=float(clean.max()),
        metadata={
            "bin_count": int(len(edges) - 1),
            "quantile_bins": bool(quantile_bins),
        },
    )


def _log(values: np.ndarray, base: float = 2.0) -> np.ndarray:
    if base == np.e:
        return np.log(values)
    return np.log(values) / np.log(base)


def entropy_from_probabilities(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * _log(p, base=base)))


def shannon_entropy(
    values: ArrayLike,
    *,
    bins: Union[int, Sequence[float]] = 32,
    bin_edges: Optional[Sequence[float]] = None,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
    base: float = 2.0,
) -> Tuple[float, ProbabilityEstimate]:
    est = estimate_probability(
        values,
        bins=bins,
        bin_edges=bin_edges,
        quantile_bins=quantile_bins,
        epsilon=epsilon,
    )
    return entropy_from_probabilities(est.probabilities, base=base), est


def _joint_histogram_probability(
    x: ArrayLike,
    y: ArrayLike,
    *,
    bins: Union[int, Tuple[int, int], Sequence[float]] = 32,
    x_bin_edges: Optional[Sequence[float]] = None,
    y_bin_edges: Optional[Sequence[float]] = None,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    xa, ya, meta = align_signals(x, y, drop_nonfinite=True)

    if xa.size == 0:
        pxy = np.array([[1.0]])
        ex = np.array([0.0, 1.0])
        ey = np.array([0.0, 1.0])
        return pxy, ex, ey, np.array([1.0]), np.array([1.0]), meta

    if isinstance(bins, tuple):
        bins_x, bins_y = int(bins[0]), int(bins[1])
    else:
        bins_x = bins_y = bins

    ex = compute_bin_edges(
        xa,
        bins=bins_x,
        bin_edges=x_bin_edges,
        quantile_bins=quantile_bins,
    )
    ey = compute_bin_edges(
        ya,
        bins=bins_y,
        bin_edges=y_bin_edges,
        quantile_bins=quantile_bins,
    )

    hist2d, ex, ey = np.histogram2d(xa, ya, bins=[ex, ey])
    hist2d = hist2d.astype(float)
    hist2d += float(epsilon)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    return pxy, ex, ey, px, py, meta


def joint_information_analysis(
    x: ArrayLike,
    y: ArrayLike,
    *,
    bins: Union[int, Tuple[int, int], Sequence[float]] = 32,
    x_bin_edges: Optional[Sequence[float]] = None,
    y_bin_edges: Optional[Sequence[float]] = None,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
    base: float = 2.0,
    normalize_mutual_information: Optional[str] = None,
) -> PairwiseInfoResult:
    """
    Compute joint entropy, conditional entropy, mutual information, KL, and JS.

    Mutual information is estimated from discretized variables derived by
    histogram or quantile-based binning.
    """
    pxy, x_edges, y_edges, px, py, align_meta = _joint_histogram_probability(
        x,
        y,
        bins=bins,
        x_bin_edges=x_bin_edges,
        y_bin_edges=y_bin_edges,
        quantile_bins=quantile_bins,
        epsilon=epsilon,
    )

    hx = entropy_from_probabilities(px, base=base)
    hy = entropy_from_probabilities(py, base=base)
    hxy = entropy_from_probabilities(pxy.ravel(), base=base)

    h_x_given_y = max(hxy - hy, 0.0)
    h_y_given_x = max(hxy - hx, 0.0)

    independent = np.outer(px, py)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * (_log(pxy[nz], base=base) - _log(independent[nz], base=base))))
    mi = max(mi, 0.0)

    norm_mi: Optional[float]
    if normalize_mutual_information is None:
        norm_mi = None
    elif normalize_mutual_information == "min":
        denom = max(min(hx, hy), epsilon)
        norm_mi = mi / denom
    elif normalize_mutual_information == "max":
        denom = max(max(hx, hy), epsilon)
        norm_mi = mi / denom
    elif normalize_mutual_information == "sqrt":
        denom = max(np.sqrt(max(hx * hy, 0.0)), epsilon)
        norm_mi = mi / float(denom)
    elif normalize_mutual_information == "sum":
        denom = max(0.5 * (hx + hy), epsilon)
        norm_mi = mi / denom
    else:
        raise ValueError("Unsupported normalization mode for mutual information.")

    kl_xy = kl_divergence(px, py, base=base, epsilon=epsilon)
    kl_yx = kl_divergence(py, px, base=base, epsilon=epsilon)
    js_div = jensen_shannon_divergence(px, py, base=base, epsilon=epsilon)
    js_dist = float(np.sqrt(max(js_div, 0.0)))

    return PairwiseInfoResult(
        entropy_x=hx,
        entropy_y=hy,
        joint_entropy=hxy,
        conditional_entropy_x_given_y=h_x_given_y,
        conditional_entropy_y_given_x=h_y_given_x,
        mutual_information=mi,
        normalized_mutual_information=norm_mi,
        kl_xy=kl_xy,
        kl_yx=kl_yx,
        js_divergence=js_div,
        js_distance=js_dist,
        sample_count=int(align_meta["aligned_length_after_mask"]),
        x_bin_edges=x_edges,
        y_bin_edges=y_edges,
        method="discrete_histogram_joint_estimate",
        epsilon=float(epsilon),
        metadata={
            **align_meta,
            "quantile_bins": bool(quantile_bins),
            "mi_normalization": normalize_mutual_information,
        },
    )


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    base: float = 2.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Safe KL divergence with smoothing.

    KL is included as an optional asymmetric diagnostic, but it is intentionally
    not the default comparison metric because it can be unstable and asymmetric.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (_log(p, base=base) - _log(q, base=base))))


def jensen_shannon_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    base: float = 2.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Jensen-Shannon divergence between two discrete probability vectors.

    JSD is symmetric and finite under smoothing, which makes it more stable than
    KL for exploratory signal comparisons.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base=base, epsilon=epsilon) + 0.5 * kl_divergence(
        q, m, base=base, epsilon=epsilon
    )


def power_spectrum(
    values: ArrayLike,
    *,
    fs: float = 1.0,
    method: str = "welch",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
) -> SignalSpectrum:
    """
    Compute a one-track power spectrum.

    Welch's method is used by default because EPIGO already emphasizes spectra
    and Welch gives a stable summary for exploratory analysis.
    """
    arr, clean_meta = clean_signal(values)
    if arr.size == 0:
        return SignalSpectrum(
            frequencies=np.array([]),
            power=np.array([]),
            method=method,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            metadata={"empty_input": True, **clean_meta},
        )

    if method == "welch":
        f, pxx = signal.welch(
            arr,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
        )
    elif method == "periodogram":
        f, pxx = signal.periodogram(arr, fs=fs, detrend=detrend)
    else:
        raise ValueError("Unsupported spectrum method. Use 'welch' or 'periodogram'.")

    return SignalSpectrum(
        frequencies=f,
        power=pxx,
        method=method,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        metadata=clean_meta,
    )


def cross_spectral_analysis(
    x: ArrayLike,
    y: ArrayLike,
    *,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
) -> CrossSpectralResult:
    """
    Pairwise spectral relationship analysis between two tracks.

    Computes:
    - cross spectral density,
    - magnitude-squared coherence,
    - normalized spatial cross-correlation,
    - optional lag estimate for strongest absolute correlation.

    Why useful
    ----------
    Coherence describes shared organization by frequency/scale, while
    cross-correlation reveals coordinate-domain alignment and lag structure.
    """
    xa, ya, meta = align_signals(x, y, drop_nonfinite=True)

    if xa.size == 0:
        empty = np.array([])
        return CrossSpectralResult(
            frequencies=empty,
            cross_spectrum=np.array([], dtype=complex),
            coherence=empty,
            lags=np.array([], dtype=int),
            cross_correlation=empty,
            lag_at_max_abs_correlation=None,
            max_abs_correlation=None,
            max_coherence=None,
            frequency_at_max_coherence=None,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            detrend=detrend,
            sample_count=0,
            metadata={"empty_input": True, **meta},
        )

    f_csd, pxy = signal.csd(
        xa,
        ya,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
    )
    f_coh, coh = signal.coherence(
        xa,
        ya,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )

    xz = xa - np.mean(xa)
    yz = ya - np.mean(ya)
    denom = float(np.linalg.norm(xz) * np.linalg.norm(yz))
    corr = signal.correlate(xz, yz, mode="full")
    if denom > 0:
        corr = corr / denom
    lags = signal.correlation_lags(len(xa), len(ya), mode="full")

    if corr.size > 0:
        idx_corr = int(np.argmax(np.abs(corr)))
        lag_best = int(lags[idx_corr])
        max_abs_corr = float(corr[idx_corr])
    else:
        lag_best = None
        max_abs_corr = None

    if coh.size > 0:
        idx_coh = int(np.argmax(coh))
        max_coh = float(coh[idx_coh])
        f_at_max = float(f_coh[idx_coh])
    else:
        max_coh = None
        f_at_max = None

    return CrossSpectralResult(
        frequencies=f_csd,
        cross_spectrum=pxy,
        coherence=coh,
        lags=lags,
        cross_correlation=corr,
        lag_at_max_abs_correlation=lag_best,
        max_abs_correlation=max_abs_corr,
        max_coherence=max_coh,
        frequency_at_max_coherence=f_at_max,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        sample_count=int(len(xa)),
        metadata=meta,
    )


def _window_slices(
    n: int,
    *,
    window_size: int,
    step_size: int,
    include_partial: bool = False,
) -> List[Tuple[int, int]]:
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive integers.")
    slices: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        stop = start + window_size
        if stop > n and not include_partial:
            break
        stop = min(stop, n)
        if stop <= start:
            break
        slices.append((start, stop))
        start += step_size
    return slices


def sliding_window_entropy(
    values: ArrayLike,
    *,
    window_size: int,
    step_size: int,
    bins: Union[int, Sequence[float]] = 32,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
    base: float = 2.0,
    include_partial: bool = False,
) -> WindowProfileResult:
    """
    Sliding-window Shannon entropy profile along the genomic coordinate axis.
    """
    arr = _as_1d_float_array(values)
    windows = _window_slices(len(arr), window_size=window_size, step_size=step_size, include_partial=include_partial)

    starts, stops, centers, vals = [], [], [], []
    for start, stop in windows:
        segment = arr[start:stop]
        h, _ = shannon_entropy(
            segment,
            bins=bins,
            quantile_bins=quantile_bins,
            epsilon=epsilon,
            base=base,
        )
        starts.append(start)
        stops.append(stop)
        centers.append((start + stop - 1) / 2.0)
        vals.append(h)

    return WindowProfileResult(
        starts=np.asarray(starts, dtype=int),
        stops=np.asarray(stops, dtype=int),
        centers=np.asarray(centers, dtype=float),
        values=np.asarray(vals, dtype=float),
        metric_name="sliding_entropy",
        parameters={
            "window_size": window_size,
            "step_size": step_size,
            "bins": bins,
            "quantile_bins": quantile_bins,
            "epsilon": epsilon,
            "base": base,
            "include_partial": include_partial,
        },
    )


def sliding_window_mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    *,
    window_size: int,
    step_size: int,
    bins: Union[int, Tuple[int, int], Sequence[float]] = 32,
    quantile_bins: bool = False,
    epsilon: float = 1e-12,
    base: float = 2.0,
    normalize_mutual_information: Optional[str] = None,
    include_partial: bool = False,
) -> WindowProfileResult:
    """
    Sliding-window mutual information between two tracks.

    MI is computed from discretized variables within each local window, allowing
    users to ask where two tracks appear most interdependent.
    """
    xa, ya, _ = align_signals(x, y, drop_nonfinite=False)
    windows = _window_slices(len(xa), window_size=window_size, step_size=step_size, include_partial=include_partial)

    starts, stops, centers, vals = [], [], [], []
    for start, stop in windows:
        local = joint_information_analysis(
            xa[start:stop],
            ya[start:stop],
            bins=bins,
            quantile_bins=quantile_bins,
            epsilon=epsilon,
            base=base,
            normalize_mutual_information=normalize_mutual_information,
        )
        value = (
            local.normalized_mutual_information
            if normalize_mutual_information is not None
            else local.mutual_information
        )
        starts.append(start)
        stops.append(stop)
        centers.append((start + stop - 1) / 2.0)
        vals.append(value)

    return WindowProfileResult(
        starts=np.asarray(starts, dtype=int),
        stops=np.asarray(stops, dtype=int),
        centers=np.asarray(centers, dtype=float),
        values=np.asarray(vals, dtype=float),
        metric_name=(
            "sliding_normalized_mutual_information"
            if normalize_mutual_information is not None
            else "sliding_mutual_information"
        ),
        parameters={
            "window_size": window_size,
            "step_size": step_size,
            "bins": bins,
            "quantile_bins": quantile_bins,
            "epsilon": epsilon,
            "base": base,
            "include_partial": include_partial,
            "mi_normalization": normalize_mutual_information,
        },
    )


def _band_energy_from_spectrum(frequencies: np.ndarray, power: np.ndarray, band: Optional[Tuple[float, float]]) -> float:
    if frequencies.size == 0 or power.size == 0:
        return 0.0
    if band is None:
        return float(np.trapezoid(power, frequencies))
    lo, hi = band
    mask = (frequencies >= lo) & (frequencies <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], frequencies[mask]))


def sliding_window_spectral_energy(
    values: ArrayLike,
    *,
    window_size: int,
    step_size: int,
    fs: float = 1.0,
    band: Optional[Tuple[float, float]] = None,
    nperseg: Optional[int] = None,
    include_partial: bool = False,
) -> WindowProfileResult:
    """
    Sliding-window spectral energy summary.

    This provides a compact local frequency summary suitable for position-dependent
    profiles and complements global spectra already present in EPIGO.
    """
    arr = _as_1d_float_array(values)
    windows = _window_slices(len(arr), window_size=window_size, step_size=step_size, include_partial=include_partial)

    starts, stops, centers, vals = [], [], [], []
    for start, stop in windows:
        seg = arr[start:stop]
        spec = power_spectrum(seg, fs=fs, method="welch", nperseg=nperseg)
        energy = _band_energy_from_spectrum(spec.frequencies, spec.power, band)
        starts.append(start)
        stops.append(stop)
        centers.append((start + stop - 1) / 2.0)
        vals.append(energy)

    return WindowProfileResult(
        starts=np.asarray(starts, dtype=int),
        stops=np.asarray(stops, dtype=int),
        centers=np.asarray(centers, dtype=float),
        values=np.asarray(vals, dtype=float),
        metric_name="sliding_spectral_energy",
        parameters={
            "window_size": window_size,
            "step_size": step_size,
            "fs": fs,
            "band": band,
            "nperseg": nperseg,
            "include_partial": include_partial,
        },
    )


def sliding_window_coherence_summary(
    x: ArrayLike,
    y: ArrayLike,
    *,
    window_size: int,
    step_size: int,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    include_partial: bool = False,
    summary: str = "mean",
) -> WindowProfileResult:
    """
    Sliding-window coherence summary between two tracks.

    The full coherence curve can be expensive to preserve for every window, so
    this helper stores a simple summary statistic per window.
    """
    xa, ya, _ = align_signals(x, y, drop_nonfinite=False)
    windows = _window_slices(len(xa), window_size=window_size, step_size=step_size, include_partial=include_partial)

    starts, stops, centers, vals = [], [], [], []
    for start, stop in windows:
        local = cross_spectral_analysis(
            xa[start:stop],
            ya[start:stop],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        if local.coherence.size == 0:
            value = np.nan
        elif summary == "mean":
            value = float(np.nanmean(local.coherence))
        elif summary == "max":
            value = float(np.nanmax(local.coherence))
        else:
            raise ValueError("summary must be 'mean' or 'max'")

        starts.append(start)
        stops.append(stop)
        centers.append((start + stop - 1) / 2.0)
        vals.append(value)

    return WindowProfileResult(
        starts=np.asarray(starts, dtype=int),
        stops=np.asarray(stops, dtype=int),
        centers=np.asarray(centers, dtype=float),
        values=np.asarray(vals, dtype=float),
        metric_name=f"sliding_coherence_{summary}",
        parameters={
            "window_size": window_size,
            "step_size": step_size,
            "fs": fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "include_partial": include_partial,
            "summary": summary,
        },
    )


def localized_spectrum_stft(
    values: ArrayLike,
    *,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    detrend: Union[str, bool] = False,
) -> STFTResult:
    """
    Localized frequency analysis using STFT.

    STFT is implemented first because it is standard, interpretable, and
    dependency-light while still exposing non-stationary local scale structure.
    """
    arr, clean_meta = clean_signal(values)
    if arr.size == 0:
        empty = np.array([])
        return STFTResult(
            frequencies=empty,
            positions=empty,
            stft=np.empty((0, 0), dtype=complex),
            magnitude=np.empty((0, 0)),
            power=np.empty((0, 0)),
            fs=fs,
            nperseg=nperseg,
            noverlap=0 if noverlap is None else noverlap,
            window=window,
            detrend=detrend,
            metadata={"empty_input": True, **clean_meta},
        )

    if noverlap is None:
        noverlap = max(0, nperseg // 2)

    f, t, zxx = signal.stft(
        arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    power = magnitude ** 2

    return STFTResult(
        frequencies=f,
        positions=t,
        stft=zxx,
        magnitude=magnitude,
        power=power,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        metadata=clean_meta,
    )


def spectrum_1d(
    values,
    *,
    fs: float = 1.0,
):
    """
    Phase-1 compatibility wrapper for power spectrum.

    Returns
    -------
    frequencies : np.ndarray
    power : np.ndarray

    Notes
    -----
    This function exists to preserve compatibility with earlier EPIGO
    visualization code that expects (freqs, power) tuples instead of
    structured dataclasses.
    """
    spec = power_spectrum(values, fs=fs)
    return spec.frequencies, spec.power
