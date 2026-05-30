import numpy as np

from neid.analysis import (
    cross_spectral_analysis,
    jensen_shannon_divergence,
    joint_information_analysis,
    sliding_window_entropy,
)
from neid.tracks import Track, TrackComparisonConfig, pairwise_comparison_matrices


def test_entropy_constant_vector_does_not_crash():
    x = np.ones(128)
    profile = sliding_window_entropy(x, window_size=32, step_size=16, bins=8)
    assert profile.values.size > 0
    assert np.all(np.isfinite(profile.values))


def test_mutual_information_nonnegative():
    rng = np.random.default_rng(0)
    x = rng.normal(size=512)
    y = x + 0.05 * rng.normal(size=512)
    result = joint_information_analysis(x, y, bins=16, epsilon=1e-10)
    assert result.mutual_information >= -1e-9


def test_js_distance_is_symmetric():
    p = np.array([0.1, 0.3, 0.6])
    q = np.array([0.2, 0.5, 0.3])
    js1 = jensen_shannon_divergence(p, q)
    js2 = jensen_shannon_divergence(q, p)
    assert np.isclose(js1, js2, atol=1e-12)


def test_coherence_output_shape_is_sensible():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1024)
    y = x + 0.1 * rng.normal(size=1024)
    result = cross_spectral_analysis(x, y, nperseg=128)
    assert result.frequencies.shape == result.coherence.shape
    assert result.cross_spectrum.shape == result.coherence.shape


def test_windowed_function_number_of_windows():
    x = np.arange(100.0)
    profile = sliding_window_entropy(x, window_size=20, step_size=10, bins=5)
    assert len(profile.values) == 9  # starts at 0..80


def test_pairwise_matrix_shapes():
    tracks = [
        Track.from_array(np.linspace(0, 1, 128), name="a"),
        Track.from_array(np.linspace(1, 0, 128), name="b"),
        Track.from_array(np.sin(np.linspace(0, 4 * np.pi, 128)), name="c"),
    ]
    mats = pairwise_comparison_matrices(tracks, config=TrackComparisonConfig(bins=8))
    for res in mats.values():
        assert res.matrix.shape == (3, 3)
        assert np.allclose(res.matrix, res.matrix.T, equal_nan=True)
