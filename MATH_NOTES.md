# EPIGO Phase 2 Math Notes

This note records the main mathematical design decisions used in Phase 2.

## 1. Discrete entropy-based comparisons

For a binned empirical distribution `p`, Shannon entropy is

\[
H(p) = -\sum_i p_i \log_b p_i
\]

where `b=2` by default for interpretability in bits.

### Joint entropy

For the joint binned distribution `p(x, y)`,

\[
H(X,Y) = -\sum_{i,j} p_{ij} \log_b p_{ij}
\]

### Conditional entropy

\[
H(X|Y) = H(X,Y) - H(Y)
\]

\[
H(Y|X) = H(X,Y) - H(X)
\]

### Mutual information

MI is estimated from the discretized variables:

\[
I(X;Y) = \sum_{i,j} p_{ij}\log_b\frac{p_{ij}}{p_i p_j}
\]

This can also be written as:

\[
I(X;Y) = H(X) + H(Y) - H(X,Y)
\]

### Jensen-Shannon divergence

\[
JSD(P \parallel Q) = \frac{1}{2} KL(P \parallel M) + \frac{1}{2} KL(Q \parallel M)
\]

with

\[
M = \frac{1}{2}(P+Q)
\]

The square root is used as Jensen-Shannon distance.

### Smoothing

All discrete distributions use epsilon smoothing before logarithms:

\[
\tilde{p}_i = \frac{p_i + \varepsilon}{\sum_j (p_j + \varepsilon)}
\]

This avoids undefined `log(0)` behavior and improves robustness for sparse bins.

## 2. Spectral comparison

Cross spectral density and coherence are computed with `scipy.signal.csd`
and `scipy.signal.coherence`.

Magnitude-squared coherence:

\[
C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f)P_{yy}(f)}
\]

This is useful for examining shared frequency/scale structure.

## 3. Windowed analysis

A signal of length `N` is split into windows of length `window_size` with
advance `step_size`. Each window receives a scalar summary such as:
- entropy
- mutual information
- spectral energy
- coherence summary

This gives genomic-position-dependent profiles.

## 4. Localized frequency analysis

We use STFT as a conservative, dependency-light localized frequency method.
It reports:
- frequency axis
- position axis
- complex STFT values
- magnitude
- power

This is appropriate when tracks are not stationary.
