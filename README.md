# EPIGO (Extended)
**Epigenetic Information Geometry Observatory — Extension Layer**  
*(or: Epigenetics, built like LEGO — now with interactions and dynamics)*  

---

## 1. What is EPIGO (Extended)?

This repository is an **extension of the March EPIGO framework**:

👉 https://github.com/cluehning/march_EPIGO  

The original project already established the core idea:  
treating epigenetic tracks as **structured objects** and analyzing them using entropy and spectra.

This version does **not redefine that purpose**.  
Instead, it **extends EPIGO into a broader analysis system** that captures:

- relationships between tracks  
- local (position‑dependent) structure  
- distributional behavior of signals  
- non‑stationary dynamics  

---

### LEGO intuition (what changed)

Original EPIGO:
- You could **measure a brick**
  - entropy → how complex it is  
  - spectrum → how it is arranged  

Extended EPIGO:
- You can now:
  - compare bricks  
  - see where bricks interact  
  - track how structure changes across space  

So instead of just asking:

> *“What is this brick?”*

You can now ask:

> *“How do these bricks interact, and where does their structure change?”*

---

## 2. Project structure

    EPIGO/
    ├─ neid/
    │  ├─ __init__.py
    │  ├─ analysis.py        # Extended math: info theory, pairwise, sliding, STFT
    │  ├─ io.py
    │  ├─ tracks.py
    │  ├─ viz.py
    │  └─ viz_tracks.py
    │
    └─ README.md

⚠️ `neid/` is a **Python package**, not a folder of standalone scripts.

## Quick start

1. Open the **EPIGO** folder in VS Code  
2. Press **F5**  
3. Choose `Run neid.viz_tracks` or `Run neid.viz`  

### Why running files directly fails

EPIGO uses **relative imports** (for example `from .analysis import ...`), which only work when code is executed **as part of a package**.

❌ Do **not** do this:
- `python neid/viz_tracks.py`

✅ Correct:
- `python -m neid.viz_tracks`

---

## 3. Code overview (what changed)

This section focuses only on **additions vs March EPIGO**.

---

### `analysis.py`

The mathematical core has been **significantly expanded**.

---

### 3.1 From spectrum-only → multi-layer analysis

Original EPIGO:
- entropy (on probabilities)
- FFT spectrum

Now includes:
- probability estimation from raw signals  
- pairwise statistics  
- localized analysis  
- non-stationary analysis  

---

### 3.2 New: Probability + entropy from raw signals

- Converts signals into histograms  
- Supports quantile binning  
- Applies smoothing for numerical stability  

New:
- `estimate_probability`
- `ProbabilityEstimate`

---

### 3.3 New: Pairwise information layer

Adds:
- mutual information  
- joint entropy  
- conditional entropy  
- KL divergence  
- Jensen–Shannon divergence  

New:
- `joint_information_analysis`
- `PairwiseInfoResult`

---

### 3.4 New: Cross-signal spectral analysis

Adds:
- cross spectrum  
- coherence  
- cross-correlation  
- lag detection  

New:
- `cross_spectral_analysis`

---

### 3.5 New: Sliding-window analysis

Adds position-based analysis:

- entropy profiles  
- mutual information profiles  
- spectral energy summaries  
- coherence summaries  

New:
- `sliding_window_entropy`
- `sliding_window_mutual_information`
- `sliding_window_spectral_energy`
- `sliding_window_coherence_summary`

---

### 3.6 New: STFT (localized frequency analysis)

Adds:
- `localized_spectrum_stft`

Allows:
- tracking frequency structure across position  

---

### 3.7 New: Signal preprocessing

Adds:
- `clean_signal`
- `align_signals`
- `zscore_signal`

Handles:
- NaNs / infinities  
- unequal track lengths  
- normalization  

---

### 3.8 New: Structured outputs

Most functions now return **dataclasses instead of raw arrays**.

Examples:
- `SignalSpectrum`
- `CrossSpectralResult`
- `WindowProfileResult`
- `STFTResult`

These include:
- results  
- parameters  
- metadata  

---

### 3.9 Backward compatibility

Still supported:
- `spectrum_1d()` → returns `(frequencies, power)`

Now backed by improved internal implementation.

---

## 4. What changed (quick comparison)

| Aspect | March EPIGO | Extended EPIGO |
|------|--------------|----------------|
| Entropy | from probabilities | from raw signals |
| Spectrum | FFT | Welch-based |
| Pairwise analysis | none | MI, KL, JS, coherence |
| Local analysis | global | sliding-window |
| Non-stationary | no | STFT |
| Outputs | arrays | structured dataclasses |

---

## 5. Design philosophy

Still:

- **Modular**  
- **Explicit**  
- **Exploratory**  
- **Non-magical**

Extended with:

- **Relational** — compares signals  
- **Local** — detects position-dependent structure  
- **Multi-scale** — global + local + spectral  

---

## 6. TL;DR

- This repo **extends March EPIGO**  
- Adds:
  - pairwise signal analysis  
  - localized structure detection  
  - non-stationary frequency analysis  
- Keeps compatibility with original workflows  

---

### Development Note
Parts of the codebase were created with AI assistance ("vibe coding"), but the underlying ideas, research direction, experimental design, mathematical reasoning, and interdisciplinary extensions are my own. AI was used as an implementation and exploration tool, with all major decisions, modifications, and interpretations guided by the author.

---

## 9. License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software.
