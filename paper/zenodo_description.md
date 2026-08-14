# Zenodo v2 — form fields

## Title
Two Domain Errors in a Symbolic Test for Deterministic Structure, and Their
Repair: Exact Calibration, Surrogate Domain, and an Open Benchmark

## Version
v2

## Description (paste into the Description field)

Version 2 supersedes version 1 and withdraws its detector claim.

Version 1 proposed DDBM, a modular-arithmetic "Diophantine phase" statistic for
separating deterministic dynamics from noise in scalar time series. Three of its
claims do not survive. The method is not embedding-free: the phase is a function
of the delay pair (N_t, N_{t+1}) at m = 2, tau = 1, quantized at 1/K, so
scanning K scans box scale. Its stated mechanism — a singular marginal measure
occupying few lattice cells — is removed by the method's own rank normalization,
which gives every cell about n/K points however singular the underlying measure;
what the test sees is the geometry of transitions. And its p-values were
invalid, for a deeper reason than the serial dependence of overlapping pairs.

The organising result of this version is that a single principle governs both
stages of such a test, and that violating it in either place destroys validity
silently: every operation must be performed in the domain where its null is
defined.

At the screening stage the null is invariant under monotone reparametrization,
so the statistic may depend on the data only through ranks. Amplitude
preprocessing performed beforehand — detrending, AR(1) prewhitening, volatility
standardization — destroys distribution-freeness, driving empirical size at a
nominal 0.05 to 0.19 on a lognormal marginal and 0.88 on a Cauchy marginal.
Rank-only symbolization restores exactness, with a one-line proof, and returns
nominal size across six marginals.

At the confirmation stage the AAFT/IAAFT null concerns a static monotone
transform of a linear Gaussian process, so the spectrum to match is that of the
underlying Gaussian. Surrogates generated from raw amplitudes reject a cubed
AR(1), and an exponentiated AR(1) — both inside their own null — in 100% of 100
replicates, while accepting the untransformed AR(1) at 0.02; generated from
ranks they reject strongly autocorrelated processes at 0.47. Generated in normal
scores, the rank-based inverse normal transform, all three land on exactly the
nominal 0.05 with identical Wilson intervals, invariance under monotone
transformation is restored, and power on deterministic systems is unaffected.
Since a monotone transform preserves ranks exactly, this invariance is a cheap
sufficient diagnostic of surrogate misspecification, and we recommend it as
routine.

One failure survives the repair. Conditional heteroscedasticity lies outside the
IAAFT null altogether and is rejected at 0.15 (95% interval [0.09, 0.23]) even
in normal scores, so heteroscedastic data require a model-based null. Applied to
the S&P 500, this resolves rather than merely withdraws the financial conclusion
of version 1: log returns are rejected by an IAAFT null (p = 0.02) but not by a
fitted GARCH(1,1) null (p = 0.31), so the apparent nonlinear structure is
conditional heteroscedasticity and not low-dimensional determinism.

Three secondary results. The cubic kernel is one member of a cyclotomic family:
S_p(N) = (N+1)^p − N^p factors into homogeneous cyclotomic polynomials evaluated
at consecutive integers, (N+1)/N is an exact p-th root of unity modulo S_p, and
the phase lattice has denominator Phi_n(1). The identity appears to be new in
this context but does not improve on simpler alternatives, since the phase is a
lossy function of the pair. The standard ordinal baseline is silently degenerate
when every pattern length in the grid is saturated, which at n = 10^4 means
d <= 6; above that floor the choice is immaterial. And on a benchmark of 40
labeled series the three statistics compared — missing ordinal patterns 37/40,
the cyclotomic phase scan 36/40, pair occupancy 34/40 — are statistically
indistinguishable, with exact McNemar p-values of 1.00, 0.25 and 0.69.

No new detection principle is claimed: the mechanism is the forbidden/missing
ordinal pattern paradigm and coarse-grained transition statistics, and the
smooth-stochastic confound has been known since the 1980s. What is offered is a
validated inference layer for those classical statistics, a catalogue of the
ways it fails, and a benchmark of 165 series whose 46 labeled members are
byte-reproducible from fixed seeds, pinned by SHA-256, and labeled by computed
Lyapunov exponents. The comparisons reported here are calibration on that set,
not independent validation; power curves and a held-out evaluation remain to be
done.

Code, benchmark, verbatim run logs and manuscript sources:
https://github.com/Theclimateguy/DDBM/tree/v2

## Keywords
time series analysis; deterministic structure; surrogate data; ordinal patterns;
distribution-free inference; Monte-Carlo calibration; IAAFT; normal scores;
cyclotomic polynomials; negative results; reproducibility; benchmark

## Related identifiers
- 10.5281/zenodo.18753233 — "is new version of"
- https://github.com/Theclimateguy/DDBM/tree/v2 — "is supplemented by"

## License
CC BY 4.0

## Notes (optional field)
Version 2 supersedes version 1; the detector claim of v1 is withdrawn. Prepared
in response to three independent referee reports, which identified the embedding
claim, the pivotality gap, the unvalidated confirmation stage, the scoring rule,
the baseline specification, the replicate counts and the benchmark's
reproducibility status.
