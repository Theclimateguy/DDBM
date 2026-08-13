# Zenodo v2 — form fields

## Title
A Calibrated, Preprocessing-Free Rank Test for Deterministic Structure in Time
Series: Operation Order, Measured Surrogate Limits, and an Open Benchmark

## Version
v2

## Description (paste into the Description field)

Version 2 supersedes version 1 and withdraws its detector claim.

Version 1 proposed a modular-arithmetic "Diophantine phase" statistic for
separating deterministic dynamics from noise. Three of its claims do not
survive. The method is not embedding-free: the phase is a function of the delay
pair (N_t, N_{t+1}) at m = 2, tau = 1, quantized at 1/K. Its stated mechanism —
a singular marginal measure occupying few lattice cells — is removed by the
method's own rank normalization; what the test sees is the geometry of
transitions. And the p-values were invalid: with the published order of
operations the procedure is not even distribution-free, reaching an empirical
size of 0.88 at a nominal 0.05 on a Cauchy marginal.

Three results replace them. First, the order of operations decides validity:
amplitude preprocessing applied before ranking destroys distribution-freeness
(empirical size 0.19 lognormal, 0.88 Cauchy), while working on ranks alone
restores it with a one-line proof and gives nominal size across six marginals.
Second, the IAAFT confirmation stage on which every verdict rests is audited and
leaks — 25% rejection on a static monotone transform of a linear process, inside
its own null, and 50–75% on heteroscedastic processes, which invalidates the
financial conclusion of version 1. Third, the standard ordinal baseline is
silently degenerate at common sample sizes: at n = 10^4 the missing-pattern
statistic is identically zero under the null for pattern lengths d ≤ 6.

The cubic kernel of version 1 turns out to be one member of a cyclotomic family,
with an exact identity for the phase lattice. The identity is correct and
useless: the phase is a deterministic function of the pair, so the map is lossy,
and a plain occupancy count on that pair outperforms it. Under a matched
protocol on 40 labeled series the classical missing-ordinal-pattern count scores
37/40, pair occupancy 35/40, and the cyclotomic phase scan 34/40.

No new detection principle is claimed; the mechanism is the forbidden/missing
ordinal pattern paradigm and coarse-grained transition statistics. What is
offered is a calibrated, preprocessing-free procedure with measured limits, and
an open benchmark of 165 series whose synthetic labels are computed from
Lyapunov exponents rather than asserted.

Code, benchmark and manuscript sources:
https://github.com/Theclimateguy/DDBM/tree/v2

## Keywords
time series analysis; deterministic structure; surrogate data; ordinal patterns;
distribution-free inference; Monte-Carlo calibration; cyclotomic polynomials;
negative results; reproducibility; benchmark

## Related identifiers
- 10.5281/zenodo.18753233 — "is new version of"
- https://github.com/Theclimateguy/DDBM/tree/v2 — "is supplemented by"

## License
CC BY 4.0
