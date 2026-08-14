# Zenodo v2 — form fields

Published: **10.5281/zenodo.21924169** (version 2 of 10.5281/zenodo.18753233).
Concept DOI, resolving to the latest version: **10.5281/zenodo.18753232**.

## Title
Two Domain Errors in a Symbolic Test for Deterministic Structure, and Their
Repair: Exact Calibration, Surrogate Domain, and an Open Benchmark

## Version
v2

## Description (short form, ~340 words)

Version 2 supersedes version 1 and withdraws its detector claim.

Version 1 proposed DDBM, a modular-arithmetic "Diophantine phase" statistic for
separating deterministic dynamics from noise in scalar time series. It is not
embedding-free — the phase is a function of the delay pair (N_t, N_{t+1}) at
m = 2, tau = 1, quantized at 1/K. Its stated mechanism, a singular marginal
measure occupying few lattice cells, is removed by the method's own rank
normalization. And its p-values were invalid.

The organising result is that one principle governs both stages of such a test,
and violating it in either place destroys validity silently: every operation
must be performed in the domain where its null is defined.

At the screening stage the statistic may depend on the data only through ranks.
Amplitude preprocessing performed beforehand destroys distribution-freeness,
driving empirical size at a nominal 0.05 to 0.19 on a lognormal marginal and
0.88 on a Cauchy marginal; rank-only symbolization restores exactness with a
one-line proof.

At the confirmation stage the AAFT/IAAFT null concerns a monotone transform of a
linear Gaussian process, so the spectrum to match is the underlying Gaussian's.
Surrogates built from raw amplitudes reject a cubed and an exponentiated AR(1) —
both inside their own null — in 100% of 100 replicates, while accepting the
untransformed AR(1) at 0.02. In normal scores all three land on exactly the
nominal 0.05. Since a monotone transform preserves ranks, that invariance is a
cheap sufficient diagnostic of surrogate misspecification.

One failure survives the repair: conditional heteroscedasticity is rejected at
0.15 [0.09, 0.23] even in normal scores and requires a model-based null. This
resolves the S&P 500 case rather than merely withdrawing it — log returns are
rejected by an IAAFT null (p = 0.02) but not by a fitted GARCH(1,1) null
(p = 0.31).

No new detection principle is claimed; the mechanism is the forbidden/missing
ordinal pattern paradigm. What is offered is a validated inference layer for
those classical statistics, a catalogue of its failure modes, and a benchmark of
165 series whose 46 labeled members are byte-reproducible, SHA-256 pinned, and
labeled by computed Lyapunov exponents. On those 40 scored series the three
statistics compared are statistically indistinguishable (exact McNemar
p = 1.00, 0.25, 0.69). The comparisons are calibration, not independent
validation.

Code, run logs and manuscript sources:
https://github.com/Theclimateguy/DDBM/tree/v2

## Keywords

Chosen against measured Zenodo usage rather than invented: each term below is
already carried by other records, so it actually reaches a search. Counts are
records using the term as a keyword, measured via the Zenodo API.

Broad reach, all accurate for this work:
time series (5172); time series analysis (1654); nonlinear dynamics (1748);
chaos (1732); reproducibility (6291); benchmark (3379); python (7630)

Narrow, targeting the niche:
symbolic dynamics (183); surrogate data (38); permutation entropy (21);
lyapunov exponent (144); hypothesis testing (254)

Optional additions: negative results (185), IAAFT (18).

Deliberately dropped, being effectively unused as keywords and therefore
invisible to search: normal scores (0 records), Monte-Carlo calibration (2),
distribution-free inference (4), deterministic structure (8), ordinal patterns
(12), cyclotomic polynomials (31 — but those records are pure number theory, the
wrong audience). Note also that Zenodo enforces no controlled vocabulary here:
across 500 sampled records not one carried a controlled subject term, so
keywords are free text and only prior usage makes them findable.

## Related identifiers
- 10.5281/zenodo.18753233 — "is new version of"
- https://github.com/Theclimateguy/DDBM/tree/v2 — "is supplemented by"

## License
CC BY 4.0

## Notes
Version 2 supersedes version 1; the detector claim of v1 is withdrawn. Prepared
in response to three independent referee reports, which identified the embedding
claim, the pivotality gap, the unvalidated confirmation stage, the scoring rule,
the baseline specification, the replicate counts and the benchmark's
reproducibility status.
