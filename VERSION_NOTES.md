# Version notes

## Original submission

Files from the original submitted version are preserved unchanged in
`archive/original_submission/`.

## Revised manuscript repository

The revised repository contains publication-relevant code, deterministic model
inputs, Monte Carlo input workbooks, and compact numerical data under
`figures/input_data/` required by the
plotting scripts. Full optimization and Monte Carlo output archives, test runs,
intermediate files, and generated figures are intentionally excluded from Git
and will be distributed separately as manuscript supplementary data where
appropriate.

The revised code uses repository-relative paths. GLPK is resolved from `PATH`
unless a solver executable is explicitly configured. The Monte Carlo analysis
uses Latin hypercube sampling, 5,000 iterations, and random seed 42 for each of
the six region/feedstock-scope combinations.
