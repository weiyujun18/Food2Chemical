# Version notes

## Original submission

The files from the repository prior to the manuscript revision are preserved in:

`archive/original_submission/`

No analytical files in that directory were intentionally modified during the repository restructuring.

## Revised manuscript

The top-level `code/`, `data/`, `results/`, and `figures/` directories are reserved for the revised manuscript.

Recommended Git tags/releases:

- `v1.0-original-submission` — tag the final commit immediately before restructuring, if still available in Git history.
- `v2.0-revised-manuscript` — create after the revised manuscript code, data, results, and documentation are finalized.

Before creating the revised release, verify that:

1. all analysis scripts use relative or configurable paths;
2. all manuscript results can be regenerated from the uploaded model inputs;
3. Monte Carlo settings and random seeds are documented;
4. each data-driven manuscript figure is linked to its source data and plotting script;
5. software and solver dependencies are documented;
6. files requiring third-party redistribution permission are excluded or replaced by citations/links.
