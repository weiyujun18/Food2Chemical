# Food2Chemical

This repository contains the code, model inputs, source data, and numerical results supporting the revised Food2Chemical study on system-level allocation of underutilized food resources to competing platform-chemical production pathways.

## Repository status

The repository is being reorganized for the revised manuscript. Files corresponding to the original submission have been preserved in `archive/original_submission/`. Revised code, data, and results should be placed in the structured directories described below.

## Study scope

The analysis considers three regions:

- China mainland
- United States
- European Union (EU-27)

Two resource scenarios are evaluated:

- Food loss (FL)
- Food loss and by-products (FLB)

The optimization is performed under two alternative decision objectives:

- maximize global warming potential (GWP) saving
- maximize potential economic margin

The revised analysis also includes Monte Carlo uncertainty analysis.

## Repository structure

```text
Food2Chemical/
├── README.md
├── VERSION_NOTES.md
├── archive/
│   └── original_submission/    # Files from the original repository
├── code/
│   ├── optimization/           # Revised optimization model
│   ├── monte_carlo/            # Monte Carlo uncertainty analysis
│   └── visualization/          # Scripts used to generate data-driven figures
├── data/
│   ├── model_inputs/
│   │   ├── FL/                 # Model inputs for the FL scenario
│   │   └── FLB/                # Model inputs for the FLB scenario
│   ├── monte_carlo_inputs/     # Uncertainty-analysis inputs
│   └── source_data/            # Source/processed data underlying model inputs
├── results/
│   ├── optimization/
│   │   ├── FL/
│   │   └── FLB/
│   ├── monte_carlo/
│   │   ├── FL/
│   │   └── FLB/
│   └── figure_source_data/     # Numerical data underlying manuscript figures
└── figures/
    ├── main/
    └── supplementary/
```

## Reproducibility workflow

The intended data and analysis workflow is:

```text
source data
    ↓
model inputs
    ↓
optimization / Monte Carlo analysis
    ↓
result tables
    ↓
figure source data
    ↓
plotting scripts
    ↓
manuscript figures
```

Revised scripts should use repository-relative paths rather than user-specific absolute paths so that the workflow can be reproduced after cloning or downloading the repository.

## Original submission

The original repository contents are preserved under:

`archive/original_submission/`

These files are retained for transparency and provenance. They correspond to an earlier version of the analysis and may contain paths or assumptions that are no longer used in the revised manuscript.

## Software environment

A finalized `requirements.txt` and solver instructions should be added after the revised analysis scripts are placed in the repository. The optimization workflow should document the Python version, required Python packages, solver and solver version, and any random seed/settings used for Monte Carlo analysis.

## Data and code availability

The final repository should contain the model inputs and outputs necessary to reproduce the results reported in the revised manuscript. Third-party datasets or reports that cannot legally be redistributed should be cited and linked rather than uploaded in full.
