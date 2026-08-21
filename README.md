# Food2Chemical

This repository contains the revised, reproducible analysis code and the input
data required for the Food2Chemical manuscript. It does not version complete
numerical result archives or generated figure files. Those outputs are intended
to be distributed separately with the manuscript as supplementary data.

## Study design

The analysis covers mainland China (CN), the United States (US), and the
European Union (EU-27). Two feedstock scopes are evaluated:

- **FL**: food-loss streams only.
- **FLB**: food-loss and food-by-product streams.

For every region and feedstock scope, the model evaluates two objectives:

- maximize global warming potential (GWP) saving;
- maximize potential economic margin.

The uncertainty workflow uses Latin hypercube Monte Carlo sampling. The revised
manuscript results were generated with 5,000 iterations per region/scope and a
random seed of 42.

## Repository structure

```text
archive/original_submission/   Frozen original-submission files
code/optimization/             Deterministic Pyomo optimization
code/monte_carlo/              Monte Carlo uncertainty analysis
code/visualization/            Figure-generation code
data/model_inputs/FL/          Deterministic FL inputs
data/model_inputs/FLB/         Deterministic FLB inputs
data/monte_carlo_inputs/       Uncertainty parameter workbooks
figures/input_data/            Compact numerical inputs required by plots
figures/main/                  Generated main figures; ignored by Git
figures/supplementary/         Generated supplementary figures; ignored by Git
outputs/                       Generated numerical outputs; ignored by Git
```

`archive/original_submission/` is a frozen provenance copy and must not be
modified when updating the revised workflow.

## Installation

Python 3.11 was used for the revised analysis. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The optimization also requires a linear-programming solver supported by Pyomo.
The default configuration uses GLPK and resolves `glpsol` from the system
`PATH`. A solver executable can be supplied explicitly through the command-line
option used by the Monte Carlo workflow or in `batch_config.json`.

## Run the deterministic optimization

From the repository root:

```bash
python -m code.optimization.batch_optimize
```

The configuration reads all six input workbooks and writes combined result and
analysis workbooks under `outputs/optimization/`. These generated workbooks are
not committed to Git.

## Run the Monte Carlo analysis

```bash
python -m code.monte_carlo.batch_optimize_monte_carlo \
  --iterations 5000 \
  --seed 42
```

Use `--only CN_FLB` (or another region/scope code) to run one independent
scenario. Generated files are written under `outputs/monte_carlo/` and are not
committed to Git.

## Regenerate figures

Compact plotting inputs are stored in `figures/input_data/`. Examples:

```bash
python -m code.visualization.allocation_chord
python -m code.visualization.plot_fl_ch_heatmap_bubbles --scenario FL
python -m code.visualization.plot_fl_ch_heatmap_bubbles --scenario FLB
python -m code.visualization.plot_fl_availability_pc_demand
python -m code.visualization.plot_optimization_analysis
python -m code.visualization.plot_mc_uncertainty_results
```

The sunburst visualization is provided as
`code/visualization/sunburst_ch3_quantification.ipynb`. Generated PNG, PDF, and
SVG files are written beneath `figures/` and are intentionally ignored by Git.

## Reproducibility workflow

```text
model inputs
    -> deterministic optimization
    -> Monte Carlo analysis
    -> supplementary result tables
    -> compact plotting input data
    -> plotting code
    -> manuscript figures
```

The model inputs and plotting input data in this repository are sufficient to
inspect the revised model and regenerate the plotted figures. Complete numerical
results should be obtained from the manuscript supplementary-data package.
