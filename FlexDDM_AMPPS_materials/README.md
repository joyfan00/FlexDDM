# Materials for "Democratizing Diffusion Decision Models: A Tutorial on Building, Validating, and Fitting Custom Models With the FlexDDM Point-and-Click Interface"

This repository holds the materials behind the worked example and the accuracy checks in the article.

- **`flexddm_outputs/`** contains the tables and figures that FlexDDM produced for the worked example. They were made with FlexDDM's built-in Validate and Fit routines, so no scripts are needed to reproduce them. Run the same routines in the app with the settings listed below.
- **`flexddm_models/`** contains the custom collapsing-boundary DMC built in Box 3 of the article. Copy both files into FlexDDM's `generated_models` folder, or rebuild the model in the Model window as Box 3 describes.
- **`benchmarks/`** contains the scripts for the accuracy checks in Appendix B and Table B1. They compare FlexDDM with a closed-form solution, PyDDM, and DMCfun.
- **`data/ulrich2015.csv`** is the flanker dataset of Ulrich et al. (2015), which also ships with FlexDDM.

## Reproducing the FlexDDM results in the app

### Data window

- **File:** `data/ulrich2015.csv`, with participant `PPT`, response time `RT` (seconds), response `Correct` (accuracy coding), and condition column `Condition`.
- **Condition coding:** the DMC template treats condition `0` as congruent and `1` as incongruent. Recode `Condition` so that 1 (congruent) becomes 0 and 2 (incongruent) becomes 1 before loading the file.
- **Filters:** no trial filters were applied.

### Settings used for every routine (Options and Fit/Validate panels)

| Setting | Value |
|---|---|
| Backend | Fokker-Planck |
| Objective | G² |
| Time step | 0.005 s |
| Grid step | 0.005 |
| Maximum time | 2.0 s |
| Differential evolution | maximum iterations 30, population size 5, seed 10 |
| Local refinement | on (polish off) |
| G² bins | FlexDDM defaults (.1, .3, .5, .7, .9 quantiles of correct response times; .25, .5, .75 accuracy bins) |

### Routines

| Folder | Window and routine | Model(s) | Routine settings | Article |
|---|---|---|---|---|
| `01_parameter_recovery_DMC` | Validate, parameter recovery | DMC | 100 synthetic participants, 336 trials each | Figure 7A, Appendix C |
| `02_model_recovery_DMC_vs_DMC_Collapsing` | Validate, model recovery | DMC, DMC_Collapsing | 30 synthetic participants per model, 336 trials each | Figure 7B, Appendix C |
| `03_split_half_reliability_DMC` | Validate, internal consistency | DMC | all 16 participants, odd and even trials | Figure 8, Appendix C |
| `04_likelihood_profiles_DMC` | Validate, likelihood profiles | DMC | 31 grid points, synthetic dataset of 336 trials | Figure 9, Appendix C |
| `05_curvature_DMC_participant1` | Validate, curvature | DMC | participant 1, maximum iterations 40, population size 8 | Figure 10, Appendix C |
| `06_fit_DMC` | Fit, every participant separately | DMC | | Figure 12 (participant 11), Appendix C |
| `07_fit_DMC_Collapsing` | Fit, every participant separately | DMC_Collapsing | | Appendix C |

In the recovery routines, FlexDDM draws generating values uniformly within each parameter's bounds. File names and columns follow FlexDDM's own exports.
- `*_estimates.csv` holds generated and recovered values, or odd- and even-half estimates.
- `*_summary.csv` holds the per-parameter correlations.
- `model_recovery_matrix.csv` holds the win proportions. `model_recovery_datasets.csv` adds each synthetic dataset's BIC values and generating parameters.
- `fit_result.csv` holds each participant's estimates, G², and BIC.
- `simulated_data.csv` holds the data simulated from the fitted model for the posterior predictive checks.

## Running the benchmarks

These scripts reproduce only the accuracy checks reported in Appendix B and Table B1. Run every script from the repository root.

### Requirements

- Python 3.11 with numpy 2.4, scipy 1.17, pandas 3.0, and PyDDM 0.9.0.
- FlexDDM installed from source (https://github.com/joyfan00/FlexDDM). The scripts call FlexDDM's Python package only to obtain its predictions for comparison.
- R 4.3.3 with DMCfun 5.0.0 (https://github.com/igmmgi/DMCfun).

### Scripts

| Table B1 row | Script | What it does | Output |
|---|---|---|---|
| 1 | `v1_solver.py` | Compares standard DDM choice probabilities with the closed-form solution at 50 random parameter sets | `v1_solver.csv` |
| 2 | `v1b_bins.py` | Compares standard DDM predicted bin proportions with PyDDM's analytical solution at 20-, 5-, and 1-ms steps | `v1b_bins.csv` |
| 3 | `v4_dmc_pred.py` | Compares DMC response time quantiles with PyDDM at 0.5-ms steps, 40 parameter sets | `v4_dmc_pred.csv` |
| 4 | `r3_dmcfun.R` | Runs DMCfun simulations (400,000 trials) at the parameter sets in `r3_params.csv` | `r3_dmcfun_out.csv` |
| 4 | `r3_compare.py` | Computes FlexDDM's DMC predictions at the same parameter sets | `r3_compare.csv` |
| 4 | `r3_euler.py` | Runs an independent simulator at 1-ms and 0.1-ms steps to locate the source of the DMCfun gap | `r3_euler.csv` |
| 5 | `fit_dmc_pyddm.py` | Refits the DMC template to each participant with PyDDM at 0.5-ms steps, using the same G² objective, bins, bounds, and optimizer settings as the FlexDDM fits in `flexddm_outputs/06_fit_DMC` | `fits_dmc_pyddm.csv` |
| all | `analyze.py` | Computes every number reported in Appendix B and Table B1 | `summary.json` |

Row 5 compares those PyDDM fits with the FlexDDM fits from the walkthrough (`flexddm_outputs/06_fit_DMC/fit_result.csv`). `analyze.py` evaluates both sets of estimates on the same fine grid.

Rows 1 and 2 use the standard DDM because it is the only FlexDDM template with an exact analytical solution, so it tests FlexDDM's solver against the true answer. The DMC runs through the same solver with a different drift function.

`models.py` (DMC definitions for FlexDDM and PyDDM), `g2tools.py` (the G² objective for PyDDM), and `common.py` (the impulse function and the simulator) are shared helpers.
