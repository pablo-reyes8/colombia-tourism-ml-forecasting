# Notebook Modularization Map

This repository started as a notebook-first workflow. The modules below now capture the highest-value reusable logic from the original notebooks.

## Notebook 1: Initial Data Construction and Descriptive Analysis

- `src/colombia_tourism/preprocessing/text.py`
  Standardizes city names and text keys used across sources.
- `src/colombia_tourism/preprocessing/time.py`
  Completes missing months and normalizes calendar keys.
- `src/colombia_tourism/preprocessing/crime.py`
  Shapes crime data into city-month tables.
- `src/colombia_tourism/preprocessing/geo.py`
  Converts location fields into geometry.
- `src/colombia_tourism/modeling/preprocess.py`
  Contains LOESS decomposition, kriging and panel assembly helpers.
- `src/colombia_tourism/features/engineering.py`
  Now adds panel-level ratios and derived variables that were previously implicit in notebook transformations.

## Notebook 2: Satellite Image Feature Engineering

- `src/colombia_tourism/features/satellite.py`
  Centralizes Earth Engine setup, Otsu thresholding, index computation and land-cover extraction.

## Notebook 3: Final Dataset Assembly and Missing Value Imputation

- Intentionally not expanded in this pass, per project priority.
- Existing reusable parts remain in `src/colombia_tourism/modeling/preprocess.py`.

## Notebook 4: Descriptive Statistics and Final Dataset Plots

- `src/colombia_tourism/analysis/eda.py`
  Replaces manual descriptive tables, target correlations, monthly profiles, share comparisons, bubble plots and 3D exploratory views.
- `scripts/eda.py`
  Generates reusable CSV summaries and core EDA figures without opening Jupyter.

## Notebook 5: Modeling and Econometric Comparison

- `src/colombia_tourism/modeling/benchmark.py`
  Adds a repeatable benchmarking layer for ML, OLS and random-effects comparisons.
- `scripts/benchmark.py`
  Runs shared-split model comparisons from the command line.

## Notebook 6: Best Model Interpretation and Insights

- `src/colombia_tourism/interpretation/lime.py`
  Now supports table outputs and repeated instance explanations.
- `src/colombia_tourism/interpretation/pdp.py`
  Adds manual partial-dependence sweeps aligned with the notebook workflow.

## Recommended Next Pass

- Move notebook 1 raw-source loaders into dedicated source-specific modules under `preprocessing/`.
- Add tests for the new feature-engineering and benchmarking APIs.
- Convert notebook 2 output artifacts into versioned data products with metadata.
