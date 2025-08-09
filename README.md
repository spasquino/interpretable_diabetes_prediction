# BRFSS Modeling (Julia)

This repository is a Julia-based machine learning pipeline for interpretable prediction of diabetes risk, using demographic and survey-based medical data. It includes modules for data preprocessing, missing-value imputation, model training (IAI and sparse logistic regression), and evaluation, enabling reproducible end-to-end analysis.

## Structure
- `src/brfss_preprocessing.jl` — preprocessing utilities
- `src/impute.jl` — imputation helpers
- `src/models_iai.jl` — models
- `src/sparse_log_reg.jl` — sparse logistic regression
- `main.jl` — entrypoint with CLI for pipeline stages

## Quickstart
1. Ensure Julia is installed (1.9+ recommended).  
2. (Optional) Activate a local environment and add dependencies:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.add([
       PackageSpec(name="ArgParse")
       # Add other packages used in the source files (CSV, DataFrames, MLJ, GLM, etc.)
   ])
   ```
3. Run a stage:
   ```bash
   julia --project=. main.jl --stage=preprocess --input data/raw.csv --output data/clean.csv
   julia --project=. main.jl --stage=impute --input data/clean.csv --output data/imputed.csv
   julia --project=. main.jl --stage=train --config configs/default.yaml
   julia --project=. main.jl --stage=evaluate --config configs/default.yaml
   ```
