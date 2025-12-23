# MLOps Take‑Home Assessment (2 hours): NYC Short‑Term Rental Price Pipeline

This repo is a take‑home assessment for an **MLOps** role. You will implement and wire together a small **end‑to‑end training pipeline** (MLflow + W&B) that estimates short‑term rental prices in NYC from tabular data.

The goal is not to build a perfect model; it’s to demonstrate **production‑oriented ML engineering**: reproducibility, configuration, artifacts, testing, and clean pipeline structure.

---

## What you will build

An MLflow pipeline orchestrated by `main.py` that:

- **Downloads data** and versions it as a W&B artifact
- **Cleans data** and versions the cleaned dataset as a W&B artifact
- **Validates data** with automated tests (including drift check)
- **Splits** into train/val and test datasets (artifacts)
- **Trains** a baseline Random Forest model and logs an MLflow model artifact
- (Optional) **Runs hyperparameter sweeps** via Hydra multi-run
- (Optional) **Tests** a promoted “prod” model on the held‑out test set

---

## Timebox & expectations

- **Timebox**: ~2 hours (don’t over‑optimize the model)
- **Focus**: correctness, clarity, reproducibility, and ML ops fundamentals
- **No hardcoding**: pipeline parameters must come from `config.yaml` (via Hydra)

---

## What to submit

Please provide:

- A PR or a patch with your changes
- A short writeup (can be in the PR description) that includes:
  - What you implemented
  - How to run the pipeline (commands)
  - Any assumptions / tradeoffs
  - (If applicable) link(s) to your W&B project runs/artifacts

---

## Evaluation rubric (what we look for)

- **Reproducibility**: deterministic configs, stable artifacts, no hidden state
- **Pipeline structure**: clear step boundaries, correct artifact wiring
- **Config discipline**: parameters come from `config.yaml` / Hydra overrides
- **Data & model hygiene**: sensible cleaning, tests, and failure modes
- **Logging & observability**: meaningful logs + W&B/MLflow tracking
- **Code quality**: readable, small functions, minimal duplication

---

## System requirements

- **Supported OS**: Ubuntu 22.04/24.04 (native or WSL) or recent macOS
- **Python**: **3.13**
- **Conda**: required (MLflow runs steps in isolated conda envs)
- **W&B account**: required (artifacts and experiment tracking)

---

## Quickstart

### 1) Create the dev environment

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### 2) Log into Weights & Biases

```bash
wandb login <YOUR_API_KEY>
```

### 3) Run the pipeline

Run everything:

```bash
mlflow run .
```

Run only selected steps during development:

```bash
mlflow run . -P steps=download
mlflow run . -P steps=download,basic_cleaning
```

Override config values via Hydra:

```bash
mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

---

## Repository tour

- `main.py`: pipeline orchestrator (the place where steps are wired)
- `config.yaml`: all pipeline parameters (Hydra config)
- `src/`: pipeline steps you will implement/complete
  - `src/data_check/`: dataset tests (PyTest) + drift checks
  - `src/train_random_forest/`: baseline model training step
- `components/`: reusable pre-built MLflow steps (used via GitHub URL)

### Pre-built components (reused as external MLflow projects)

These are invoked from `main.py` using `config["main"]["components_repository"]`:

- `get_data`: downloads the data and logs a raw data artifact
- `train_val_test_split`: creates `trainval_data.csv` and `test_data.csv` artifacts
- `test_regression_model`: evaluates a model against the test set

---

## Assessment tasks

### Task A — Implement **basic cleaning** step (required)

Create a new MLflow step under `src/basic_cleaning/` that:

- Downloads a raw dataset artifact from W&B (e.g. `sample.csv:latest`)
- Applies basic cleaning:
  - Remove `price` outliers using configured bounds (`etl.min_price`, `etl.max_price`)
  - Convert `last_review` to a datetime column
- Saves to `clean_sample.csv` with `index=False`
- Logs `clean_sample.csv` back to W&B as a new artifact (name/type/description are parameters)

Expected parameters:

- `input_artifact` (str)
- `output_artifact` (str)
- `output_type` (str)
- `output_description` (str)
- `min_price` (float)
- `max_price` (float)

Then wire it into `main.py` as the `basic_cleaning` step.

**Important**: when referencing W&B artifacts, always include an alias/version (e.g. `:latest`, `:v3`, `:reference`).

---

### Task B — Add/complete **data checks** (required)

In `src/data_check/test_data.py`, add tests such as:

- `test_row_count(data)`: sanity check dataset size
- `test_price_range(data, min_price, max_price)`: assert `price` is within bounds

Then add the `data_check` step call in `main.py` so it runs after cleaning.

Use:

- `csv`: `clean_sample.csv:latest`
- `ref`: `clean_sample.csv:reference` (tag the current latest as `reference` in the W&B UI)

Run just this step:

```bash
mlflow run . -P steps=data_check
```

---

### Task C — Train/test split (required)

Call the provided `train_val_test_split` component from `main.py` after `data_check`.

Use config values for:

- `test_size`
- `random_seed`
- `stratify_by`

This produces `trainval_data.csv` and `test_data.csv` artifacts.

---

### Task D — Train a baseline Random Forest (required)

Complete `src/train_random_forest/run.py` (all TODOs are marked with `# YOUR CODE HERE`).

Then call it from `main.py` and log the output model artifact.

---

### Task E — Hyperparameter sweep (optional, if time)

Use Hydra multi-run to sweep:

- `modeling.max_tfidf_features` in `10,15,30`
- `modeling.random_forest.max_features` in `0.1,0.33,0.5,0.75,1`

Example:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

Select the best run in W&B by lowest MAE and tag its model artifact as `prod`.

---

### Task F — Test the promoted model (optional, if time)

Wire the `test_regression_model` component in `main.py`. This step is not run by default; run explicitly:

```bash
mlflow run . -P steps=test_regression_model
```

Use:

- `mlflow_model`: `random_forest_export:prod`
- `test_artifact`: `test_data.csv:latest`

---

## Expected “gotcha” (data boundary failure on sample2)

When training on `sample2.csv`, one of the geo boundary tests may fail (this is intentional).
Fix by adding this filter in `basic_cleaning` right before saving:

```python
idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
df = df[idx].copy()
```

Then tag a new release (e.g. `1.0.1`) and retry.

---

## Running a release against a new sample

After creating a GitHub release, you can run the pipeline from the tagged version:

```bash
mlflow run <YOUR_REPO_URL> -v <VERSION_TAG> -P hydra_options="etl.sample='sample2.csv'"
```

---

## Notes & troubleshooting

- **Hydra working directory**: steps run in a Hydra-managed directory; when referencing local step paths from `main.py`, use `hydra.utils.get_original_cwd()` to build paths correctly.
- **Environment corruption**: if MLflow-created conda envs become inconsistent, you may need to remove the `mlflow-*` conda envs and re-run.
- **Version alignment**: keep `mlflow` and `wandb` versions consistent across steps; ensure each step declares its dependencies in its `conda.yml`.


