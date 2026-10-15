# Energy Price Forecasting

Day-ahead forecasting of the daily Regional Reference Price (`RRP`) of electricity with
feature engineering and several neural network architectures (Keras / TensorFlow).
The repository also contains a FastAPI prediction service and an alternative ensemble
script built on the `darts` library.

## What is in the repository

| Path | Content |
| --- | --- |
| `electricity/preprocessing.py` | scikit-learn transformer pipeline: calendar, rolling, interaction and lag features |
| `electricity/data.py` | `DataConfig` and `SupervisedDataset`: future-target creation, chronological split, scaling, save/load |
| `electricity/model.py` | `ModelConfig` and `EnergyPricePredictor` (four architectures), multi-model training, ensembling |
| `electricity/helpers.py` | plots, metrics, error analysis, JSON/CSV reporting |
| `electricity/main.py` | end-to-end pipeline (CLI and `run_quick_experiment`) |
| `electricity/api.py` | FastAPI app serving a saved Keras model (`POST /predict`) |
| `electricity/darts.py` | separate `darts`-based ensemble script (RF, linear, XGBoost, RNN/LSTM/GRU, Transformer, Prophet) |
| `raw_data/complete_dataset.csv` | input data: 2106 daily rows, 2015-01-01 to 2020-10-06, 14 columns |
| `quick_experiment/` | artifacts committed from one run of `run_quick_experiment` (data splits, model, two plots) |
| `data/`, `results/` | intermediate snapshots and a saved model / array committed by the authors |
| `notebooks/`, `boilerplates/` | exploratory notebooks and the scripts exported from them |

`electricity/shapie_values.py` and `electricity/streamlit_app.py` are empty placeholders.

## Input data

`raw_data/complete_dataset.csv` has the columns `date, demand, RRP, demand_pos_RRP,
RRP_positive, demand_neg_RRP, RRP_negative, frac_at_neg_RRP, min_temperature,
max_temperature, solar_exposure, rainfall, school_day, holiday`.
The columns `demand_pos_RRP`, `RRP_positive`, `demand_neg_RRP`, `RRP_negative` and
`frac_at_neg_RRP` are derived from the same-day price and are treated as potentially
leaky: they are dropped by default (`keep_leaky=False`).
The repository does not state where the dataset comes from.

## Setup

The repository has no `requirements.txt` or `pyproject.toml` (the `Dockerfile` expects a
`requirements.txt`). The code imports:

- always: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `tensorflow`
- `electricity/api.py`: `fastapi`, `uvicorn`, `python-multipart` (for file upload)
- `electricity/darts.py`: `darts`, `xgboost`

`.python-version` pins Python 3.11. The quickstart below was run with Python 3.11 and
TensorFlow 2.21 (CPU).

```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow pandas scikit-learn scipy matplotlib seaborn
```

## Quickstart

The modules import each other without a package prefix, so run Python from inside the
`electricity/` directory. This preprocesses the raw CSV, builds the supervised dataset and
trains the standard network for 3 epochs on CPU (a smoke run, not a tuned model; it takes
well under a minute on a laptop-class CPU):

```python
# run from electricity/
from preprocessing import preprocess_data
from data import DataConfig, SupervisedDataset
from model import ModelConfig, train_model_from_data

preprocess_data("../raw_data/complete_dataset.csv", "out/preprocessed.csv")

train, val, test = SupervisedDataset(DataConfig(horizon=1)).prepare_data_from_csv(
    "out/preprocessed.csv"
)  # each is (X, y); X is float32 (n_samples, 60), y is (n_samples, 1)

model, results = train_model_from_data(train, val, test, ModelConfig(epochs=3), "out/model")
print(results["test"])  # dict with RMSE, MAE, R2, AIC, MSE, n_params
```

Outputs: `out/preprocessed.csv`, `out/model/best_model.keras`, `out/model/model_config.json`.

## Pipeline

1. **Preprocessing** (`preprocess_data`, `build_preprocessing_pipeline`): parse and sort by
   date, drop leaky columns, map `school_day` / `holiday` from `Y`/`N` to 1/0, add calendar
   features (year, month, week, day of week, ... with sin/cos encodings and yearly Fourier
   harmonics), rolling mean/std (windows 7 and 30) plus 7-day min/max for `RRP` and
   `demand`, an `RRP_high_vol` flag, interaction terms, and lags
   (default: `RRP` and `demand` at 1, 2, 7, 30 days; weather variables at 1 and 7), then
   drop rows with NaN. On the raw file this gives 2065 rows x 61 columns.
2. **Dataset** (`SupervisedDataset`): shift the target by `horizon` days, split
   chronologically (default 80/10/10), fit a `StandardScaler` on the training split only.
   `target_mode` is `"single"` (`RRP`), `"all_rrp"` (five RRP-related columns, which
   requires `keep_leaky=True` in preprocessing) or `"custom"`. With `sequence_length` set,
   `prepare_for_multiple_models` also returns 3D sequences for the transformer.
3. **Models** (`EnergyPricePredictor`, `ModelConfig.model_type`):
   - `standard`: feed-forward layers with optional batch norm and dropout
   - `residual`: feed-forward network with skip connections
   - `mlp_complex`: per block three dense layers (units, units/2, units), each with batch norm and dropout
   - `transformer`: Conv1D plus multi-head attention on 3D input
4. **Training / evaluation**: Adam, early stopping, learning-rate reduction on plateau,
   best-weights checkpoint, fixed seed (`seed=42`). Metrics per split: RMSE, MAE, R2, MSE,
   AIC (`n*log(MSE) + 2k`, `k` = parameter count) and parameter count.
   `train_multiple_models` trains several architectures and returns them keyed by type.

Full pipeline from the command line (writes to `--output_dir`, default `artifacts/`):

```bash
cd electricity
python main.py --input_csv ../raw_data/complete_dataset.csv --epochs 50
```

Run with no arguments, `main.py` runs `run_quick_experiment()` with paths relative to the
current directory (`raw_data/complete_dataset.csv`, output in `quick_experiment/`).

Known issues in the current code are listed under [Known limitations](#known-limitations).

## Results

No evaluation metrics are stored in the repository. What is committed:

- `quick_experiment/evaluation/training_history.png` and `test_predictions.png`: plots
  from one run of the quick experiment.
- `quick_experiment/model/model_config.json`: the configuration of that run
  (`standard` network, hidden layers 128/64/32, 50 epochs, seed 42).
- `quick_experiment/data/`: the scaled splits of that run: `X_train` (1651, 60),
  `X_val` (206, 60), `X_test` (207, 60), target `RRP_t+1`.

Rerun the quickstart or `run_quick_experiment()` to get metrics for your own run.

## API service

`electricity/api.py` loads a Keras model at import time (`MODEL_PATH` environment
variable, default `electricity/rnn_model.keras`) and exposes:

- `GET /`: welcome message
- `POST /predict`: multipart upload of a `.npy` (or `.npz`, first array) file; returns
  `input_shape` and a flat `prediction` list. Errors return HTTP 400.

A saved model is committed at `results/rnn_model.keras` with a sample input
`results/my_x.npy` (float32, shape (782, 60, 14)). The default path above does not exist in
the repository, so set `MODEL_PATH` explicitly:

```bash
pip install fastapi uvicorn python-multipart
MODEL_PATH=results/rnn_model.keras uvicorn electricity.api:app --port 8080
```

The `Dockerfile` builds an image that runs this app with `uvicorn`.

## Known limitations

Found during a code review; these are documented rather than changed:

- `main.py` `main()` passes `use_leaky` to `DataConfig`, which has no such field, so the
  command-line path raises `TypeError` before training.
- `run_quick_experiment()` stops in `plot_residuals` (`x and y must be the same size`):
  `y_test` has shape `(n, 1)` while `EnergyPricePredictor.predict` returns a flattened
  `(n,)` array, so `y_true - y_pred` broadcasts to `(n, n)`. Training, evaluation metrics
  and the first two plots complete before this point.
- `calculate_detailed_metrics` in `helpers.py` calls `stats.skew` / `stats.kurtosis`
  without importing `scipy.stats`, so it raises `NameError`.
- `electricity/darts.py` shares its name with the `darts` package; run as a script from
  its own directory it imports itself instead of the library. It also reads
  `complete_dataset.csv` from the working directory by default.
- The `Makefile` targets reference `complete_dataset.csv` / `data.csv` paths that do not
  match the repository layout.
- No automated tests are included.
