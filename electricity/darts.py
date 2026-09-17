# -*- coding: utf-8 -*-
"""
Configurable ensemble forecasting with modular model selection.
Fixes:
- Transformer (past-covariates) now uses an *extended* past_covariates that
  reaches through the forecast horizon (required when n > output_chunk_length).
- Future covariates slices include +max_future_lag beyond horizon.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from darts import TimeSeries
from darts.metrics import mae, rmse, r2_score
from darts.models import (
    SKLearnModel,
    LinearRegressionModel,
    RNNModel,              # RNN/LSTM/GRU (dual covariates)
    Prophet,               # future-covariates model
    TransformerModel       # past-covariates model
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as SkLinearBlend
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

# --------------------------
# Config
# --------------------------
CSV_PATH    = "complete_dataset.csv"
RESULTS_DIR = "results"
TEST_H      = 250
VAL_H       = 250
RANDOM_SEED = 42

LAGS_TARGET = [-1, -2, -7, -14]
LAGS_FUT    = [0, 1, 7]  # current, +1, +7

AIC_K = {
    "rf":         50,
    "lin":        None,
    "xgb":        50,
    "rnn":        2000,
    "lstm":       2000,
    "gru":        2000,
    "transformer": 1500,
    "prophet":    20,
    "blend":      5,
    "blend_eq":   5
}

PALETTE = {"train":"black", "test":"#1f77b4", "pred":"#d627ff"}

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def _savefig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved plot → {path}")

def plot_like_notebook(train, test, pred, title, out_dir, fname):
    _ensure_dir(out_dir)
    plt.figure(figsize=(12,5))
    train.plot(label="Train", lw=1.2, color=PALETTE["train"])
    test.plot(label="Test (True)", lw=1.5, color=PALETTE["test"])
    pred.plot(label="Forecast", lw=2.0, color=PALETTE["pred"])
    plt.title(title); plt.grid(alpha=0.25); plt.legend()
    _savefig(os.path.join(out_dir, fname))

def plot_true_vs_pred_subplots(test, pred, title_top, title_bottom, out_dir, fname):
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(2, 1, figsize=(12,7), sharex=True)
    axes[0].plot(test.time_index, test.values().ravel(), label="Test (True)", lw=1.7, color=PALETTE["test"])
    axes[0].set_title(title_top); axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].plot(pred.time_index, pred.values().ravel(), label="Forecast", lw=1.7, color=PALETTE["pred"])
    axes[1].set_title(title_bottom); axes[1].grid(alpha=0.25); axes[1].legend()
    _savefig(os.path.join(out_dir, fname))

def plot_zoom_last(series, test, pred, pre_days, title, out_dir, fname):
    _ensure_dir(out_dir)
    full_idx = series.time_index
    start_pos = max(0, full_idx.get_loc(test.start_time()) - pre_days)
    base = series[full_idx[start_pos]:test.end_time()]
    plt.figure(figsize=(12,5))
    base.plot(label="History (zoom)", lw=1.0, color="#999999")
    test.plot(label="Test (True)", lw=1.7, color=PALETTE["test"])
    pred.plot(label="Forecast", lw=1.7, color=PALETTE["pred"])
    plt.title(title); plt.grid(alpha=0.25); plt.legend()
    _savefig(os.path.join(out_dir, fname))

def plot_parity(y_true_ts, y_pred_ts, out_dir, fname, title):
    _ensure_dir(out_dir)
    y, yhat = y_true_ts.values().ravel(), y_pred_ts.values().ravel()
    mn, mx = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
    plt.figure(figsize=(5.3,5.3))
    plt.scatter(y, yhat, s=14, alpha=0.7)
    plt.plot([mn, mx], [mn, mx], ls="--", lw=1.0, color="#444444")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title); plt.grid(alpha=0.25)
    _savefig(os.path.join(out_dir, fname))

def plot_residuals(y_true_ts, y_pred_ts, out_dir, fname, title):
    _ensure_dir(out_dir)
    resid = y_pred_ts.values().ravel() - y_true_ts.values().ravel()
    plt.figure(figsize=(12,3.6))
    plt.plot(y_true_ts.time_index, resid, lw=1.0)
    plt.axhline(0.0, ls="--", color="#555555", lw=1.0)
    plt.title(title); plt.xlabel("Time"); plt.ylabel("Residual"); plt.grid(alpha=0.25)
    _savefig(os.path.join(out_dir, fname))

def plot_rolling_mae(y_true_ts, y_pred_ts, window, out_dir, fname, title):
    _ensure_dir(out_dir)
    y = pd.Series(y_true_ts.values().ravel(), index=y_true_ts.time_index)
    yhat = pd.Series(y_pred_ts.values().ravel(), index=y_pred_ts.time_index)
    e = (y - yhat).abs().rolling(window).mean()
    plt.figure(figsize=(12,3.6)); plt.plot(e.index, e.values, lw=1.2)
    plt.title(f"{title} (window={window})"); plt.xlabel("Time"); plt.ylabel("Rolling MAE"); plt.grid(alpha=0.25)
    _savefig(os.path.join(out_dir, fname))

def plot_feature_importance_if_any(model_name, model_obj, feature_names, out_dir, fname, title,
                                   lags_target=None, lags_fut=None, cov_cols=None):
    est = getattr(model_obj, "model", None)
    if est is None: return
    if hasattr(est, "feature_importances_"):
        importances = np.array(est.feature_importances_)
    elif hasattr(est, "coef_"):
        importances = np.abs(np.ravel(est.coef_))
    else:
        return

    names = list(feature_names) if feature_names is not None else None

    def _build(lags_t, lags_f, covs):
        names_t  = [f"y(t{lt})" for lt in (lags_t or [])]
        names_fc = [f"{c}(t+{lf})" for lf in (lags_f or []) for c in (covs or [])]
        return names_t + names_fc

    if names is None and (lags_target is not None and lags_fut is not None and cov_cols is not None):
        names = _build(lags_target, lags_fut, cov_cols)
    if names is None:
        names = [f"feat_{i}" for i in range(len(importances))]

    if len(names) < len(importances):
        names += [f"feat_{i}" for i in range(len(names), len(importances))]
    elif len(names) > len(importances):
        names = names[:len(importances)]

    order = np.argsort(importances)[::-1][: min(30, len(importances))]
    names_ord = np.array(names)[order][::-1]
    imps_ord  = np.array(importances)[order][::-1]

    _ensure_dir(out_dir)
    plt.figure(figsize=(10,6))
    plt.barh(names_ord, imps_ord)
    plt.title(title); plt.xlabel("Importance")
    _savefig(os.path.join(out_dir, fname))

# --------------------------
# Model Factory
# --------------------------
class ModelFactory:
    @staticmethod
    def create_model(model_name: str):
        if model_name == "rf":
            return SKLearnModel(
                model=RandomForestRegressor(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1),
                lags=LAGS_TARGET, lags_future_covariates=LAGS_FUT
            )
        if model_name == "lin":
            return LinearRegressionModel(lags=LAGS_TARGET, lags_future_covariates=LAGS_FUT)
        if model_name == "xgb":
            return SKLearnModel(
                model=XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                    random_state=RANDOM_SEED, n_jobs=-1
                ),
                lags=LAGS_TARGET, lags_future_covariates=LAGS_FUT
            )
        if model_name in ("rnn", "lstm", "gru"):
            rnn_type = {"rnn":"RNN", "lstm":"LSTM", "gru":"GRU"}[model_name]
            return RNNModel(
                model=rnn_type,
                input_chunk_length=30,
                training_length=48,   # MUST be > input_chunk_length
                hidden_dim=64, n_rnn_layers=2, dropout=0.1,
                batch_size=32, n_epochs=50, random_state=RANDOM_SEED,
                optimizer_kwargs={"lr": 1e-3},
                pl_trainer_kwargs={"enable_progress_bar": False}
            )
        if model_name == "transformer":
            return TransformerModel(
                input_chunk_length=30,
                output_chunk_length=1,
                d_model=64, nhead=4,
                num_encoder_layers=2, num_decoder_layers=2,
                dim_feedforward=256, dropout=0.1,
                batch_size=32, n_epochs=50, random_state=RANDOM_SEED,
                optimizer_kwargs={"lr": 1e-3},
                pl_trainer_kwargs={"enable_progress_bar": False}
            )
        if model_name == "prophet":
            return Prophet(
                random_state=RANDOM_SEED,
                suppress_stdout_stderror=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
        raise ValueError(f"Unknown model name: {model_name}")

# --------------------------
# Metrics / Tables
# --------------------------
def compute_aic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    n = len(y_true); rss = np.sum((y_true - y_pred) ** 2)
    sigma2 = max(rss / max(n, 1), 1e-12); return 2 * k + n * np.log(sigma2)

def add_metrics(metrics: dict, name: str, y_true_ts: TimeSeries, y_pred_ts: TimeSeries, k: int):
    metrics[name] = {
        "MAE": float(mae(y_true_ts, y_pred_ts)),
        "RMSE": float(rmse(y_true_ts, y_pred_ts)),
        "R2": float(r2_score(y_true_ts, y_pred_ts)),
        "AIC": float(compute_aic(y_true_ts.values().ravel(), y_pred_ts.values().ravel(), k)),
    }

def save_tables_and_base_overlays(results_dir, train, val, test, preds, blend_weights, model_names):
    os.makedirs(results_dir, exist_ok=True)
    out_df = pd.DataFrame({"date": test.time_index, "y_true": test.values().ravel()})
    for name, ts in preds.items():
        out_df[name] = ts.values().ravel()
    out_df.to_csv(os.path.join(results_dir, "predictions_vs_truth.csv"), index=False)
    print(f"Saved table → {os.path.join(results_dir, 'predictions_vs_truth.csv')}")

    plt.figure(figsize=(12,5))
    test.plot(label="True", lw=1.5, color=PALETTE["test"])
    for name in model_names:
        if name in preds:
            preds[name].plot(label=name.upper(), lw=1.2)
    if "blend" in preds:
        preds["blend"].plot(label=f"Blended (w={np.round(blend_weights,2)})", lw=1.8, color=PALETTE["pred"])
    plt.title("Forecasts vs Truth (Test Window)"); plt.legend(); plt.grid(alpha=0.25)
    _savefig(os.path.join(results_dir, "forecast_comparison.png"))

# --------------------------
# Data prep
# --------------------------
def prepare_data():
    """
    Build:
      - series (target)
      - feature_cols (for naming/importance)
      - past_cov_ext: *extended* past-covariates (needed by Transformer)
      - future_cov: extended future-covariates (for rf/lin/xgb/rnn/prophet)
    """
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # binary
    df['school_day'] = df['school_day'].map({'Y':1, 'N':0})
    df['holiday']    = df['holiday'].map({'Y':1, 'N':0})

    # calendar
    df['dayofweek']  = df['date'].dt.dayofweek
    df['month']      = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # target lags (for linear/tree)
    for lag in [1,7,14]:
        df[f'lag_{lag}'] = df['RRP'].shift(lag)

    # rollings
    df['rolling_mean_7']  = df['RRP'].rolling(7).mean()
    df['rolling_mean_30'] = df['RRP'].rolling(30).mean()

    # interactions
    df['temp_range'] = df['max_temperature'] - df['min_temperature']
    df['rain_x_temp'] = df['rainfall'] * df['max_temperature']

    # fill
    df = df.bfill().ffill()

    # target
    series = TimeSeries.from_dataframe(df, 'date', 'RRP')

    feature_cols = [
        'min_temperature', 'max_temperature', 'solar_exposure', 'rainfall',
        'school_day', 'holiday',
        'dayofweek', 'month', 'weekofyear', 'is_weekend',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_mean_30',
        'temp_range', 'rain_x_temp'
    ]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.bfill().ffill()

    # ---- EXTEND into the future for BOTH past_cov and future_cov ----
    extend_days = TEST_H + VAL_H + 60  # safe buffer
    extra_days = pd.date_range(df['date'].max() + pd.Timedelta(days=1),
                               periods=extend_days, freq="D")
    extra = pd.DataFrame(index=extra_days)
    # deterministic calendar
    extra['dayofweek']  = extra_days.dayofweek
    extra['month']      = extra_days.month
    extra['weekofyear'] = extra_days.isocalendar().week.astype(int)
    extra['is_weekend'] = (extra_days.dayofweek >= 5).astype(int)
    # unknowns → conservative defaults
    extra['school_day'] = 0
    extra['holiday']    = 0
    # non-calendar just repeat last known value
    for col in feature_cols:
        if col not in extra.columns:
            extra[col] = df[col].iloc[-1]

    extra = extra.reset_index().rename(columns={'index':'date'})
    df_ext = pd.concat([df, extra], ignore_index=True)

    # Build extended covariates
    past_cov_ext = TimeSeries.from_dataframe(df_ext, 'date', feature_cols)   # for Transformer
    future_cov   = TimeSeries.from_dataframe(df_ext, 'date', feature_cols)   # for rf/lin/xgb/rnn/prophet

    # AIC param count for linear
    AIC_K["lin"] = len(LAGS_TARGET) + len(LAGS_FUT) * len(feature_cols) + 1

    return series, feature_cols, past_cov_ext, future_cov

def split_data(series, past_cov_ext, future_cov):
    """
    Slice covariates with *enough* coverage:
      - Transformer (past-covariates): must reach base_end + horizon when n > output_chunk_length.
      - Others (future-covariates): must reach base_end + horizon + max_future_lag.
    """
    train = series[:-(TEST_H + VAL_H)]
    val   = series[-(TEST_H + VAL_H):-TEST_H]
    test  = series[-TEST_H:]

    max_future_lag = max(LAGS_FUT) if LAGS_FUT else 0

    # ---- Transformer past-covariates (extended) ----
    past_train    = past_cov_ext[:len(train)]
    past_for_val  = past_cov_ext[:len(train) + VAL_H]                 # base(train) + val horizon
    past_for_test = past_cov_ext[:len(train) + len(val) + TEST_H]     # base(train+val) + test horizon

    # ---- Future covariates (extended) ----
    fut_train = future_cov[:len(train) + max_future_lag]
    fut_val   = future_cov[:len(train) + VAL_H + max_future_lag]
    fut_test  = future_cov[:len(train) + len(val) + TEST_H + max_future_lag]

    print(f"Splits → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    print(f"Transformer past_cov → train:{len(past_train)}  val:{len(past_for_val)}  test:{len(past_for_test)}")
    print(f"Future cov → train:{len(fut_train)}  val:{len(fut_val)}  test:{len(fut_test)}")
    print(f"max_future_lag: {max_future_lag}")
    print(f"past_for_test range: {past_for_test.start_time()} → {past_for_test.end_time()}")
    print(f"fut_test range:      {fut_test.start_time()} → {fut_test.end_time()}")
    print(f"test horizon:        {test.start_time()} → {test.end_time()}")

    return {
        'train': train, 'val': val, 'test': test,
        'past_train': past_train, 'past_for_val': past_for_val, 'past_for_test': past_for_test,
        'fut_train': fut_train, 'fut_val': fut_val, 'fut_test': fut_test
    }

# --------------------------
# Fit / Predict per model type
# --------------------------
def fit_model(model, model_name, ds):
    print(f"Training {model_name.upper()}...")
    if model_name == "transformer":
        model.fit(ds['train'], past_covariates=ds['past_train'])
    elif model_name in ("rnn", "lstm", "gru"):
        model.fit(ds['train'], future_covariates=ds['fut_train'], verbose=False)
    elif model_name == "prophet":
        model.fit(ds['train'], future_covariates=ds['fut_train'])
    else:  # rf / lin / xgb
        model.fit(ds['train'], future_covariates=ds['fut_train'])
    print(f"✓ {model_name.upper()} training completed")
    return model

def predict_model(model, model_name, ds, horizon, phase='test'):
    if phase == 'val':
        base_series = ds['train']
        fut_cov     = ds['fut_val']
        past_cov    = ds['past_for_val']
    else:
        base_series = ds['train'].append(ds['val'])
        fut_cov     = ds['fut_test']
        past_cov    = ds['past_for_test']

    if model_name == "transformer":
        return model.predict(horizon, series=base_series, past_covariates=past_cov, show_warnings=False)
    elif model_name in ("rnn", "lstm", "gru"):
        return model.predict(horizon, series=base_series, future_covariates=fut_cov, show_warnings=False)
    elif model_name == "prophet":
        return model.predict(horizon, series=base_series, future_covariates=fut_cov)
    else:  # rf / lin / xgb
        return model.predict(horizon, series=base_series, future_covariates=fut_cov, show_warnings=False)

# --------------------------
# Orchestration
# --------------------------
def train_ensemble(models_to_train: List[str],
                   csv_path: str = CSV_PATH,
                   results_dir: str = RESULTS_DIR,
                   enable_blending: bool = True):

    global CSV_PATH, RESULTS_DIR
    CSV_PATH, RESULTS_DIR = csv_path, results_dir

    available = ['rf', 'lin', 'xgb', 'rnn', 'lstm', 'gru', 'transformer', 'prophet']
    invalid = set(models_to_train) - set(available)
    if invalid:
        raise ValueError(f"Invalid model names: {invalid}. Available: {available}")

    print(f"Training models: {models_to_train}  |  Blending: {enable_blending}")

    series, feature_cols, past_cov_ext, future_cov = prepare_data()
    ds = split_data(series, past_cov_ext, future_cov)

    models, val_preds, test_preds = {}, {}, {}
    for name in models_to_train:
        m = ModelFactory.create_model(name)
        models[name] = fit_model(m, name, ds)
        if enable_blending:
            val_preds[name]  = predict_model(models[name], name, ds, VAL_H, 'val')
        test_preds[name] = predict_model(models[name], name, ds, TEST_H, 'test')

    # Blending
    blend_weights = None
    if enable_blending and len(models_to_train) > 1:
        print("\nLearning ensemble weights...")
        P_val = np.column_stack([val_preds[n].values().ravel() for n in models_to_train])
        y_val = ds['val'].values().ravel()
        blender = SkLinearBlend(fit_intercept=False).fit(P_val, y_val)
        w = np.maximum(blender.coef_, 0); w = w / w.sum() if w.sum() > 0 else np.ones(len(models_to_train))/len(models_to_train)
        blend_weights = w
        print(f"Learned weights {dict(zip(models_to_train, np.round(w, 3)))}")

        P_test = np.column_stack([test_preds[n].values().ravel() for n in models_to_train])
        blend_vals = P_test @ w
        test_preds["blend"] = TimeSeries.from_times_and_values(ds['test'].time_index, blend_vals)

        eq_w = np.ones(len(models_to_train)) / len(models_to_train)
        blend_eq_vals = P_test @ eq_w
        test_preds["blend_eq"] = TimeSeries.from_times_and_values(ds['test'].time_index, blend_eq_vals)

    # Metrics
    metrics = {}
    for name, pred in test_preds.items():
        k = AIC_K.get(name, AIC_K["blend"])
        add_metrics(metrics, name, ds['test'], pred, k)

    print("\n=== Test metrics ===")
    for name, vals in metrics.items():
        print(f"{name:12s} → MAE={vals['MAE']:.3f}  RMSE={vals['RMSE']:.3f}  R²={vals['R2']:.3f}  AIC={vals['AIC']:.1f}")
    if blend_weights is not None:
        print(f"Weights {dict(zip(models_to_train, np.round(blend_weights, 3)))}")

    # Tables + quick overlay
    _ensure_dir(results_dir)
    save_tables_and_base_overlays(results_dir, ds['train'], ds['val'], ds['test'],
                                  test_preds, blend_weights, models_to_train)

    # Plots per model
    plots_dir = _ensure_dir(os.path.join(results_dir, "plots"))
    train_plus_val = ds['train'].append(ds['val'])
    for name, pred in test_preds.items():
        title = f"Vorhersage mit {name.upper()}"
        plot_like_notebook(train_plus_val, ds['test'], pred, title, plots_dir, f"{name}_overlay_like_notebook.png")
        plot_true_vs_pred_subplots(ds['test'], pred, f"{name.upper()} — Test (True)", f"{name.upper()} — Forecast",
                                   plots_dir, f"{name}_true_vs_pred_subplots.png")
        plot_zoom_last(train_plus_val.append(ds['test']), ds['test'], pred, pre_days=60,
                       title=f"{name.upper()} — Zoomed Test Window", out_dir=plots_dir, fname=f"{name}_zoom_test.png")
        plot_parity(ds['test'], pred, plots_dir, f"{name}_parity.png", f"{name.upper()} — Parity")
        plot_residuals(ds['test'], pred, plots_dir, f"{name}_residuals.png", f"{name.upper()} — Residuals over Time")
        plot_rolling_mae(ds['test'], pred, window=14, out_dir=plots_dir,
                         fname=f"{name}_rolling_mae_14.png", title=f"{name.upper()} — Rolling MAE")

        if name in ("rf", "xgb") and isinstance(models[name], SKLearnModel):
            plot_feature_importance_if_any(
                name, models[name],
                feature_names=None,
                out_dir=plots_dir, fname=f"{name}_feature_importance.png",
                title=f"{name.upper()} — Feature Importance",
                lags_target=LAGS_TARGET, lags_fut=LAGS_FUT, cov_cols=feature_cols
            )

    payload = {"metrics": metrics}
    if blend_weights is not None:
        payload["weights"] = blend_weights.tolist()
        payload["model_names"] = models_to_train
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics → {os.path.join(results_dir, 'metrics.json')}")
    print(f"\nArtifacts saved in: {os.path.abspath(results_dir)}")

    return {"models": models, "predictions": test_preds, "metrics": metrics, "weights": blend_weights}

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    print("=== EXAMPLE: models with blending (incl. RNN/Transformer/Prophet) ===")
    _ = train_ensemble(
        models_to_train=['rf', 'lin', 'xgb', 'lstm', 'transformer', 'prophet'],
        results_dir="results_all_models",
        enable_blending=True
    )
