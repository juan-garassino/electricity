# # ml_pipeline.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer, make_column_selector
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_validate
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.base import BaseEstimator, TransformerMixin
# import warnings
# warnings.filterwarnings('ignore')

# # --- Optional libraries ---
# try:
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#     STATSMODELS_AVAILABLE = True
# except ImportError:
#     print("⚠ statsmodels not available. Install with: pip install statsmodels")
#     STATSMODELS_AVAILABLE = False

# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     print("⚠ XGBoost not available. Install with: pip install xgboost")
#     XGBOOST_AVAILABLE = False


# # ======================================================
# # Custom Transformers
# # ======================================================
# class DateCyclicalFeatures(BaseEstimator, TransformerMixin):
#     """Extract cyclical time features from DatetimeIndex"""
#     def __init__(self, use_index=True, date_col='date'):
#         self.use_index = use_index
#         self.date_col = date_col

#     def _get_dates(self, X):
#         if self.use_index and isinstance(X.index, pd.DatetimeIndex):
#             return X.index.to_series()
#         elif self.date_col in X.columns:
#             return pd.to_datetime(X[self.date_col])
#         else:
#             raise ValueError("No DatetimeIndex or 'date' column found.")

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         dates = self._get_dates(X)

#         X['year']      = dates.dt.year.astype(int)
#         X['month']     = dates.dt.month.astype(int)
#         X['week']      = dates.dt.isocalendar().week.astype(int)
#         X['dayofweek'] = dates.dt.dayofweek.astype(int)
#         X['dayofyear'] = dates.dt.dayofyear.astype(int)

#         # cyclical
#         X['month_sin'] = np.sin(2*np.pi*X['month']/12.0)
#         X['month_cos'] = np.cos(2*np.pi*X['month']/12.0)
#         X['week_sin']  = np.sin(2*np.pi*X['week']/52.0)
#         X['week_cos']  = np.cos(2*np.pi*X['week']/52.0)
#         X['doy_sin']   = np.sin(2*np.pi*X['dayofyear']/365.25)
#         X['doy_cos']   = np.cos(2*np.pi*X['dayofyear']/365.25)

#         return X


# class CorrelationSelector(BaseEstimator, TransformerMixin):
#     """Remove correlated features"""
#     def __init__(self, threshold=0.95):
#         self.threshold = threshold
#         self.selected_features_ = None

#     def fit(self, X, y=None):
#         X_df = pd.DataFrame(X)
#         corr_matrix = X_df.corr().abs()
#         upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#         drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
#         self.selected_features_ = [c for c in X_df.columns if c not in drop]
#         return self

#     def transform(self, X):
#         return pd.DataFrame(X)[self.selected_features_]


# class VIFSelector(BaseEstimator, TransformerMixin):
#     """Remove features with high VIF"""
#     def __init__(self, threshold=5.0):
#         self.threshold = threshold
#         self.selected_features_ = None

#     def fit(self, X, y=None):
#         if not STATSMODELS_AVAILABLE:
#             self.selected_features_ = list(pd.DataFrame(X).columns)
#             return self

#         X_df = pd.DataFrame(X).copy()
#         features = list(X_df.columns)
#         while True and len(features) > 1:
#             vif = [variance_inflation_factor(X_df[features].values, i) for i in range(len(features))]
#             max_vif = max(vif)
#             if max_vif <= self.threshold:
#                 break
#             remove = features[np.argmax(vif)]
#             features.remove(remove)
#         self.selected_features_ = features
#         return self

#     def transform(self, X):
#         return pd.DataFrame(X)[self.selected_features_]


# # ======================================================
# # Helpers
# # ======================================================
# def load_and_clean_data(filepath="complete_dataset.csv"):
#     df = pd.read_csv(filepath)
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)

#     # Convert flags
#     if 'holiday' in df.columns:
#         df['holiday'] = df['holiday'].map({'Y': 1, 'N': 0})
#     if 'school_day' in df.columns:
#         df['school_day'] = df['school_day'].map({'Y': 1, 'N': 0})

#     # Drop leaky features
#     drop_cols = ['RRP_positive', 'RRP_negative',
#                  'demand_pos_RRP', 'demand_neg_RRP',
#                  'frac_at_neg_RRP']
#     df = df.drop(columns=[c for c in drop_cols if c in df.columns])

#     X = df.drop(columns=['RRP'])
#     y = df['RRP']
#     return X, y


# def create_preprocessor():
#     num_tf = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
#     cat_tf = Pipeline([
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#         ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
#     ])
#     return ColumnTransformer([
#         ('num', num_tf, make_column_selector(dtype_include=['int64','float64'])),
#         ('cat', cat_tf, make_column_selector(dtype_include=['object']))
#     ])


# def get_models():
#     models = {
#         "Linear Regression": LinearRegression(),
#         "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     }
#     if XGBOOST_AVAILABLE:
#         models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
#     return models


# def evaluate_pipeline(pipeline, X, y):
#     cv = cross_validate(pipeline, X, y, cv=5,
#                         scoring=['r2','neg_mean_squared_error'],
#                         return_train_score=True)
#     return {
#         "r2": cv['test_r2'].mean(),
#         "rmse": np.sqrt(-cv['test_neg_mean_squared_error']).mean(),
#         "train_r2": cv['train_r2'].mean(),
#         "overfit": cv['train_r2'].mean() - cv['test_r2'].mean()
#     }


# # ======================================================
# # Main Experiment
# # ======================================================
# def run_experiment(X, y, selector="correlation"):
#     if selector=="correlation":
#         fs = CorrelationSelector()
#     elif selector=="vif":
#         fs = VIFSelector()
#     elif selector=="pca":
#         fs = PCA(n_components=0.95)
#     else:
#         fs = None

#     results = {}
#     for name, model in get_models().items():
#         steps = [
#             ("time", DateCyclicalFeatures()),
#             ("prep", create_preprocessor())
#         ]
#         if fs: steps.append(("fs", fs))
#         steps.append(("model", model))
#         pipe = Pipeline(steps)

#         try:
#             res = evaluate_pipeline(pipe, X, y)
#             results[name] = res
#             print(f"{name:15} | R²={res['r2']:.3f} | RMSE={res['rmse']:.2f} | Overfit={res['overfit']:.3f}")
#         except Exception as e:
#             print(f"{name} failed: {e}")
#     return results


# # ======================================================
# # Visualization
# # ======================================================
# def plot_results(all_results):
#     df = []
#     for sel, models in all_results.items():
#         for m, res in models.items():
#             df.append([sel, m, res['r2'], res['rmse'], res['overfit']])
#     df = pd.DataFrame(df, columns=["Selector","Model","R2","RMSE","Overfit"])

#     fig, axes = plt.subplots(1,3,figsize=(16,5))
#     sns.barplot(df, x="Model", y="R2", hue="Selector", ax=axes[0])
#     sns.barplot(df, x="Model", y="RMSE", hue="Selector", ax=axes[1])
#     sns.barplot(df, x="Model", y="Overfit", hue="Selector", ax=axes[2])
#     for ax in axes: ax.tick_params(axis='x', rotation=30)
#     plt.tight_layout(); plt.show()
#     return df


# # ======================================================
# # Main
# # ======================================================
# def main():
#     print("🚀 Electricity Price Prediction ML Comparison")
#     X, y = load_and_clean_data()

#     all_results = {}
#     for sel in ["correlation","vif","pca"]:
#         print("\n=== Feature Selection:", sel.upper(), "===")
#         all_results[sel] = run_experiment(X, y, selector=sel)

#     summary = plot_results(all_results)
#     print("\nSummary:\n", summary.round(3))


# if __name__=="__main__":
#     main()



# # energy_price_experiment.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# from typing import Dict, List, Optional
# from dataclasses import dataclass

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer, make_column_selector
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import TimeSeriesSplit, cross_validate
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor

# # --- Optional libs ---
# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except Exception:
#     print("⚠ XGBoost not installed. `pip install xgboost` for that model.")
#     XGBOOST_AVAILABLE = False

# try:
#     import statsmodels.api as sm
#     STATSMODELS_AVAILABLE = True
# except Exception:
#     print("⚠ statsmodels not installed. `pip install statsmodels` for ARIMA.")
#     STATSMODELS_AVAILABLE = False

# try:
#     import tensorflow as tf
#     from tensorflow.keras import Sequential
#     from tensorflow.keras.layers import LSTM, Dense
#     TF_AVAILABLE = True
# except Exception:
#     print("⚠ TensorFlow not installed. `pip install tensorflow` for LSTM.")
#     TF_AVAILABLE = False


# # =============== Transformers ===============

# class CorrelationSelector(BaseEstimator, TransformerMixin):
#     """Drop highly correlated features after numeric/categorical preprocessing."""
#     def __init__(self, threshold: float = 0.95):
#         self.threshold = threshold
#         self.selected_: Optional[List[str]] = None
#         self.dropped_: Optional[List[str]] = None

#     def fit(self, X, y=None):
#         X_df = pd.DataFrame(X)
#         corr = X_df.corr(numeric_only=True).abs()
#         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
#         self.selected_ = [c for c in X_df.columns if c not in drop]
#         self.dropped_ = drop
#         if drop:
#             print(f"🔎 CorrelationSelector dropped {len(drop)} features.")
#         return self

#     def transform(self, X):
#         X_df = pd.DataFrame(X)
#         return X_df[self.selected_].values


# # =============== Preprocessor ===============

# @dataclass
# class PreprocessConfig:
#     filepath: str = "complete_dataset.csv"
#     lags: Dict[str, List[int]] = None  # e.g., {"RRP":[1,7,30], "demand":[1,7,30]}
#     corr_threshold: float = 0.95
#     save_snapshot_path: str = "preprocessed_snapshot.csv"

#     def __post_init__(self):
#         if self.lags is None:
#             self.lags = {"RRP": [1, 7, 30]}  # add more keys (e.g., "demand") if present


# class DataPreprocessor:
#     def __init__(self, config: PreprocessConfig):
#         self.cfg = config
#         self._raw_df: Optional[pd.DataFrame] = None

#     # --------- data loading / cleaning ----------
#     def load(self) -> pd.DataFrame:
#         df = pd.read_csv(self.cfg.filepath)
#         df["date"] = pd.to_datetime(df["date"])
#         df = df.sort_values("date").set_index("date")

#         # binary maps
#         if "holiday" in df.columns:
#             df["holiday"] = df["holiday"].map({"Y": 1, "N": 0})
#         if "school_day" in df.columns:
#             df["school_day"] = df["school_day"].map({"Y": 1, "N": 0})

#         # drop leaky features if present
#         leaky = ["RRP_positive", "RRP_negative",
#                  "demand_pos_RRP", "demand_neg_RRP",
#                  "frac_at_neg_RRP"]
#         present = [c for c in leaky if c in df.columns]
#         if present:
#             df = df.drop(columns=present)
#             print(f"🧹 Dropped leaky columns: {present}")

#         self._raw_df = df
#         return df

#     # --------- feature engineering ----------
#     @staticmethod
#     def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
#         if not isinstance(df.index, pd.DatetimeIndex):
#             raise ValueError("DataFrame must be indexed by DatetimeIndex.")
#         X = df.copy()
#         dates = X.index

#         X["year"] = dates.year
#         X["month"] = dates.month
#         X["week"] = dates.isocalendar().week.astype(int)
#         X["dayofweek"] = dates.dayofweek
#         X["dayofyear"] = dates.dayofyear

#         X["month_sin"] = np.sin(2*np.pi*X["month"]/12.0)
#         X["month_cos"] = np.cos(2*np.pi*X["month"]/12.0)
#         X["week_sin"]  = np.sin(2*np.pi*X["week"]/52.0)
#         X["week_cos"]  = np.cos(2*np.pi*X["week"]/52.0)
#         X["doy_sin"]   = np.sin(2*np.pi*X["dayofyear"]/365.25)
#         X["doy_cos"]   = np.cos(2*np.pi*X["dayofyear"]/365.25)
#         return X

#     def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
#         X = df.copy()
#         for col, lags in self.cfg.lags.items():
#             if col not in X.columns:
#                 print(f"⚠ lag base column '{col}' not found; skipping its lags.")
#                 continue
#             for lag in lags:
#                 X[f"{col}_lag{lag}"] = X[col].shift(lag)
#         return X

#     def build_design_matrix(self) -> (pd.DataFrame, pd.Series):
#         if self._raw_df is None:
#             self.load()
#         df = self._raw_df.copy()

#         if "RRP" not in df.columns:
#             raise ValueError("Target column 'RRP' not found.")
#         # add time features + lags
#         df = self._add_cyclical_time_features(df)
#         df = self._add_lags(df)

#         # Drop rows with NaN introduced by lags
#         df_clean = df.dropna().copy()

#         y = df_clean["RRP"]
#         X = df_clean.drop(columns=["RRP"])
#         return X, y

#     # --------- sklearn preprocessor ----------
#     @staticmethod
#     def make_column_transformer() -> ColumnTransformer:
#         num_tf = Pipeline([
#             ("imp", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ])
#         cat_tf = Pipeline([
#             ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
#             ("oh", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
#         ])
#         pre = ColumnTransformer([
#             ("num", num_tf, make_column_selector(dtype_include=["int64", "float64"])),
#             ("cat", cat_tf, make_column_selector(dtype_include=["object"])),
#         ])
#         return pre

#     def make_ml_pipeline(self, model, save_snapshot: bool = False, snapshot_tag: str = "") -> Pipeline:
#         steps = [
#             ("prep", self.make_column_transformer()),
#             ("corr", CorrelationSelector(threshold=self.cfg.corr_threshold)),
#         ]
#         if save_snapshot:
#             # Save after preprocessing + correlation selection using a simple wrapper
#             steps.append(("save", _SaveOnce(path=self._snapshot_path(snapshot_tag))))
#         steps.append(("model", model))
#         return Pipeline(steps)

#     def _snapshot_path(self, tag: str) -> str:
#         base = self.cfg.save_snapshot_path
#         if tag:
#             dot = base.rfind(".")
#             if dot == -1:
#                 return f"{base}_{tag}"
#             return f"{base[:dot]}_{tag}{base[dot:]}"
#         return base

#     def save_full_preprocessed_snapshot(self, X: pd.DataFrame, tag: str = "full"):
#         """
#         Fit the column transformer on all X (past data), run correlation selection,
#         and write a single CSV snapshot. This is for **artifacting**, not for training.
#         """
#         pre = self.make_column_transformer()
#         Xp = pre.fit_transform(X)
#         # run correlation selection once for snapshot consistency
#         corr = CorrelationSelector(threshold=self.cfg.corr_threshold)
#         Xc = corr.fit_transform(Xp)
#         df_out = pd.DataFrame(Xc)
#         path = self._snapshot_path(tag)
#         df_out.to_csv(path, index=False)
#         print(f"✅ Saved preprocessed snapshot to {path}")


# # helper to save preprocessed data inside a Pipeline only once (first call)
# class _SaveOnce(BaseEstimator, TransformerMixin):
#     def __init__(self, path: str):
#         self.path = path
#         self._saved = False

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         if not self._saved:
#             pd.DataFrame(X).to_csv(self.path, index=False)
#             print(f"💾 Pipeline snapshot saved to {self.path}")
#             self._saved = True
#         return X


# # =============== Trainer ===============

# class ModelTrainer:
#     def __init__(self, preproc: DataPreprocessor, X: pd.DataFrame, y: pd.Series):
#         self.preproc = preproc
#         self.X = X
#         self.y = y
#         self.results = []

#     def _evaluate_cv(self, pipe: Pipeline, model_name: str, n_splits: int = 5):
#         tscv = TimeSeriesSplit(n_splits=n_splits)
#         cv = cross_validate(pipe, self.X, self.y, cv=tscv, scoring="r2",
#                             return_estimator=True)
#         r2_scores = cv["test_score"]
#         rmse_scores = []
#         for est, (_, test_idx) in zip(cv["estimator"], tscv.split(self.X)):
#             y_pred = est.predict(self.X.iloc[test_idx])
#             rmse_scores.append(np.sqrt(mean_squared_error(self.y.iloc[test_idx], y_pred)))

#         self.results.append({
#             "Model": model_name,
#             "R2_mean": float(np.mean(r2_scores)),
#             "R2_std": float(np.std(r2_scores)),
#             "RMSE_mean": float(np.mean(rmse_scores)),
#             "RMSE_std": float(np.std(rmse_scores)),
#         })

#     # --------- classical ML ---------
#     def run_ml_block(self, save_snapshot_per_model: bool = True):
#         models = {
#             "LinearRegression": LinearRegression(),
#             "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
#         }
#         if XGBOOST_AVAILABLE:
#             models["XGBoost"] = xgb.XGBRegressor(
#                 n_estimators=400, learning_rate=0.05, subsample=0.8,
#                 colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
#             )

#         for name, model in models.items():
#             pipe = self.preproc.make_ml_pipeline(
#                 model=model,
#                 save_snapshot=save_snapshot_per_model,
#                 snapshot_tag=name
#             )
#             print(f"\n=== CV: {name} ===")
#             self._evaluate_cv(pipe, name)

#     # --------- ARIMA (univariate) ---------
#     def run_arima(self, forecast_horizon: int = 30, order=(5,1,0)):
#         if not STATSMODELS_AVAILABLE:
#             return
#         series = self.y.copy()
#         train, test = series.iloc[:-forecast_horizon], series.iloc[-forecast_horizon:]
#         model = sm.tsa.ARIMA(train, order=order)
#         res = model.fit()
#         pred = res.forecast(steps=forecast_horizon)
#         r2 = r2_score(test, pred)
#         rmse = np.sqrt(mean_squared_error(test, pred))
#         self.results.append({
#             "Model": f"ARIMA{order}",
#             "R2_mean": float(r2), "R2_std": 0.0,
#             "RMSE_mean": float(rmse), "RMSE_std": 0.0
#         })
#         print(f"\n=== ARIMA{order} ===\nR2={r2:.4f}  RMSE={rmse:.2f}")

#     # --------- LSTM (univariate) ---------
#     @staticmethod
#     def _window(series: np.ndarray, window: int = 30):
#         Xw, Yw = [], []
#         for i in range(len(series)-window):
#             Xw.append(series[i:i+window])
#             Yw.append(series[i+window])
#         return np.array(Xw), np.array(Yw)

#     def run_lstm(self, window: int = 30, epochs: int = 5, test_ratio: float = 0.2):
#         if not TF_AVAILABLE:
#             return
#         series = self.y.values.astype("float32")
#         Xw, Yw = self._window(series, window)
#         Xw = Xw.reshape((Xw.shape[0], Xw.shape[1], 1))

#         split = int(len(Xw)*(1 - test_ratio))
#         Xtr, Xte = Xw[:split], Xw[split:]
#         Ytr, Yte = Yw[:split], Yw[split:]

#         model = Sequential([LSTM(64, input_shape=(window,1)), Dense(1)])
#         model.compile(optimizer="adam", loss="mse")
#         model.fit(Xtr, Ytr, epochs=epochs, batch_size=32, verbose=0)

#         pred = model.predict(Xte, verbose=0).flatten()
#         r2 = r2_score(Yte, pred)
#         rmse = np.sqrt(mean_squared_error(Yte, pred))
#         self.results.append({
#             "Model": f"LSTM(w={window})",
#             "R2_mean": float(r2), "R2_std": 0.0,
#             "RMSE_mean": float(rmse), "RMSE_std": 0.0
#         })
#         print(f"\n=== LSTM(window={window}) ===\nR2={r2:.4f}  RMSE={rmse:.2f}")

#     # --------- reporting ---------
#     def summary_df(self) -> pd.DataFrame:
#         df = pd.DataFrame(self.results)
#         order = ["Model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std"]
#         if not df.empty:
#             df = df[order]
#         return df.sort_values("R2_mean", ascending=False)

#     def plot(self):
#         df = self.summary_df()
#         if df.empty:
#             print("No results to plot.")
#             return
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#         axes[0].bar(df["Model"], df["R2_mean"])
#         axes[0].set_title("R² (higher is better)")
#         axes[0].tick_params(axis="x", rotation=30)
#         axes[1].bar(df["Model"], df["RMSE_mean"])
#         axes[1].set_title("RMSE (lower is better)")
#         axes[1].tick_params(axis="x", rotation=30)
#         plt.tight_layout()
#         plt.show()


# # =============== Main runner ===============

# def main():
#     # --- Configure preprocessing (add more lag bases if present, e.g., "demand") ---
#     cfg = PreprocessConfig(
#         filepath="complete_dataset.csv",
#         lags={"RRP": [1, 7, 30], "demand": [1, 7, 30]},  # "demand" is optional, will be skipped if absent
#         corr_threshold=0.95,
#         save_snapshot_path="preprocessed_snapshot.csv"
#     )
#     pre = DataPreprocessor(cfg)

#     # --- Build design matrix (adds cyclical + lags; drops NA from lags) ---
#     X, y = pre.build_design_matrix()
#     print(f"✅ Design matrix built: X={X.shape}, y={y.shape}")

#     # Save one global snapshot of fully preprocessed features (for lineage)
#     pre.save_full_preprocessed_snapshot(X, tag="full")

#     # --- Train/evaluate ---
#     trainer = ModelTrainer(pre, X, y)
#     trainer.run_ml_block(save_snapshot_per_model=True)  # Linear / RF / (XGB if installed)
#     trainer.run_arima(forecast_horizon=30, order=(5,1,0))
#     trainer.run_lstm(window=30, epochs=5)

#     # --- Report ---
#     summary = trainer.summary_df()
#     print("\n=== Final Results ===")
#     print(summary.round(4))
#     trainer.plot()


# if __name__ == "__main__":
#     main()

# electricity_price_benchmark.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ================== sklearn deps ==================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =============== optional libraries =================
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    print("⚠ XGBoost not installed. `pip install xgboost` to include it.")
    XGBOOST_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    print("⚠ statsmodels not installed. `pip install statsmodels` for ARIMA.")
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    print("⚠ TensorFlow not installed. `pip install tensorflow` for LSTM.")
    TF_AVAILABLE = False


# ================== Helpers ==================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ================== Transformers ==================
class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Drop highly correlated columns (post-preprocessing)."""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.selected_: Optional[List[str]] = None
        self.dropped_: Optional[List[str]] = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr = X_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.selected_ = [c for c in X_df.columns if c not in drop]
        self.dropped_ = drop
        if drop:
            print(f"🔎 CorrelationSelector dropped {len(drop)} features.")
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.selected_].values


class _SaveOnce(BaseEstimator, TransformerMixin):
    """Internal: save a copy of transformed features only once (for lineage)."""
    def __init__(self, path: str):
        self.path = path
        self._saved = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self._saved:
            pd.DataFrame(X).to_csv(self.path, index=False)
            print(f"💾 Snapshot saved to {self.path}")
            self._saved = True
        return X


# ================== Preprocessor ==================
@dataclass
class PreprocessConfig:
    filepath: str = "complete_dataset.csv"
    lags: Dict[str, List[int]] = None          # e.g., {"RRP":[1,7,30], "demand":[1,7,30]}
    corr_threshold: float = 0.95
    snapshot_path: str = "preprocessed_snapshot.csv"

    def __post_init__(self):
        if self.lags is None:
            self.lags = {"RRP": [1, 7, 30]}     # default


class DataPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._raw: Optional[pd.DataFrame] = None

    # ---------- load & clean ----------
    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        # map flags
        if "holiday" in df.columns:
            df["holiday"] = df["holiday"].map({"Y": 1, "N": 0})
        if "school_day" in df.columns:
            df["school_day"] = df["school_day"].map({"Y": 1, "N": 0})

        # remove leaky columns
        leaky = ["RRP_positive", "RRP_negative", "demand_pos_RRP",
                 "demand_neg_RRP", "frac_at_neg_RRP"]
        drop_cols = [c for c in leaky if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"🧹 Dropped leaky columns: {drop_cols}")

        self._raw = df
        return df

    # ---------- feature engineering ----------
    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex.")
        X = df.copy()
        dt = X.index
        X["year"] = dt.year
        X["month"] = dt.month
        X["week"] = dt.isocalendar().week.astype(int)
        X["dayofweek"] = dt.dayofweek
        X["dayofyear"] = dt.dayofyear

        # cyclical
        X["month_sin"] = np.sin(2*np.pi*X["month"]/12)
        X["month_cos"] = np.cos(2*np.pi*X["month"]/12)
        X["week_sin"]  = np.sin(2*np.pi*X["week"]/52)
        X["week_cos"]  = np.cos(2*np.pi*X["week"]/52)
        X["doy_sin"]   = np.sin(2*np.pi*X["dayofyear"]/365.25)
        X["doy_cos"]   = np.cos(2*np.pi*X["dayofyear"]/365.25)
        return X

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for col, lags in self.cfg.lags.items():
            if col not in X.columns:
                print(f"⚠ lag base '{col}' not found; skipping.")
                continue
            for L in lags:
                X[f"{col}_lag{L}"] = X[col].shift(L)
        return X

    def build_design(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._raw is None:
            self.load()
        df = self._add_time_features(self._raw)
        df = self._add_lags(df)
        df = df.dropna()

        if "RRP" not in df.columns:
            raise ValueError("Target 'RRP' missing.")

        y = df["RRP"].copy()
        X = df.drop(columns=["RRP"]).copy()
        return X, y

    # ---------- preprocessing block ----------
    @staticmethod
    def build_column_transformer() -> ColumnTransformer:
        num = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])
        return ColumnTransformer([
            ("num", num, make_column_selector(dtype_include=["int64", "float64"])),
            ("cat", cat, make_column_selector(dtype_include=["object"]))
        ])

    def make_pipeline(self, model, snapshot_tag: str = "") -> Pipeline:
        path = self._tagged_snapshot(snapshot_tag)
        return Pipeline([
            ("prep", self.build_column_transformer()),
            ("corr", CorrelationSelector(self.cfg.corr_threshold)),
            ("save", _SaveOnce(path)),  # save once for lineage
            ("model", model)
        ])

    def save_full_snapshot(self, X: pd.DataFrame, tag: str = "full"):
        pre = self.build_column_transformer()
        Xp = pre.fit_transform(X)
        corr = CorrelationSelector(self.cfg.corr_threshold)
        Xc = corr.fit_transform(Xp)
        pd.DataFrame(Xc).to_csv(self._tagged_snapshot(tag), index=False)
        print(f"✅ Full preprocessed snapshot saved: {self._tagged_snapshot(tag)}")

    def _tagged_snapshot(self, tag: str) -> str:
        base = self.cfg.snapshot_path
        if not tag:
            return base
        i = base.rfind(".")
        return f"{base[:i]}_{tag}{base[i:]}" if i != -1 else f"{base}_{tag}"


# ================== Trainer ==================
class ModelTrainer:
    def __init__(self, pre: DataPreprocessor, X: pd.DataFrame, y: pd.Series,
                 n_splits: int = 5, horizon: int = 30):
        self.pre = pre
        self.X = X
        self.y = y
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.h = horizon
        self.rows: List[dict] = []

    # ---------- classical ML ----------
    def run_ml(self):
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, max_depth=6, random_state=42, n_jobs=-1, verbosity=0
            )

        for name, model in models.items():
            print(f"\n=== CV (ML): {name} ===")
            pipe = self.pre.make_pipeline(model, snapshot_tag=name)
            # R2 via cross_validate; RMSE computed explicitly per fold
            cv = cross_validate(pipe, self.X, self.y, cv=self.tscv, scoring="r2",
                                return_estimator=True)
            r2s = list(cv["test_score"])
            rmses = []
            for est, (_, test_idx) in zip(cv["estimator"], self.tscv.split(self.X)):
                yhat = est.predict(self.X.iloc[test_idx])
                rmses.append(rmse(self.y.iloc[test_idx], yhat))

            self.rows.append({
                "Model": name,
                "R2_mean": float(np.mean(r2s)),
                "R2_std": float(np.std(r2s)),
                "RMSE_mean": float(np.mean(rmses)),
                "RMSE_std": float(np.std(rmses))
            })

    # ---------- ARIMA (rolling CV with fixed horizon) ----------
    def run_arima_cv(self, order=(5,1,0)):
        if not STATSMODELS_AVAILABLE:
            return
        print(f"\n=== CV (ARIMA{order}) ===")
        r2s, rmses = [], []
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(self.y), 1):
            train, test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            # forecast only up to horizon (or length of test)
            steps = min(self.h, len(test))
            model = sm.tsa.ARIMA(train, order=order)
            res = model.fit()
            preds = res.forecast(steps=steps)
            # truncate to comparable length
            r2s.append(r2_score(test[:steps], preds))
            rmses.append(rmse(test[:steps], preds))
            print(f"  fold {fold}: R2={r2s[-1]:.4f}  RMSE={rmses[-1]:.2f}")

        self.rows.append({
            "Model": f"ARIMA{order}",
            "R2_mean": float(np.mean(r2s)),
            "R2_std": float(np.std(r2s)),
            "RMSE_mean": float(np.mean(rmses)),
            "RMSE_std": float(np.std(rmses))
        })

    # ---------- LSTM (rolling CV, recursive horizon forecast) ----------
    @staticmethod
    def _make_sequences(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        Xs, Ys = [], []
        for i in range(len(series) - window):
            Xs.append(series[i:i+window])
            Ys.append(series[i+window])
        return np.array(Xs), np.array(Ys)

    def run_lstm_cv(self, window: int = 30, epochs: int = 5):
        if not TF_AVAILABLE:
            return
        print(f"\n=== CV (LSTM, window={window}) ===")
        r2s, rmses = [], []

        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(self.y), 1):
            train, test = self.y.iloc[train_idx].values.astype("float32"), self.y.iloc[test_idx].values.astype("float32")
            # Build sequences on the training series
            Xtr, Ytr = self._make_sequences(train, window)
            if len(Xtr) < 10:  # too small
                continue
            Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))

            model = Sequential([LSTM(64, input_shape=(window,1)), Dense(1)])
            model.compile(optimizer="adam", loss="mse")
            model.fit(Xtr, Ytr, epochs=epochs, batch_size=32, verbose=0)

            # Recursive predict for 'h' steps from the end of train
            steps = min(self.h, len(test))
            last = train[-window:].reshape(1, window, 1)
            preds = []
            for _ in range(steps):
                nxt = model.predict(last, verbose=0)[0, 0]
                preds.append(nxt)
                # slide window
                last = np.append(last[:,1:,:], [[[nxt]]], axis=1)

            r2s.append(r2_score(test[:steps], preds))
            rmses.append(rmse(test[:steps], preds))
            print(f"  fold {fold}: R2={r2s[-1]:.4f}  RMSE={rmses[-1]:.2f}")

        if r2s:
            self.rows.append({
                "Model": f"LSTM(w={window})",
                "R2_mean": float(np.mean(r2s)),
                "R2_std": float(np.std(r2s)),
                "RMSE_mean": float(np.mean(rmses)),
                "RMSE_std": float(np.std(rmses))
            })

    # ---------- reporting ----------
    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        cols = ["Model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std"]
        return df[cols].sort_values(["R2_mean", "RMSE_mean"], ascending=[False, True])

    def plot(self):
        df = self.summary()
        if df.empty:
            print("No results to plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(df["Model"], df["R2_mean"])
        axes[0].set_title("R² (higher is better)")
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].bar(df["Model"], df["RMSE_mean"])
        axes[1].set_title("RMSE (lower is better)")
        axes[1].tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.show()


# ================== Main ==================
def main():
    # --- configure ---
    cfg = PreprocessConfig(
        filepath="complete_dataset.csv",        # <-- your CSV
        lags={"RRP": [1, 7, 30], "demand": [1, 7, 30]},  # demand lags are optional
        corr_threshold=0.95,
        snapshot_path="preprocessed_snapshot.csv"
    )

    # --- build design matrix ---
    pre = DataPreprocessor(cfg)
    X, y = pre.build_design()
    print(f"✅ Design matrix: X={X.shape}, y={y.shape}")
    pre.save_full_snapshot(X, tag="full")

    # --- train/eval ---
    trainer = ModelTrainer(pre, X, y, n_splits=5, horizon=30)
    trainer.run_ml()
    trainer.run_arima_cv(order=(5,1,0))
    trainer.run_lstm_cv(window=30, epochs=5)

    # --- report ---
    summary = trainer.summary()
    print("\n=== Final Results (rolling CV, comparable) ===")
    print(summary.round(4))
    trainer.plot()


if __name__ == "__main__":
    main()
