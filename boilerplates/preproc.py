# unified_preprocessor_demo_XY_keepcyclic_fixed.py
# ------------------------------------------------------------
# End-to-end demo:
# - Ensures ColumnTransformer outputs pandas (keeps feature names)
# - CorrelationSelector robust to arrays and mixed-type columns
# - Preserves sine/cosine cyclical pairs
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, List, Tuple
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# =========================
# Custom transformers
# =========================

class DateCyclicalFeatures(BaseEstimator, TransformerMixin):
    """Add calendar/cyclical features from DatetimeIndex or a 'date' column.
       Safe: will NOT overwrite if columns already exist (to avoid duplicates)."""
    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            if self.date_col not in X.columns:
                raise KeyError(
                    f"DateCyclicalFeatures: '{self.date_col}' not in columns "
                    "and index is not DatetimeIndex."
                )
            dates = pd.to_datetime(X[self.date_col])

        year      = dates.year
        month     = dates.month
        week      = dates.isocalendar().week.astype(int)
        dayofweek = dates.dayofweek
        dayofyear = dates.dayofyear

        def _set(col, series):
            if col not in X.columns:
                X[col] = series

        # raw calendar
        _set("year", year)
        _set("month", month)
        _set("week", week)
        _set("dayofweek", dayofweek)
        _set("dayofyear", dayofyear)

        # cyclical encodings
        _set("month_sin", np.sin(2 * np.pi * month / 12))
        _set("month_cos", np.cos(2 * np.pi * month / 12))
        _set("week_sin",  np.sin(2 * np.pi * week / 52))
        _set("week_cos",  np.cos(2 * np.pi * week / 52))
        _set("doy_sin",   np.sin(2 * np.pi * dayofyear / 365.25))
        _set("doy_cos",   np.cos(2 * np.pi * dayofyear / 365.25))
        return X


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Drop highly correlated features (keep one from any correlated pair).
    `keep_always` ensures critical columns (e.g., sine/cosine pairs) are never dropped.
    Robust to numpy arrays (no names) and mixed-type columns.
    """
    def __init__(self, threshold: float = 0.95, keep_always: Optional[List[str]] = None, verbose: bool = True):
        self.threshold = threshold
        self.keep_always = keep_always or []
        self.verbose = verbose
        self.selected_: Optional[List[str]] = None

    def fit(self, X, y=None):
        # If we get a numpy array, make a DF with numeric column names
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)

        num = X_df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            self.selected_ = list(X_df.columns)
            return self

        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # Only protect columns that actually exist in X
        keep_set = {k for k in self.keep_always if k in X_df.columns}

        drop = []
        for c in upper.columns:
            if c in keep_set:
                continue
            # If any correlation above threshold, mark for drop
            if any(upper[c] > self.threshold):
                drop.append(c)

        # Keep everything not in drop; this preserves original column order
        selected = [c for c in X_df.columns if c not in drop]

        # Ensure protected columns are included if present (they should be, but double-guard)
        for k in keep_set:
            if k not in selected:
                selected.append(k)

        self.selected_ = selected

        if self.verbose and drop:
            print(f":mag_right: CorrelationSelector dropped {len(drop)} features: {drop}")
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else pd.DataFrame(X)
        # Select only columns that exist (guards against any mismatch)
        cols = [c for c in self.selected_ if c in X_df.columns]
        return X_df[cols]


# =========================
# Main unified class
# =========================

@dataclass
class Preprocessor:
    filepath: str | Path
    date_col: str = "date"
    target_col: str = "RRP"

    # columns that may leak label info (we keep them in Y as optional targets)
    leaky_cols: Iterable[str] = field(default_factory=lambda: (
        "RRP_positive", "RRP_negative",
        "demand_pos_RRP", "demand_neg_RRP",
        "frac_at_neg_RRP",
    ))

    corr_threshold: float = 0.95
    add_date_features: bool = True
    random_state: int = 42

    # internal
    df: Optional[pd.DataFrame] = None
    pipeline_: Optional[Pipeline] = None

    def set_data(self, df: pd.DataFrame):
        """Supply a DataFrame and ensure DatetimeIndex."""
        if isinstance(df.index, pd.DatetimeIndex):
            self.df = df
            return self
        if self.date_col in df.columns:
            df = df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="raise")
            df = df.sort_values(self.date_col).set_index(self.date_col)
            self.df = df
            return self
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="raise")
            self.df = df
            return self
        except Exception:
            raise ValueError(
                f"set_data: DataFrame must have a DatetimeIndex or a '{self.date_col}' column."
            )

    # ----- Targets / Split -----
    def get_target_cols(self) -> List[str]:
        """Primary target + all leaky columns as potential targets."""
        return [self.target_col] + list(self.leaky_cols)

    def make_XY(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        X = df without target + leaky columns
        Y = df with [target] + leaky columns (only those present)
        """
        if self.df is None:
            raise ValueError("make_XY: call set_data(df) first.")
        target_cols = [c for c in self.get_target_cols() if c in self.df.columns]
        Y = self.df[target_cols].copy()
        X = self.df.drop(columns=target_cols, errors="ignore").copy()
        return X, Y

    # ----- Pipeline -----
    def build_pipeline(self) -> Pipeline:
        """
        Preprocessing pipeline for X:
          [Date features] -> ColumnTransformer -> CorrelationSelector
        Ensures ColumnTransformer outputs pandas to preserve column names.
        """
        ohe_kwargs = dict(handle_unknown="ignore", drop="first")
        try:
            ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)  # sklearn >= 1.2
        except TypeError:
            ohe = OneHotEncoder(sparse=False, **ohe_kwargs)         # older sklearn

        num_tf = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler()
        )

        cat_tf = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            ohe
        )

        num_sel = make_column_selector(dtype_include=["int64", "float64", "Int8", "Int16", "Int32", "Int64", "float32", "float64"])
        cat_sel = make_column_selector(dtype_include=["object", "string", "category"])

        pre_ct = make_column_transformer(
            (num_tf, num_sel),
            (cat_tf, cat_sel),
            remainder="drop"
        )

        # ✨ ensure downstream steps receive a DataFrame with names (sklearn >= 1.3)
        try:
            pre_ct.set_output(transform="pandas")
        except Exception:
            # If not available, downstream selector will still work,
            # but 'keep_always' will only apply when names are present.
            pass

        cyc_pairs = ["month_sin","month_cos","week_sin","week_cos","doy_sin","doy_cos"]

        steps = []
        if self.add_date_features:
            steps.append(("date_features", DateCyclicalFeatures(self.date_col)))
        steps.append(("pre", pre_ct))
        steps.append(("corr_prune", CorrelationSelector(
            threshold=self.corr_threshold,
            keep_always=cyc_pairs,
            verbose=True
        )))
        self.pipeline_ = Pipeline(steps)
        return self.pipeline_


# =========================
# DEMO: Fake data + X/Y + tracking
# =========================

if __name__ == "__main__":
    # 1) Fake data
    csv_text = """date,demand,RRP,demand_pos_RRP,RRP_positive,demand_neg_RRP,RRP_negative,frac_at_neg_RRP,min_temperature,max_temperature,solar_exposure,rainfall,school_day,holiday
2015-01-01,99635.03,25.63369643387471,97319.24000000002,26.415952619440922,2315.79,-7.239999999999997,0.020833334,13.3,26.9,23.6,0.0,N,Y
2015-01-02,129606.00999999994,33.13898756122499,121082.01499999994,38.837660977974316,8523.994999999999,-47.809776501511315,0.0625,15.4,38.8,26.8,0.0,N,N
2015-01-03,142300.53999999998,34.56485482908218,142300.53999999998,34.56485482908218,0.0,0.0,0.0,20.0,38.2,26.5,0.0,N,N
2015-01-04,104330.715,25.00556023842067,104330.715,25.00556023842067,0.0,0.0,0.0,16.3,21.4,25.2,4.2,N,N
2015-01-05,118132.19999999994,26.72417627793271,118132.19999999994,26.72417627793271,0.0,0.0,0.0,15.0,22.0,30.7,0.0,N,N
"""
    df_raw = pd.read_csv(StringIO(csv_text))

    # 2) Set data (ensures DatetimeIndex)
    pp = Preprocessor(filepath="(in-memory)")
    pp.set_data(df_raw)

    print("\n== INDEX TYPE ==")
    print(type(pp.df.index))

    print("\n== ORIGINAL COLUMNS ==")
    print(list(pp.df.columns))

    # 3) Split into X and Y
    X_raw, Y = pp.make_XY()
    print("\n== TARGET COLUMNS (Y) ==")
    print(list(Y.columns))

    print("\n== X COLUMNS BEFORE PIPELINE ==")
    print(list(X_raw.columns))

    # 4) Build pipeline for X
    pipe = pp.build_pipeline()
    date_step: DateCyclicalFeatures = pipe.named_steps["date_features"]
    pre_ct                           = pipe.named_steps["pre"]
    corr_step: CorrelationSelector   = pipe.named_steps["corr_prune"]

    # Step A: date features on X
    cols_before = set(X_raw.columns)
    X1 = date_step.fit_transform(X_raw)
    cols_after = set(X1.columns)
    added_cols = sorted(list(cols_after - cols_before))
    print("\n== DATE/CYCLICAL FEATURES ADDED TO X ==")
    print(added_cols)

    # Step B: ColumnTransformer on X
    pre_ct.fit(X1)
    try:
        ct_feature_names = list(pre_ct.get_feature_names_out())
    except Exception:
        # Fallback if very old sklearn (approximate names)
        num_mask = make_column_selector(dtype_include=["int64", "float64", "Int8", "Int16", "Int32", "Int64", "float32", "float64"])(X1)
        cat_mask = make_column_selector(dtype_include=["object", "string", "category"])(X1)
        num_cols = list(num_mask)
        cat_cols = list(cat_mask)
        cat_pipeline = pre_ct.transformers_[1][1]
        ohe = None
        for name, step in cat_pipeline.named_steps.items():
            if isinstance(step, OneHotEncoder):
                ohe = step
                break
        ohe_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else cat_cols
        ct_feature_names = num_cols + ohe_names

    X2 = pre_ct.transform(X1)  # thanks to set_output, this should be a DataFrame
    if not isinstance(X2, pd.DataFrame):
        # Defensive: if set_output not available, coerce to DataFrame now
        X2 = pd.DataFrame(X2, columns=ct_feature_names, index=X1.index)

    print("\n== COLUMNTRANSFORMER OUTPUT FEATURES (X) ==")
    print(list(X2.columns))

    # Step C: Correlation pruning on X (with keep_always for cyc pairs)
    X3 = corr_step.fit_transform(X2)
    print("\n== FINAL KEPT FEATURES AFTER CorrelationSelector (X) ==")
    print(list(X3.columns))

    print(f"\nShapes: X1={X1.shape}, X2={X2.shape}, X3={X3.shape}, Y={Y.shape}")

    # 5) End-to-end fit/transform on X only (sanity check)
    Xt_full = pipe.fit_transform(X_raw)
    print("\n== Full pipeline output shape for X ==")
    print(Xt_full.shape)

    # Peek Y
    print("\n== Y HEAD ==")
    print(Y.head())
