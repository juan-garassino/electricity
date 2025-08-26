# preprocessing.py
"""
Feature engineering pipeline for energy price forecasting.
Creates enriched features from raw data including lags, rolling stats, and interactions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Sanitize(BaseEstimator, TransformerMixin):
    """Ensure datetime index and map flags. Optionally keep leaky cols."""
    def __init__(self, date_col: str = "date", keep_leaky: bool = False):
        self.date_col = date_col
        self.keep_leaky = keep_leaky
        self.leaky_cols = [
            "demand_pos_RRP", "RRP_positive",
            "demand_neg_RRP", "RRP_negative", 
            "frac_at_neg_RRP",
        ]
        self.bin_map = {"Y": 1, "N": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.date_col not in df.columns:
            raise ValueError(f"Missing '{self.date_col}' column.")
            
        # Convert to datetime and sort
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col]).sort_values(self.date_col).set_index(self.date_col)

        # Handle leaky columns
        if not self.keep_leaky:
            df = df.drop(columns=[c for c in self.leaky_cols if c in df.columns], errors="ignore")
        
        # Map binary flags
        if "school_day" in df.columns:
            df["school_day"] = df["school_day"].map(self.bin_map).fillna(df["school_day"]).astype(int)
        if "holiday" in df.columns:
            df["holiday"] = df["holiday"].map(self.bin_map).fillna(df["holiday"]).astype(int)
            
        return df


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Add calendar, cyclical sin/cos + Fourier harmonics."""
    def __init__(self, add_harmonics: bool = True, harmonics: tuple = (1, 2, 3)):
        self.add_harmonics = add_harmonics
        self.harmonics = harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        dt = df.index
        
        # Basic calendar features
        df["year"] = dt.year
        df["month"] = dt.month
        df["week"] = dt.isocalendar().week.astype(int)
        df["dayofweek"] = dt.dayofweek
        df["dayofyear"] = dt.dayofyear
        df["quarter"] = dt.quarter
        df["month_end"] = dt.is_month_end.astype(int)
        
        # Cyclical encodings
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        
        # Yearly Fourier harmonics
        if self.add_harmonics:
            for k in self.harmonics:
                df[f"year_sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
                df[f"year_cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)
                
        return df


class RollingStats(BaseEstimator, TransformerMixin):
    """Rolling mean/std/min/max for selected columns."""
    def __init__(self, cols: tuple = ("RRP", "demand"), windows: tuple = (7, 30)):
        self.cols = cols
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        for col in self.cols:
            if col in df.columns:
                for w in self.windows:
                    df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
                    df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
                    
                # Additional rolling stats for 7-day window
                df[f"{col}_roll7_min"] = df[col].rolling(7, min_periods=1).min()
                df[f"{col}_roll7_max"] = df[col].rolling(7, min_periods=1).max()
        
        # Simple regime flag for RRP
        if "RRP" in df.columns:
            m30 = df["RRP"].rolling(30, min_periods=1).mean()
            s30 = df["RRP"].rolling(30, min_periods=1).std().fillna(0.0)
            df["RRP_high_vol"] = (df["RRP"] > m30 + 2 * s30).astype(int)
            
        return df


class Interactions(BaseEstimator, TransformerMixin):
    """Key interactions (demand x temps, calendar, weather)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Temperature interactions
        if "demand" in df.columns and "min_temperature" in df.columns:
            df["demand_x_minT"] = df["demand"] * df["min_temperature"]
        if "demand" in df.columns and "max_temperature" in df.columns:
            df["demand_x_maxT"] = df["demand"] * df["max_temperature"]
            
        # Calendar interactions
        if "demand" in df.columns and "holiday" in df.columns:
            df["demand_x_holiday"] = df["demand"] * df["holiday"]
        if "demand" in df.columns and "school_day" in df.columns:
            df["demand_x_school"] = df["demand"] * df["school_day"]
            
        # Weather interactions
        if "rainfall" in df.columns and "solar_exposure" in df.columns:
            df["rain_x_solar"] = df["rainfall"] * df["solar_exposure"]
            
        return df


class LagFeatures(BaseEstimator, TransformerMixin):
    """Create lagged columns based on a dict: {col: [lags,...]}."""
    def __init__(self, lag_map: Dict[str, List[int]]):
        self.lag_map = lag_map or {}

    def fit(self, X, y=None):
        missing = [c for c in self.lag_map if c not in X.columns]
        if missing:
            print(f"[LagFeatures] Warning: missing columns {missing} (will be skipped).")
        return self

    def transform(self, X):
        df = X.copy()
        blocks = []
        
        for col, lags in self.lag_map.items():
            if col not in df.columns:
                continue
            block = {f"{col}_lag{k}": df[col].shift(k) for k in lags}
            blocks.append(pd.DataFrame(block, index=df.index))
            
        if blocks:
            lag_df = pd.concat(blocks, axis=1)
            df = pd.concat([df, lag_df], axis=1)
            
        return df


class CleanFinite(BaseEstimator, TransformerMixin):
    """Replace infs, and optionally drop rows with NA created by lag/rolling."""
    def __init__(self, drop_na: bool = True, fill_method: str = "forward"):
        self.drop_na = drop_na
        self.fill_method = fill_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.replace([np.inf, -np.inf], np.nan)
        
        if not self.drop_na and self.fill_method:
            if self.fill_method == "forward":
                df = df.fillna(method="ffill")
            elif self.fill_method == "backward":
                df = df.fillna(method="bfill")
            elif self.fill_method == "zero":
                df = df.fillna(0)
        elif self.drop_na:
            df = df.dropna()
            
        return df


def build_preprocessing_pipeline(
    date_col: str = "date",
    keep_leaky: bool = False,
    lag_map: Optional[Dict[str, List[int]]] = None,
    rolling_windows: tuple = (7, 30),
    add_harmonics: bool = True,
    harmonics: tuple = (1, 2, 3),
    drop_na: bool = True
) -> Pipeline:
    """Build the complete preprocessing pipeline."""
    
    # Default lag configuration
    if lag_map is None:
        lag_map = {
            "RRP": [1, 2, 7, 30],
            "demand": [1, 2, 7, 30],
            "min_temperature": [1, 7],
            "max_temperature": [1, 7],
            "solar_exposure": [1, 7],
            "rainfall": [1, 7],
        }
    
    return Pipeline(steps=[
        ("sanitize", Sanitize(date_col=date_col, keep_leaky=keep_leaky)),
        ("calendar", CalendarFeatures(add_harmonics=add_harmonics, harmonics=harmonics)),
        ("rolling", RollingStats(cols=("RRP", "demand"), windows=rolling_windows)),
        ("interact", Interactions()),
        ("lags", LagFeatures(lag_map=lag_map)),
        ("clean", CleanFinite(drop_na=drop_na)),
    ])


def preprocess_data(
    input_csv: str,
    output_csv: str,
    date_col: str = "date",
    keep_leaky: bool = False,
    **pipeline_kwargs
) -> pd.DataFrame:
    """
    Load raw data, apply preprocessing pipeline, and save enriched data.
    
    Args:
        input_csv: Path to raw CSV file
        output_csv: Path to save preprocessed CSV
        date_col: Name of date column
        keep_leaky: Whether to keep potentially leaky columns
        **pipeline_kwargs: Additional arguments for pipeline configuration
        
    Returns:
        Preprocessed DataFrame
    """
    import os
    
    # Load and preprocess data
    raw_df = pd.read_csv(input_csv)
    print(f"Loaded raw data: {raw_df.shape}")
    
    # Build and apply preprocessing pipeline
    pipeline = build_preprocessing_pipeline(
        date_col=date_col,
        keep_leaky=keep_leaky,
        **pipeline_kwargs
    )
    
    processed_df = pipeline.fit_transform(raw_df)
    print(f"Preprocessed data: {processed_df.shape}")
    
    # Save preprocessed data
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    # Keep datetime index as column for traceability
    output_df = processed_df.copy()
    output_df.insert(0, date_col, processed_df.index)
    output_df.reset_index(drop=True, inplace=True)
    
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Saved preprocessed data to: {output_csv}")
    
    return processed_df