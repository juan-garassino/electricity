# data.py
"""
Dataset creation and management for neural network training.
Handles supervised dataset creation with future targets and data splitting.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class DataConfig:
    """Configuration for dataset creation and splitting."""
    date_col: str = "date"
    target_col: str = "RRP"
    horizon: int = 1  # predict t + horizon
    use_leaky: bool = False  # whether to use potentially leaky columns
    
    # Data splitting ratios (chronological)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Optional: columns to exclude from features
    exclude_cols: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.exclude_cols is None:
            self.exclude_cols = []


class SupervisedDataset:
    """Manages supervised dataset creation and splitting for time series."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_supervised_data(
        self, 
        preprocessed_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create supervised dataset with future targets.
        
        Args:
            preprocessed_df: DataFrame from preprocessing pipeline
            
        Returns:
            Tuple of (features_df, targets_array)
        """
        df = preprocessed_df.copy()
        
        # Ensure we have the target column
        if self.config.target_col not in df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' not found in data")
        
        # Create future target
        target_name = f"{self.config.target_col}_t+{self.config.horizon}"
        y_future = df[self.config.target_col].shift(-self.config.horizon).rename(target_name)
        
        # Handle leaky columns
        leaky_cols = [
            "demand_pos_RRP", "RRP_positive",
            "demand_neg_RRP", "RRP_negative", 
            "frac_at_neg_RRP",
        ]
        
        # Prepare features
        exclude_cols = [self.config.target_col, self.config.date_col] + self.config.exclude_cols
        
        # If using leaky columns, create multi-output targets
        if self.config.use_leaky:
            # Keep leaky columns for multi-output prediction
            available_leaky = [col for col in leaky_cols if col in df.columns]
            if available_leaky:
                print(f"Using leaky columns for multi-output: {available_leaky}")
                
                # Create targets array: [main_target, leaky_col1, leaky_col2, ...]
                targets_list = [y_future]
                for col in available_leaky:
                    targets_list.append(df[col].shift(-self.config.horizon))
                
                # Combine targets
                y_multi = pd.concat(targets_list, axis=1)
                y_multi.columns = [target_name] + [f"{col}_t+{self.config.horizon}" for col in available_leaky]
            else:
                print("Warning: No leaky columns found for multi-output")
                y_multi = y_future.to_frame()
        else:
            # Exclude leaky columns from features
            exclude_cols.extend([col for col in leaky_cols if col in df.columns])
            y_multi = y_future.to_frame()
        
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors="ignore")
        
        # Combine and drop rows with missing targets
        supervised = pd.concat([X, y_multi], axis=1).dropna()
        
        y = supervised[y_multi.columns].values  # Return as numpy array
        X = supervised.drop(columns=y_multi.columns)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"Created supervised dataset:")
        print(f"  Features: {X.shape}")
        print(f"  Targets: {y.shape} ({'multi-output' if y.shape[1] > 1 else 'single output'})")
        print(f"  Target columns: {list(y_multi.columns)}")
        if not self.config.use_leaky:
            print(f"  Excluded leaky columns: {[col for col in leaky_cols if col in preprocessed_df.columns]}")
        
        return X, y
    
    def chronological_split(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split data chronologically into train/val/test sets.
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        n = len(X)
        test_start = int(n * (1 - self.config.test_ratio))
        val_start = int(n * (1 - self.config.test_ratio - self.config.val_ratio))
        
        # Split indices
        train_idx = slice(0, val_start)
        val_idx = slice(val_start, test_start)
        test_idx = slice(test_start, n)
        
        # Split data
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        
        print(f"Data splits:")
        print(f"  Train: {X_train.shape[0]} samples ({train_idx.start} to {train_idx.stop-1})")
        print(f"  Val:   {X_val.shape[0]} samples ({val_idx.start} to {val_idx.stop-1})")
        print(f"  Test:  {X_test.shape[0]} samples ({test_idx.start} to {test_idx.stop-1})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale_features(
        self, 
        train_data: Tuple[pd.DataFrame, np.ndarray],
        val_data: Tuple[pd.DataFrame, np.ndarray],
        test_data: Tuple[pd.DataFrame, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Scale features using StandardScaler fitted on training data.
        
        Returns:
            Tuple of scaled (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        
        # Convert targets to numpy arrays with proper dtype
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32) 
        y_test = y_test.astype(np.float32)
        
        print(f"Feature scaling completed:")
        print(f"  Scaler fitted on {X_train_scaled.shape[0]} training samples")
        print(f"  Target shape: {y_train.shape} ({'multi-output' if len(y_train.shape) > 1 and y_train.shape[1] > 1 else 'single output'})")
        print(f"  Mean scaling: {self.scaler.mean_[:5]}")  # Show first 5 means
        print(f"  Scale factors: {self.scaler.scale_[:5]}")  # Show first 5 scales
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def prepare_data_from_csv(
        self, 
        preprocessed_csv: str
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Complete data preparation pipeline from preprocessed CSV.
        
        Returns:
            Tuple of scaled (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Load preprocessed data
        df = pd.read_csv(preprocessed_csv)
        if self.config.date_col in df.columns:
            df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
            df = df.sort_values(self.config.date_col).set_index(self.config.date_col)
        
        # Create supervised dataset
        X, y = self.create_supervised_data(df)
        
        # Chronological split
        train_data, val_data, test_data = self.chronological_split(X, y)
        
        # Scale features
        return self.scale_features(train_data, val_data, test_data)
    
    def save_processed_data(
        self, 
        output_dir: str,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray]
    ):
        """Save processed and split data to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Save data splits
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # Save scaler parameters
        np.save(os.path.join(output_dir, "scaler_mean.npy"), self.scaler.mean_)
        np.save(os.path.join(output_dir, "scaler_scale.npy"), self.scaler.scale_)
        
        # Save feature names
        with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
            f.write("\n".join(self.feature_names))
        
        print(f"✅ Saved processed data to: {output_dir}")
    
    def load_processed_data(
        self, 
        data_dir: str
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load processed data from disk."""
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        
        # Load scaler parameters
        scaler_mean = np.load(os.path.join(data_dir, "scaler_mean.npy"))
        scaler_scale = np.load(os.path.join(data_dir, "scaler_scale.npy"))
        self.scaler.mean_ = scaler_mean
        self.scaler.scale_ = scaler_scale
        
        # Load feature names
        with open(os.path.join(data_dir, "feature_names.txt"), "r") as f:
            self.feature_names = [line.strip() for line in f]
        
        print(f"✅ Loaded processed data from: {data_dir}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_supervised_dataset(
    preprocessed_csv: str,
    output_csv: str,
    config: DataConfig
) -> pd.DataFrame:
    """
    Create supervised dataset and save to CSV.
    
    Args:
        preprocessed_csv: Path to preprocessed CSV file
        output_csv: Path to save supervised CSV
        config: Data configuration
        
    Returns:
        Supervised DataFrame
    """
    dataset = SupervisedDataset(config)
    
    # Load preprocessed data
    df = pd.read_csv(preprocessed_csv, parse_dates=[config.date_col])
    df = df.sort_values(config.date_col).set_index(config.date_col)
    
    # Create supervised data
    X, y = dataset.create_supervised_data(df)
    
    # Combine for saving
    supervised_df = X.copy()
    supervised_df.insert(0, config.date_col, X.index)
    supervised_df[f"{config.target_col}_t+{config.horizon}"] = y.values
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    supervised_df.reset_index(drop=True, inplace=True)
    supervised_df.to_csv(output_csv, index=False)
    
    print(f"✅ Saved supervised dataset to: {output_csv}")
    print(f"   Shape: {supervised_df.shape}")
    print(f"   Label column: {config.target_col}_t+{config.horizon}")
    
    return supervised_df