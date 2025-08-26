# data.py
"""
Dataset creation and management for neural network training.
Handles supervised dataset creation with future targets and data splitting.
Enhanced to support multiple RRP-related target variables.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Literal
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class DataConfig:
    """Configuration for dataset creation and splitting."""
    date_col: str = "date"
    target_col: Union[str, List[str]] = "RRP"  # Can be single column or list of columns
    target_mode: Literal["single", "all_rrp", "custom"] = "single"  # Controls which targets to use
    horizon: int = 1  # predict t + horizon
    
    # Data splitting ratios (chronological)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Optional: columns to exclude from features
    exclude_cols: Optional[List[str]] = None
    
    # Sequence parameters for transformer models
    sequence_length: Optional[int] = None  # For creating 3D sequences
    
    def __post_init__(self):
        if self.exclude_cols is None:
            self.exclude_cols = []
        
        # Set target_col based on target_mode
        if self.target_mode == "single":
            if isinstance(self.target_col, list):
                self.target_col = self.target_col[0]  # Use first target
        elif self.target_mode == "all_rrp":
            # Define all RRP-related columns
            self.target_col = [
                "RRP", "demand_pos_RRP", "RRP_positive", 
                "demand_neg_RRP", "RRP_negative"
            ]
        # For "custom" mode, use whatever was provided in target_col
        
        print(f"Target mode: {self.target_mode}")
        print(f"Target columns: {self.target_col}")


class SupervisedDataset:
    """Manages supervised dataset creation and splitting for time series."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_names = []
        
    def create_supervised_data(
        self, 
        preprocessed_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Create supervised dataset with future targets.
        
        Args:
            preprocessed_df: DataFrame from preprocessing pipeline
            
        Returns:
            Tuple of (features_df, targets_array, target_names)
        """
        df = preprocessed_df.copy()
        
        # Handle single vs multiple targets
        if isinstance(self.config.target_col, str):
            target_cols = [self.config.target_col]
        else:
            target_cols = self.config.target_col.copy()
        
        # Filter to only columns that exist in the data
        available_target_cols = [col for col in target_cols if col in df.columns]
        
        if not available_target_cols:
            raise ValueError(f"None of the target columns {target_cols} found in data. Available columns: {list(df.columns)}")
        
        if len(available_target_cols) < len(target_cols):
            missing = set(target_cols) - set(available_target_cols)
            print(f"Warning: Missing target columns: {missing}")
        
        print(f"Using target columns: {available_target_cols}")
        
        # Create future targets
        target_names = []
        y_targets = []
        
        for col in available_target_cols:
            target_name = f"{col}_t+{self.config.horizon}"
            y_future = df[col].shift(-self.config.horizon)
            y_targets.append(y_future)
            target_names.append(target_name)
        
        # Combine targets
        if len(y_targets) == 1:
            y_combined = y_targets[0].to_frame()
        else:
            y_combined = pd.concat(y_targets, axis=1)
        
        y_combined.columns = target_names
        
        # Prepare features (exclude target columns and specified exclusions)
        all_exclude_cols = available_target_cols + [self.config.date_col] + self.config.exclude_cols
        X = df.drop(columns=[col for col in all_exclude_cols if col in df.columns], errors="ignore")
        
        # Combine and drop rows with missing targets
        supervised = pd.concat([X, y_combined], axis=1).dropna()
        
        y = supervised[target_names].values  # Return as numpy array
        X_clean = supervised.drop(columns=target_names)
        
        # Store names for later use
        self.feature_names = X_clean.columns.tolist()
        self.target_names = target_names
        
        print(f"Created supervised dataset:")
        print(f"  Features: {X_clean.shape}")
        print(f"  Targets: {y.shape} ({'multi-target' if y.shape[1] > 1 else 'single target'})")
        print(f"  Target columns: {target_names}")
        print(f"  Feature columns: {len(self.feature_names)} features")
        
        return X_clean, y, target_names
    
    def create_sequences(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequences for transformer models.
        
        Args:
            X: Features dataframe
            y: Target array
            
        Returns:
            Tuple of (X_sequences, y_sequences) with shapes:
            - X_sequences: (n_samples, sequence_length, n_features)
            - y_sequences: (n_samples, n_targets)
        """
        if self.config.sequence_length is None:
            raise ValueError("sequence_length must be specified for sequence creation")
        
        seq_len = self.config.sequence_length
        X_array = X.values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_len, len(X_array)):
            X_sequences.append(X_array[i-seq_len:i])  # Look back seq_len steps
            y_sequences.append(y[i])  # Current target
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Created sequences:")
        print(f"  X_sequences: {X_sequences.shape}")
        print(f"  y_sequences: {y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def chronological_split(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: np.ndarray,
        create_sequences: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split data chronologically into train/val/test sets.
        
        Args:
            X: Features (DataFrame or array)
            y: Targets
            create_sequences: Whether to create 3D sequences for transformers
            
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        if create_sequences and self.config.sequence_length is not None:
            # Create sequences first, then split
            if isinstance(X, pd.DataFrame):
                X_seq, y_seq = self.create_sequences(X, y)
            else:
                # Convert array back to DataFrame for sequence creation
                X_df = pd.DataFrame(X, columns=self.feature_names)
                X_seq, y_seq = self.create_sequences(X_df, y)
            
            X, y = X_seq, y_seq
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        n = len(X)
        test_start = int(n * (1 - self.config.test_ratio))
        val_start = int(n * (1 - self.config.test_ratio - self.config.val_ratio))
        
        # Split indices
        train_idx = slice(0, val_start)
        val_idx = slice(val_start, test_start)
        test_idx = slice(test_start, n)
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        data_type = "sequences" if len(X.shape) == 3 else "tabular"
        print(f"Data splits ({data_type}):")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        if len(X.shape) == 3:
            print(f"  Sequence shape: (samples, {X.shape[1]}, {X.shape[2]})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale_features(
        self, 
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Scale features using StandardScaler fitted on training data.
        Handles both 2D (tabular) and 3D (sequence) data.
        
        Returns:
            Tuple of scaled (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Handle 3D sequence data by reshaping
        is_3d = len(X_train.shape) == 3
        
        if is_3d:
            # Reshape to 2D for scaling: (samples * timesteps, features)
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_2d = X_train.reshape(-1, n_features)
            X_val_2d = X_val.reshape(-1, n_features)
            X_test_2d = X_test.reshape(-1, n_features)
            
            # Fit scaler on training data only
            X_train_scaled_2d = self.scaler.fit_transform(X_train_2d).astype(np.float32)
            X_val_scaled_2d = self.scaler.transform(X_val_2d).astype(np.float32)
            X_test_scaled_2d = self.scaler.transform(X_test_2d).astype(np.float32)
            
            # Reshape back to 3D
            X_train_scaled = X_train_scaled_2d.reshape(X_train.shape).astype(np.float32)
            X_val_scaled = X_val_scaled_2d.reshape(X_val.shape).astype(np.float32)
            X_test_scaled = X_test_scaled_2d.reshape(X_test.shape).astype(np.float32)
        else:
            # Standard 2D scaling
            X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
            X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        
        # Convert targets to numpy arrays with proper dtype
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32) 
        y_test = y_test.astype(np.float32)
        
        print(f"Feature scaling completed:")
        print(f"  Data type: {'3D sequences' if is_3d else '2D tabular'}")
        print(f"  Scaler fitted on {X_train_scaled.shape[0]} training samples")
        print(f"  Target shape: {y_train.shape} ({'multi-target' if len(y_train.shape) > 1 and y_train.shape[1] > 1 else 'single target'})")
        print(f"  Feature scaling stats (first 5):")
        print(f"    Means: {self.scaler.mean_[:5]}")
        print(f"    Scales: {self.scaler.scale_[:5]}")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def prepare_data_from_csv(
        self, 
        preprocessed_csv: str,
        for_transformer: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Complete data preparation pipeline from preprocessed CSV.
        
        Args:
            preprocessed_csv: Path to preprocessed CSV
            for_transformer: Whether to create 3D sequences for transformer models
            
        Returns:
            Tuple of scaled (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Load preprocessed data
        df = pd.read_csv(preprocessed_csv)
        if self.config.date_col in df.columns:
            df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
            df = df.sort_values(self.config.date_col).set_index(self.config.date_col)
        
        # Create supervised dataset
        X, y, target_names = self.create_supervised_data(df)
        
        # Chronological split (with optional sequence creation)
        train_data, val_data, test_data = self.chronological_split(X, y, create_sequences=for_transformer)
        
        # Scale features
        return self.scale_features(train_data, val_data, test_data)
    
    def prepare_for_multiple_models(
        self, 
        preprocessed_csv: str
    ) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],  # 2D data
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]   # 3D data
    ]:
        """
        Prepare both 2D and 3D data for training multiple model types.
        
        Returns:
            Tuple of (2D_data_splits, 3D_data_splits)
        """
        if self.config.sequence_length is None:
            print("Warning: sequence_length not set, using default of 24")
            self.config.sequence_length = 24
        
        # Prepare 2D data for standard/residual/mlp models
        print("Preparing 2D tabular data...")
        data_2d = self.prepare_data_from_csv(preprocessed_csv, for_transformer=False)
        
        # Prepare 3D sequential data for transformer models
        print("\nPreparing 3D sequence data...")
        data_3d = self.prepare_data_from_csv(preprocessed_csv, for_transformer=True)
        
        return data_2d, data_3d
    
    def save_processed_data(
        self, 
        output_dir: str,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        suffix: str = ""
    ):
        """Save processed and split data to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Add suffix for different data types
        suffix = f"_{suffix}" if suffix else ""
        
        # Save data splits
        np.save(os.path.join(output_dir, f"X_train{suffix}.npy"), X_train)
        np.save(os.path.join(output_dir, f"y_train{suffix}.npy"), y_train)
        np.save(os.path.join(output_dir, f"X_val{suffix}.npy"), X_val)
        np.save(os.path.join(output_dir, f"y_val{suffix}.npy"), y_val)
        np.save(os.path.join(output_dir, f"X_test{suffix}.npy"), X_test)
        np.save(os.path.join(output_dir, f"y_test{suffix}.npy"), y_test)
        
        # Save scaler parameters (only once)
        if suffix == "" or suffix == "_2d":
            np.save(os.path.join(output_dir, "scaler_mean.npy"), self.scaler.mean_)
            np.save(os.path.join(output_dir, "scaler_scale.npy"), self.scaler.scale_)
            
            # Save feature and target names
            with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
                f.write("\n".join(self.feature_names))
            
            with open(os.path.join(output_dir, "target_names.txt"), "w") as f:
                f.write("\n".join(self.target_names))
        
        data_type = "3D sequences" if len(X_train.shape) == 3 else "2D tabular"
        print(f"✅ Saved {data_type} processed data to: {output_dir}")
    
    def load_processed_data(
        self, 
        data_dir: str,
        suffix: str = ""
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load processed data from disk."""
        suffix = f"_{suffix}" if suffix else ""
        
        X_train = np.load(os.path.join(data_dir, f"X_train{suffix}.npy"))
        y_train = np.load(os.path.join(data_dir, f"y_train{suffix}.npy"))
        X_val = np.load(os.path.join(data_dir, f"X_val{suffix}.npy"))
        y_val = np.load(os.path.join(data_dir, f"y_val{suffix}.npy"))
        X_test = np.load(os.path.join(data_dir, f"X_test{suffix}.npy"))
        y_test = np.load(os.path.join(data_dir, f"y_test{suffix}.npy"))
        
        # Load scaler parameters and names (only once)
        if suffix == "" or suffix == "_2d":
            scaler_mean = np.load(os.path.join(data_dir, "scaler_mean.npy"))
            scaler_scale = np.load(os.path.join(data_dir, "scaler_scale.npy"))
            self.scaler.mean_ = scaler_mean
            self.scaler.scale_ = scaler_scale
            
            # Load feature and target names
            with open(os.path.join(data_dir, "feature_names.txt"), "r") as f:
                self.feature_names = [line.strip() for line in f]
            
            with open(os.path.join(data_dir, "target_names.txt"), "r") as f:
                self.target_names = [line.strip() for line in f]
        
        data_type = "3D sequences" if len(X_train.shape) == 3 else "2D tabular"
        print(f"✅ Loaded {data_type} processed data from: {data_dir}")
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
    X, y, target_names = dataset.create_supervised_data(df)
    
    # Combine for saving
    supervised_df = X.copy()
    supervised_df.insert(0, config.date_col, X.index)
    
    # Add target columns
    if y.ndim == 1:
        supervised_df[target_names[0]] = y
    else:
        for i, name in enumerate(target_names):
            supervised_df[name] = y[:, i]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    supervised_df.reset_index(drop=True, inplace=True)
    supervised_df.to_csv(output_csv, index=False)
    
    print(f"✅ Saved supervised dataset to: {output_csv}")
    print(f"   Shape: {supervised_df.shape}")
    print(f"   Target columns: {target_names}")
    
    return supervised_df


# Example usage functions
def example_single_target():
    """Example: Single RRP target prediction."""
    config = DataConfig(
        target_mode="single",
        target_col="RRP",
        horizon=1,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    dataset = SupervisedDataset(config)
    # data = dataset.prepare_data_from_csv("preprocessed_data.csv")
    return dataset, config


def example_multi_target():
    """Example: Multi-target RRP prediction."""
    config = DataConfig(
        target_mode="all_rrp",  # This will use all RRP-related columns
        horizon=1,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    dataset = SupervisedDataset(config)
    # data = dataset.prepare_data_from_csv("preprocessed_data.csv")
    return dataset, config


def example_transformer_ready():
    """Example: Prepare data for transformer models."""
    config = DataConfig(
        target_mode="single",
        target_col="RRP",
        horizon=1,
        sequence_length=24,  # Look back 24 time steps
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    dataset = SupervisedDataset(config)
    # data_2d, data_3d = dataset.prepare_for_multiple_models("preprocessed_data.csv")
    return dataset, config


def example_custom_targets():
    """Example: Custom target selection."""
    config = DataConfig(
        target_mode="custom",
        target_col=["RRP", "demand_pos_RRP"],  # Custom selection
        horizon=1,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    dataset = SupervisedDataset(config)
    return dataset, config