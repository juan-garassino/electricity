# model.py
"""
Neural network model definition and training utilities for energy price forecasting.
Enhanced with multiple architectures: MLP, ResNet, and Transformer-based models.
Works with single or multiple target variables from data.py pipeline.
"""
from __future__ import annotations
import os
import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List, Literal
from enum import Enum

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Conv1D, Flatten,
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelType(Enum):
    """Available model architectures."""
    STANDARD = "standard"
    RESIDUAL = "residual" 
    MLP_COMPLEX = "mlp_complex"
    TRANSFORMER = "transformer"


@dataclass
class ModelConfig:
    """Configuration for neural network model."""
    # Architecture selection
    model_type: str = "standard"  # standard, residual, mlp_complex, transformer
    
    # Standard/Residual architecture
    hidden_layers: Tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"
    
    # MLP Complex architecture (for mlp_complex type)
    hidden: Tuple[int, ...] = (256, 128, 64)  # Base units for complex MLP
    dropout: float = 0.2
    
    # Transformer architecture (for transformer type) 
    conv_filters: int = 64
    conv_kernel: int = 3
    mha_heads: int = 8
    mha_key_dim: int = 64
    ffn_units: int = 256
    
    # Training
    learning_rate: float = 1e-3
    lr: float = 1e-3  # Alternative naming for consistency
    batch_size: int = 64
    epochs: int = 100
    
    # Regularization & callbacks
    early_stopping_patience: int = 15
    lr_reduction_patience: int = 7
    lr_reduction_factor: float = 0.5
    min_learning_rate: float = 1e-5
    
    # Model saving
    save_best_only: bool = True
    monitor_metric: str = "val_loss"
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Ensure lr and learning_rate are consistent."""
        if self.lr != self.learning_rate:
            self.lr = self.learning_rate


class EnergyPricePredictor:
    """Neural network model for energy price prediction with multiple architectures."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[Model] = None
        self.history = None
        self.feature_dim = None
        
        # Set random seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducible results."""
        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)
    
    def build_model(self, input_dim: int, output_dim: int = 1, input_steps: int = None) -> Model:
        """
        Build the neural network architecture based on model_type.
        
        Args:
            input_dim: Number of input features (or n_features for transformer)
            output_dim: Number of output neurons
            input_steps: Sequence length for transformer models
            
        Returns:
            Compiled Keras model
        """
        self.feature_dim = input_dim
        
        if self.config.model_type == ModelType.STANDARD.value:
            model = self._build_standard_model(input_dim, output_dim)
        elif self.config.model_type == ModelType.RESIDUAL.value:
            model = self._build_residual_model(input_dim, output_dim)
        elif self.config.model_type == ModelType.MLP_COMPLEX.value:
            model = self._build_mlp_complex(input_dim, output_dim)
        elif self.config.model_type == ModelType.TRANSFORMER.value:
            if input_steps is None:
                raise ValueError("input_steps must be provided for transformer models")
            model = self._build_conv_attn_model(input_steps, input_dim, output_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
        
        self.model = model
        print(f"Built {self.config.model_type} model:")
        model.summary()
        
        return model
    
    def _build_standard_model(self, input_dim: int, output_dim: int) -> Model:
        """Build the standard feed-forward neural network."""
        model = Sequential([Input(shape=(input_dim,), name="input_features")])
        
        # Add hidden layers
        for i, units in enumerate(self.config.hidden_layers):
            model.add(Dense(
                units, 
                activation=self.config.activation,
                name=f"dense_{i+1}"
            ))
            
            if self.config.use_batch_norm:
                model.add(BatchNormalization(name=f"batch_norm_{i+1}"))
            
            if self.config.dropout_rate > 0:
                model.add(Dropout(
                    self.config.dropout_rate,
                    name=f"dropout_{i+1}"
                ))
        
        # Output layer
        model.add(Dense(output_dim, activation="linear", name="output"))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def _build_mlp_complex(self, input_dim: int, output_dim: int) -> Model:
        """Build complex MLP with expanded layers per block."""
        m = Sequential([Input(shape=(input_dim,))])
        
        for units in self.config.hidden:
            # First sub-layer
            m.add(Dense(units, activation="relu"))
            m.add(BatchNormalization())
            m.add(Dropout(self.config.dropout))
            
            # Second sub-layer (compressed)
            m.add(Dense(int(units / 2), activation="relu"))
            m.add(BatchNormalization())
            m.add(Dropout(self.config.dropout))
            
            # Third sub-layer (back to original size)
            m.add(Dense(units, activation="relu"))
            m.add(BatchNormalization())
            m.add(Dropout(self.config.dropout))
        
        m.add(Dense(output_dim, activation="linear"))
        m.compile(
            optimizer=Adam(self.config.lr), 
            loss="mse", 
            metrics=["mae"]
        )
        return m
    
    def _build_residual_model(self, input_dim: int, output_dim: int) -> Model:
        """Build a residual neural network."""
        inputs = Input(shape=(input_dim,))
        x = inputs
        
        # Initial dense layer to match residual block dimensions
        if self.config.hidden_layers:
            x = Dense(self.config.hidden_layers[0], activation="relu")(x)
            x = BatchNormalization()(x)
            
            # Add residual blocks
            for units in self.config.hidden_layers:
                x = self._residual_block(x, units, self.config.dropout_rate)
        
        # Output layer
        outputs = Dense(output_dim, activation="linear")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="ResidualPredictor")
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def _residual_block(self, inputs, units: int, dropout_rate: float):
        """Create a residual block."""
        x = Dense(units, activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        
        # Residual connection
        if inputs.shape[-1] == units:
            x = x + inputs
        
        x = Dropout(dropout_rate)(x)
        return x
    
    def _transformer_block(self, x, training: bool = True):
        """
        Simple Transformer encoder block:
        - Pre-norm LayerNormalization
        - MultiHeadAttention (self-attention)
        - Residual
        - Feed-Forward network (Dense -> Dropout -> Dense) with residual
        """
        # Self-attention
        attn_input = LayerNormalization(epsilon=1e-6)(x)
        attn_out = MultiHeadAttention(
            num_heads=self.config.mha_heads, 
            key_dim=self.config.mha_key_dim
        )(attn_input, attn_input)
        attn_out = Dropout(self.config.dropout)(attn_out, training=training)
        x = x + attn_out  # residual

        # Feed-forward
        ffn_input = LayerNormalization(epsilon=1e-6)(x)
        ffn = Dense(self.config.ffn_units, activation="relu")(ffn_input)
        ffn = Dropout(self.config.dropout)(ffn, training=training)
        ffn = Dense(x.shape[-1], activation="linear")(ffn)
        x = x + ffn  # residual
        return x
    
    def _build_conv_attn_model(self, input_steps: int, n_features: int, output_dim: int) -> Model:
        """
        Conv1D → Transformer Encoder (MHA) → Dense(1)
        """
        inp = Input(shape=(input_steps, n_features))

        # Feature mixing along time using Conv1D
        x = Conv1D(
            filters=self.config.conv_filters,
            kernel_size=self.config.conv_kernel,
            padding="same",
            activation="relu"
        )(inp)

        # Self-attention blocks (captures long-range temporal dependencies)
        x = self._transformer_block(x, training=True)
        x = self._transformer_block(x, training=True)  
        x = self._transformer_block(x, training=True)

        # Flatten the output of the attention block
        x = Flatten()(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(100, activation="relu")(x)
        
        out = Dense(output_dim, activation="linear")(x)

        model = Model(inp, out)
        model.compile(
            optimizer=Adam(self.config.lr), 
            loss="mse", 
            metrics=["mae"]
        )
        return model
    
    def create_callbacks(self, model_save_path: str) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor=self.config.monitor_metric,
            factor=self.config.lr_reduction_factor,
            patience=self.config.lr_reduction_patience,
            min_lr=self.config.min_learning_rate,
            verbose=1
        ))
        
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=model_save_path,
            monitor=self.config.monitor_metric,
            save_best_only=self.config.save_best_only,
            verbose=1
        ))
        
        return callbacks
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        model_save_path: str,
        use_leaky: bool = False,
        input_steps: int = None
    ) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            model_save_path: Path to save the best model
            use_leaky: Whether using leaky columns (affects output dimension)
            input_steps: Sequence length for transformer models
            
        Returns:
            Training history dictionary
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Determine output dimension
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # For transformer models, input_dim represents n_features
        if self.config.model_type == ModelType.TRANSFORMER.value:
            if len(X_train.shape) == 3:  # (samples, steps, features)
                input_dim = X_train.shape[2]  # n_features
                if input_steps is None:
                    input_steps = X_train.shape[1]  # sequence length
            else:
                raise ValueError("Transformer models require 3D input (samples, steps, features)")
        else:
            input_dim = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[-1]
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_dim, output_dim, input_steps)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Train model
        print(f"Starting training {self.config.model_type} model with {X_train.shape[0]} training samples...")
        print(f"Output dimension: {output_dim} ({'multi-output with leaky cols' if output_dim > 1 else 'single target'})")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
        train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on different data splits.
        
        Args:
            test_data: Test data tuple (X_test, y_test)
            train_data: Optional training data for evaluation
            val_data: Optional validation data for evaluation
            
        Returns:
            Dictionary with metrics for each split
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        results = {}
        
        def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            # Handle multi-output case - use first column as main target
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true_main = y_true[:, 0]  # Main target (RRP)
                y_pred_main = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
            else:
                y_true_main = y_true.ravel()
                y_pred_main = y_pred.ravel()
            
            # Calculate number of parameters for AIC
            n_params = sum([np.prod(layer.shape) for layer in self.model.get_weights()])
            n_samples = len(y_true_main)
            
            # Calculate metrics
            mse = mean_squared_error(y_true_main, y_pred_main)
            mae = mean_absolute_error(y_true_main, y_pred_main)
            r2 = r2_score(y_true_main, y_pred_main)
            
            # Calculate AIC: AIC = n*log(MSE) + 2*k
            # where n is sample size, k is number of parameters
            aic = n_samples * np.log(mse) + 2 * n_params
            
            return {
                "RMSE": math.sqrt(mse),
                "MAE": mae,
                "R2": r2,
                "AIC": aic,
                "MSE": mse,
                "n_params": n_params
            }
        
        # Evaluate on test data
        X_test, y_test = test_data
        y_pred_test = self.model.predict(X_test, verbose=0)
        results["test"] = compute_metrics(y_test, y_pred_test)
        
        # Optionally evaluate on training data
        if train_data is not None:
            X_train, y_train = train_data
            y_pred_train = self.model.predict(X_train, verbose=0)
            results["train"] = compute_metrics(y_train, y_pred_train)
        
        # Optionally evaluate on validation data
        if val_data is not None:
            X_val, y_val = val_data
            y_pred_val = self.model.predict(X_val, verbose=0)
            results["val"] = compute_metrics(y_val, y_pred_val)
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, verbose=0).ravel()
    
    def save_config(self, config_path: str):
        """Save model configuration to JSON file."""
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"Saved model configuration to: {config_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model from file."""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from: {model_path}")
    
    @classmethod
    def load_config(cls, config_path: str) -> 'EnergyPricePredictor':
        """Load model configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = ModelConfig(**config_dict)
        return cls(config)


def train_multiple_models(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    base_config: ModelConfig,
    artifacts_dir: str = "artifacts_nn",
    model_types: List[str] = None,
    input_steps: int = None,
    data_3d: Optional[Tuple[
        Tuple[np.ndarray, np.ndarray], 
        Tuple[np.ndarray, np.ndarray], 
        Tuple[np.ndarray, np.ndarray]
    ]] = None
) -> Dict[str, Tuple[EnergyPricePredictor, Dict[str, Any]]]:
    """
    Train multiple model architectures and compare their performance.
    
    Args:
        train_data: Training data tuple (2D for standard/residual/mlp models)
        val_data: Validation data tuple (2D)
        test_data: Test data tuple (2D)
        base_config: Base model configuration
        artifacts_dir: Directory to save model artifacts
        model_types: List of model types to train
        input_steps: Sequence length for transformer models
        data_3d: Optional 3D data tuple for transformer models
        
    Returns:
        Dictionary mapping model_type to (trained_model, evaluation_results)
    """
    if model_types is None:
        model_types = ["standard", "residual", "mlp_complex"]
        # Only add transformer if we have 3D data
        if data_3d is not None:
            model_types.append("transformer")
    
    results = {}
    
    print(f"Training {len(model_types)} different model architectures...")
    print(f"2D data shape: {train_data[0].shape}")
    if data_3d is not None:
        print(f"3D data shape: {data_3d[0][0].shape}")
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Create config for this model type
        config = ModelConfig(**asdict(base_config))
        config.model_type = model_type
        
        # Create model-specific artifacts directory
        model_artifacts_dir = os.path.join(artifacts_dir, f"{model_type}_model")
        os.makedirs(model_artifacts_dir, exist_ok=True)
        
        try:
            # Use 3D data for transformer, 2D for others
            if model_type == "transformer" and data_3d is not None:
                predictor, eval_results = train_model_from_data(
                    data_3d[0], data_3d[1], data_3d[2], 
                    config, model_artifacts_dir, input_steps
                )
            else:
                predictor, eval_results = train_model_from_data(
                    train_data, val_data, test_data, 
                    config, model_artifacts_dir, input_steps
                )
            
            results[model_type] = (predictor, eval_results)
        
        except Exception as e:
            print(f"Failed to train {model_type} model: {str(e)}")
            continue
    
    results = {}
    
    print(f"Training {len(model_types)} different model architectures...")
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Create config for this model type
        config = ModelConfig(**asdict(base_config))
        config.model_type = model_type
        
        # Create model-specific artifacts directory
        model_artifacts_dir = os.path.join(artifacts_dir, f"{model_type}_model")
        os.makedirs(model_artifacts_dir, exist_ok=True)
        
        try:
            # Train model
            predictor, eval_results = train_model_from_data(
                train_data, val_data, test_data, 
                config, model_artifacts_dir, use_leaky, input_steps
            )
            
            results[model_type] = (predictor, eval_results)
            
        except Exception as e:
            print(f"Failed to train {model_type} model: {str(e)}")
            continue
    
    # Print comparison
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':15} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'AIC':>10} {'Params':>8}")
    print("-" * 80)
    
    for model_type, (predictor, eval_results) in results.items():
        test_metrics = eval_results["test"]
        print(f"{model_type:15} {test_metrics['RMSE']:8.3f} {test_metrics['MAE']:8.3f} "
              f"{test_metrics['R2']:8.4f} {test_metrics['AIC']:10.1f} {test_metrics['n_params']:8d}")
    
    return results


def train_model_from_data(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    config: ModelConfig,
    artifacts_dir: str = "artifacts_nn",
    use_leaky: bool = False,
    input_steps: int = None
) -> Tuple[EnergyPricePredictor, Dict[str, Any]]:
    """
    Complete model training pipeline.
    
    Args:
        train_data: Training data tuple
        val_data: Validation data tuple  
        test_data: Test data tuple
        config: Model configuration
        artifacts_dir: Directory to save model artifacts
        use_leaky: Whether using leaky columns for multi-output
        input_steps: Sequence length for transformer models
        
    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create model
    predictor = EnergyPricePredictor(config)
    
    # Define paths
    model_path = os.path.join(artifacts_dir, "best_model.keras")
    config_path = os.path.join(artifacts_dir, "model_config.json")
    
    # Train model
    history = predictor.train(train_data, val_data, model_path, use_leaky, input_steps)
    
    # Save configuration
    predictor.save_config(config_path)
    
    # Evaluate model
    results = predictor.evaluate(test_data, train_data, val_data)
    
    # Print results with AIC
    print(f"\n=== {config.model_type.upper()} Model Evaluation ===")
    for split_name, metrics in results.items():
        print(f"{split_name.upper():>5}  RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  R²={metrics['R2']:.4f}  AIC={metrics['AIC']:.1f}")
    
    return predictor, results


def create_ensemble_predictions(
    models: List[EnergyPricePredictor],
    X: np.ndarray,
    method: str = "mean"
) -> np.ndarray:
    """
    Create ensemble predictions from multiple models.
    
    Args:
        models: List of trained models
        X: Input features
        method: Ensemble method ("mean", "median")
        
    Returns:
        Ensemble predictions
    """
    predictions = np.array([model.predict(X) for model in models])
    
    if method == "mean":
        return np.mean(predictions, axis=0)
    elif method == "median":
        return np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


# Example usage function
def example_multi_model_training():
    """Example of how to train multiple model types."""
    
    # Create sample configuration
    config = ModelConfig(
        # Standard architecture settings
        hidden_layers=(256, 128, 64),
        dropout_rate=0.2,
        
        # Complex MLP settings  
        hidden=(256, 128, 64),
        dropout=0.2,
        
        # Transformer settings
        conv_filters=64,
        conv_kernel=3,
        mha_heads=8,
        mha_key_dim=64,
        ffn_units=256,
        
        # Training settings
        learning_rate=1e-3,
        batch_size=64,
        epochs=100,
        
        seed=42
    )
    
    # Example usage (you would replace with your actual data):
    """
    # For 2D data (standard, residual, mlp_complex)
    results_2d = train_multiple_models(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val), 
        test_data=(X_test, y_test),
        base_config=config,
        model_types=["standard", "residual", "mlp_complex"]
    )
    
    # For 3D sequential data (including transformer)
    results_3d = train_multiple_models(
        train_data=(X_train_3d, y_train),  # X_train_3d shape: (samples, timesteps, features)
        val_data=(X_val_3d, y_val),
        test_data=(X_test_3d, y_test), 
        base_config=config,
        model_types=["standard", "residual", "mlp_complex", "transformer"],
        input_steps=X_train_3d.shape[1]  # sequence length
    )
    """
    
    return config