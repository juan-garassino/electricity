# model.py
"""
Neural network model definition and training utilities for energy price forecasting.
"""
from __future__ import annotations
import os
import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelConfig:
    """Configuration for neural network model."""
    # Architecture
    hidden_layers: Tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"
    
    # Training
    learning_rate: float = 1e-3
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


class EnergyPricePredictor:
    """Neural network model for energy price prediction."""
    
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
    
    def build_model(self, input_dim: int, output_dim: int = 1) -> Model:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output neurons (1 for main target, 6 for multi-output with leaky cols)
            
        Returns:
            Compiled Keras model
        """
        self.feature_dim = input_dim
        
        model = Sequential([
            Input(shape=(input_dim,), name="input_features")
        ])
        
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
        
        # Output layer - multiple neurons if using leaky columns
        model.add(Dense(output_dim, activation="linear", name="output"))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        self.model = model
        print(f"Built model with {input_dim} input features and {output_dim} output neurons:")
        model.summary()
        
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
        use_leaky: bool = False
    ) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            model_save_path: Path to save the best model
            use_leaky: Whether using leaky columns (affects output dimension)
            
        Returns:
            Training history dictionary
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Determine output dimension
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1], output_dim)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Train model
        print(f"Starting training with {X_train.shape[0]} training samples...")
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


def train_model_from_data(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    config: ModelConfig,
    artifacts_dir: str = "artifacts_nn",
    use_leaky: bool = False
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
    history = predictor.train(train_data, val_data, model_path, use_leaky)
    
    # Save configuration
    predictor.save_config(config_path)
    
    # Evaluate model
    results = predictor.evaluate(test_data, train_data, val_data)
    
    # Print results with AIC
    print("\n=== Model Evaluation ===")
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


# Advanced model architectures
class ResidualBlock(tf.keras.layers.Layer):
    """Residual block for deeper networks."""
    
    def __init__(self, units: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        self.dense1 = Dense(units, activation="relu")
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        
        self.dense2 = Dense(units, activation="relu")
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        
        # Residual connection
        if inputs.shape[-1] == self.units:
            x = x + inputs
        
        x = self.dropout2(x, training=training)
        return x


def build_residual_model(input_dim: int, config: ModelConfig) -> Model:
    """Build a residual neural network."""
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Initial dense layer to match residual block dimensions
    if config.hidden_layers:
        x = Dense(config.hidden_layers[0], activation="relu")(x)
        x = BatchNormalization()(x)
        
        # Add residual blocks
        for units in config.hidden_layers:
            x = ResidualBlock(units, config.dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation="linear")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="ResidualPredictor")
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    
    return model