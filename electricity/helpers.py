# helpers.py
"""
Utility functions for visualization, evaluation, and data analysis.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history["loss"], label="Training Loss", alpha=0.8)
    axes[0].plot(history["val_loss"], label="Validation Loss", alpha=0.8)
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    if "mae" in history:
        axes[1].plot(history["mae"], label="Training MAE", alpha=0.8)
        axes[1].plot(history["val_mae"], label="Validation MAE", alpha=0.8)
        axes[1].set_title("Model MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Absolute Error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history plot to: {save_path}")
    
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Predictions vs True Values",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8)
):
    """
    Plot predicted vs true values over time.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional datetime index for x-axis
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    x_axis = dates if dates is not None else range(len(y_true))
    
    # Time series plot
    axes[0].plot(x_axis, y_true, label="True", alpha=0.8, linewidth=1)
    axes[0].plot(x_axis, y_pred, label="Predicted", alpha=0.8, linewidth=1)
    axes[0].set_title(title)
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.6, s=1)
    
    # Add diagonal line for perfect predictions
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title("Predicted vs True (Scatter)")
    axes[1].grid(True, alpha=0.3)
    
    # Add R² score to scatter plot
    r2 = r2_score(y_true, y_pred)
    axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved predictions plot to: {save_path}")
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional datetime index
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    x_axis = dates if dates is not None else range(len(y_true))
    
    # Residuals over time
    axes[0, 0].plot(x_axis, residuals, alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs predicted values
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_title("Residuals vs Predicted")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_title("Distribution of Residuals")
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot (approximate)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (Normal)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved residuals plot to: {save_path}")
    
    plt.show()


def calculate_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: Optional[int] = None,
    return_dict: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics including AIC.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_params: Number of model parameters (for AIC calculation)
        return_dict: Whether to return as dictionary
        
    Returns:
        Dictionary of metrics
    """
    # Handle multi-output case - use first column as main target
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_main = y_true[:, 0]  # Main target (RRP)
        y_pred_main = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
    else:
        y_true_main = y_true.ravel()
        y_pred_main = y_pred.ravel()
    
    residuals = y_true_main - y_pred_main
    n_samples = len(y_true_main)
    mse = mean_squared_error(y_true_main, y_pred_main)
    
    metrics = {
        # Basic metrics
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true_main, y_pred_main),
        "R2": r2_score(y_true_main, y_pred_main),
        "MSE": mse,
        
        # Percentage metrics
        "MAPE": np.mean(np.abs((y_true_main - y_pred_main) / np.where(y_true_main == 0, 1e-8, y_true_main))) * 100,
        "sMAPE": 200 * np.mean(np.abs(y_true_main - y_pred_main) / (np.abs(y_true_main) + np.abs(y_pred_main) + 1e-8)),
        
        # Statistical metrics
        "Mean_Residual": np.mean(residuals),
        "Std_Residual": np.std(residuals),
        "Max_Residual": np.max(np.abs(residuals)),
        
        # Distribution metrics
        "Residual_Skewness": stats.skew(residuals),
        "Residual_Kurtosis": stats.kurtosis(residuals),
    }
    
    # Calculate AIC if number of parameters provided
    if n_params is not None:
        # AIC = n*log(MSE) + 2*k where n is sample size, k is number of parameters
        aic = n_samples * np.log(mse + 1e-8) + 2 * n_params
        metrics["AIC"] = aic
        metrics["n_params"] = n_params
        metrics["n_samples"] = n_samples
    
    if return_dict:
        return metrics
    
    # Print formatted metrics
    print("=== Detailed Metrics ===")
    for metric, value in metrics.items():
        if metric == "AIC":
            print(f"{metric:>18}: {value:.1f}")
        else:
            print(f"{metric:>18}: {value:.4f}")
    
    return metrics


def feature_importance_analysis(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    method: str = "permutation",
    n_repeats: int = 5,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze feature importance using permutation importance.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        feature_names: List of feature names
        method: Importance method ("permutation")
        n_repeats: Number of permutation repeats
        save_path: Optional path to save results
        
    Returns:
        DataFrame with feature importance scores
    """
    if method == "permutation":
        from sklearn.inspection import permutation_importance
        
        # Get baseline predictions
        baseline_score = r2_score(
            model.predict(X_test).ravel(),
            model.predict(X_test).ravel()
        )  # This should be 1.0
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test, model.predict(X_test).ravel(),
            n_repeats=n_repeats,
            random_state=42,
            scoring='r2'
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        if save_path:
            importance_df.to_csv(save_path, index=False)
            print(f"Saved feature importance to: {save_path}")
        
        return importance_df
    
    else:
        raise ValueError(f"Unknown importance method: {method}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot feature importance scores.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to plot
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    
    # Horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['importance_mean'], 
                   xerr=top_features['importance_std'], alpha=0.8)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.grid(True, alpha=0.3)
    
    # Invert y-axis to show most important at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature importance plot to: {save_path}")
    
    plt.show()


def create_evaluation_report(
    results: Dict[str, Dict[str, float]],
    output_path: str
):
    """
    Create a comprehensive evaluation report.
    
    Args:
        results: Dictionary with evaluation results for different splits
        output_path: Path to save the report
    """
    report = {
        "evaluation_summary": results,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    
    # Add summary statistics
    if "test" in results:
        report["test_performance"] = results["test"]
    
    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Saved evaluation report to: {output_path}")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex],
    output_path: str,
    split_name: str = "test"
):
    """
    Save predictions to CSV file.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        dates: Optional datetime index
        output_path: Path to save CSV
        split_name: Name of the data split
    """
    df = pd.DataFrame({
        f'{split_name}_true': y_true,
        f'{split_name}_pred': y_pred,
        f'{split_name}_residual': y_true - y_pred
    })
    
    if dates is not None:
        df.insert(0, 'date', dates)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {split_name} predictions to: {output_path}")


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> Dict[str, Any]:
    """
    Analyze prediction errors by different criteria.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional datetime index
        quantiles: Quantiles for error analysis
        
    Returns:
        Dictionary with error analysis results
    """
    errors = np.abs(y_true - y_pred)
    residuals = y_true - y_pred
    
    analysis = {
        "error_quantiles": {f"q{int(q*100)}": np.quantile(errors, q) for q in quantiles},
        "largest_errors": {
            "indices": np.argsort(errors)[-10:].tolist(),
            "values": errors[np.argsort(errors)[-10:]].tolist()
        },
        "error_by_value_range": {},
        "temporal_patterns": {}
    }
    
    # Error by value ranges
    value_ranges = [
        (y_true.min(), np.quantile(y_true, 0.33)),
        (np.quantile(y_true, 0.33), np.quantile(y_true, 0.67)),
        (np.quantile(y_true, 0.67), y_true.max())
    ]
    
    for i, (low, high) in enumerate(value_ranges):
        mask = (y_true >= low) & (y_true <= high)
        analysis["error_by_value_range"][f"range_{i+1}"] = {
            "range": [low, high],
            "mean_error": np.mean(errors[mask]),
            "median_error": np.median(errors[mask]),
            "count": np.sum(mask)
        }
    
    # Temporal patterns (if dates available)
    if dates is not None:
        df = pd.DataFrame({
            'date': dates,
            'error': errors,
            'residual': residuals
        })
        
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour if hasattr(df['date'].dt, 'hour') else 0
        
        analysis["temporal_patterns"] = {
            "monthly_errors": df.groupby('month')['error'].mean().to_dict(),
            "weekday_errors": df.groupby('weekday')['error'].mean().to_dict(),
        }
    
    return analysis