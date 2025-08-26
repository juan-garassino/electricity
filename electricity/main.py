# main.py
"""
Main script for end-to-end energy price forecasting pipeline.
Orchestrates preprocessing, data preparation, model training, and evaluation.
"""
from __future__ import annotations
import os
import argparse
from typing import Dict, Any

# Local imports
from preprocessing import preprocess_data, build_preprocessing_pipeline
from data import SupervisedDataset, DataConfig, create_supervised_dataset
from model import EnergyPricePredictor, ModelConfig, train_model_from_data
from helpers import (
    plot_training_history, plot_predictions, plot_residuals,
    calculate_detailed_metrics, create_evaluation_report,
    save_predictions, analyze_prediction_errors
)


def run_preprocessing_pipeline(
    input_csv: str,
    output_dir: str = "artifacts_preprocessing",
    keep_leaky: bool = False,
    **preprocessing_kwargs
) -> str:
    """
    Run the preprocessing pipeline.
    
    Args:
        input_csv: Path to raw CSV file
        output_dir: Directory to save preprocessed data
        keep_leaky: Whether to keep potentially leaky columns
        **preprocessing_kwargs: Additional preprocessing parameters
        
    Returns:
        Path to preprocessed CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_csv = os.path.join(output_dir, "preprocessed_data.csv")
    
    print("=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    
    # Default preprocessing parameters
    default_params = {
        "lag_map": {
            "RRP": [1, 2, 7, 30],
            "demand": [1, 2, 7, 30],
            "min_temperature": [1, 7],
            "max_temperature": [1, 7],
            "solar_exposure": [1, 7],
            "rainfall": [1, 7],
        },
        "rolling_windows": (7, 30),
        "add_harmonics": True,
        "harmonics": (1, 2, 3),
        "drop_na": True
    }
    
    # Update with any provided parameters
    default_params.update(preprocessing_kwargs)
    
    # Run preprocessing
    preprocess_data(
        input_csv=input_csv,
        output_csv=preprocessed_csv,
        keep_leaky=keep_leaky,
        **default_params
    )
    
    return preprocessed_csv


def run_data_preparation(
    preprocessed_csv: str,
    data_config: DataConfig,
    output_dir: str = "artifacts_data"
) -> str:
    """
    Run data preparation pipeline to create supervised dataset.
    
    Args:
        preprocessed_csv: Path to preprocessed CSV
        data_config: Data configuration
        output_dir: Directory to save data artifacts
        
    Returns:
        Path to data artifacts directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 2: DATA PREPARATION")
    print("=" * 60)
    
    # Create supervised dataset
    dataset = SupervisedDataset(data_config)
    
    # Prepare data (includes splitting and scaling)
    train_data, val_data, test_data = dataset.prepare_data_from_csv(preprocessed_csv)
    
    # Save processed data
    dataset.save_processed_data(output_dir, train_data, val_data, test_data)
    
    return output_dir


def run_model_training(
    data_dir: str,
    model_config: ModelConfig,
    output_dir: str = "artifacts_model",
    use_leaky: bool = False
) -> tuple:
    """
    Run model training pipeline.
    
    Args:
        data_dir: Directory with processed data
        model_config: Model configuration
        output_dir: Directory to save model artifacts
        use_leaky: Whether using leaky columns for multi-output
        
    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    
    # Load processed data
    dataset = SupervisedDataset(DataConfig())  # Config not needed for loading
    train_data, val_data, test_data = dataset.load_processed_data(data_dir)
    
    # Train model
    model, results = train_model_from_data(
        train_data, val_data, test_data, model_config, output_dir, use_leaky
    )
    
    return model, results


def run_evaluation_and_visualization(
    model: EnergyPricePredictor,
    data_dir: str,
    results: Dict[str, Any],
    output_dir: str = "artifacts_evaluation"
):
    """
    Run comprehensive evaluation and create visualizations.
    
    Args:
        model: Trained model
        data_dir: Directory with processed data
        results: Training results
        output_dir: Directory to save evaluation artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 4: EVALUATION & VISUALIZATION")
    print("=" * 60)
    
    # Load data for evaluation
    dataset = SupervisedDataset(DataConfig())
    train_data, val_data, test_data = dataset.load_processed_data(data_dir)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Plot training history if available
    if hasattr(model, 'history') and model.history:
        plot_training_history(
            model.history.history,
            save_path=os.path.join(output_dir, "training_history.png")
        )
    
    # Plot predictions for test set
    plot_predictions(
        y_test, y_pred_test,
        title="Test Set: Predictions vs True Values",
        save_path=os.path.join(output_dir, "test_predictions.png")
    )
    
    # Plot residuals for test set
    plot_residuals(
        y_test, y_pred_test,
        save_path=os.path.join(output_dir, "test_residuals.png")
    )
    
    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(y_test, y_pred_test, return_dict=True)
    
    # Save predictions
    save_predictions(
        y_test, y_pred_test, None,
        os.path.join(output_dir, "test_predictions.csv"),
        "test"
    )
    
    # Analyze prediction errors
    error_analysis = analyze_prediction_errors(y_test, y_pred_test)
    
    # Create comprehensive evaluation report
    evaluation_data = {
        **results,
        "detailed_metrics": detailed_metrics,
        "error_analysis": error_analysis
    }
    
    create_evaluation_report(
        evaluation_data,
        os.path.join(output_dir, "evaluation_report.json")
    )
    
    print("✅ Evaluation completed successfully!")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Energy Price Forecasting Pipeline")
    
    # Input/Output paths
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to raw CSV file")
    parser.add_argument("--output_dir", type=str, default="artifacts",
                        help="Base directory for all outputs")
    
    # Pipeline control
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip preprocessing step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training step")
    parser.add_argument("--keep_leaky", action="store_true",
                        help="Keep potentially leaky columns")
    
    # Model parameters
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction horizon")
    parser.add_argument("--hidden_layers", nargs="+", type=int, default=[256, 128, 64],
                        help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    
    # Data splitting
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio")
    
    args = parser.parse_args()
    
    # Create output directories
    preprocessing_dir = os.path.join(args.output_dir, "preprocessing")
    data_dir = os.path.join(args.output_dir, "data")
    model_dir = os.path.join(args.output_dir, "model")
    evaluation_dir = os.path.join(args.output_dir, "evaluation")
    
    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        preprocessed_csv = run_preprocessing_pipeline(
            input_csv=args.input_csv,
            output_dir=preprocessing_dir,
            keep_leaky=args.keep_leaky
        )
    else:
        preprocessed_csv = os.path.join(preprocessing_dir, "preprocessed_data.csv")
        if not os.path.exists(preprocessed_csv):
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_csv}")
    
    # Step 2: Data preparation
    data_config = DataConfig(
        horizon=args.horizon,
        use_leaky=args.keep_leaky,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    data_artifacts_dir = run_data_preparation(
        preprocessed_csv=preprocessed_csv,
        data_config=data_config,
        output_dir=data_dir
    )
    
    # Step 3: Model training
    if not args.skip_training:
        model_config = ModelConfig(
            hidden_layers=tuple(args.hidden_layers),
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        model, results = run_model_training(
            data_dir=data_artifacts_dir,
            model_config=model_config,
            output_dir=model_dir,
            use_leaky=args.keep_leaky
        )
        
        # Step 4: Evaluation and visualization
        run_evaluation_and_visualization(
            model=model,
            data_dir=data_artifacts_dir,
            results=results,
            output_dir=evaluation_dir
        )
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 All artifacts saved to: {args.output_dir}")
    print(f"   - Preprocessing: {preprocessing_dir}")
    print(f"   - Data: {data_dir}")
    print(f"   - Model: {model_dir}")
    print(f"   - Evaluation: {evaluation_dir}")


def run_quick_experiment(
    input_csv: str = "raw_data/complete_dataset.csv",
    output_dir: str = "quick_experiment",
    **kwargs
):
    """
    Run a quick experiment with default parameters.
    
    Args:
        input_csv: Path to input CSV
        output_dir: Output directory
        **kwargs: Additional parameters to override defaults
    """
    print("🚀 Running Quick Experiment...")
    
    # Default configurations
    data_config = DataConfig(
        horizon=kwargs.get("horizon", 1),
        use_leaky=kwargs.get("use_leaky", False),
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    model_config = ModelConfig(
        hidden_layers=kwargs.get("hidden_layers", (128, 64, 32)),
        dropout_rate=kwargs.get("dropout", 0.2),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epochs=kwargs.get("epochs", 50),  # Shorter for quick experiment
        batch_size=kwargs.get("batch_size", 64)
    )
    
    try:
        # Step 1: Preprocessing
        preprocessed_csv = run_preprocessing_pipeline(
            input_csv, 
            os.path.join(output_dir, "preprocessing"),
            keep_leaky=data_config.use_leaky
        )
        
        # Step 2: Data preparation
        data_dir = run_data_preparation(
            preprocessed_csv, 
            data_config, 
            os.path.join(output_dir, "data")
        )
        
        # Step 3: Training
        model, results = run_model_training(
            data_dir, 
            model_config, 
            os.path.join(output_dir, "model"),
            use_leaky=data_config.use_leaky
        )
        
        # Step 4: Evaluation
        run_evaluation_and_visualization(
            model, 
            data_dir, 
            results, 
            os.path.join(output_dir, "evaluation")
        )
        
        print(f"\n✅ Quick experiment completed! Results in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    # You can either run the full CLI or a quick experiment
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - run quick experiment
        print("No arguments provided. Running quick experiment...")
        run_quick_experiment()
    else:
        # Arguments provided - run full CLI
        main()