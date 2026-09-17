# Energy Price Forecasting System

A complete machine learning pipeline for energy price forecasting with multiple neural network architectures and flexible target prediction capabilities.

## Overview

This system provides an end-to-end solution for predicting energy prices using advanced feature engineering, multiple model architectures, and flexible target configurations. It's designed to handle both single-target and multi-target predictions with various neural network approaches.

## System Components

### 1. Feature Engineering Pipeline (preprocessing.py)

**Purpose**: Transforms raw energy data into enriched features suitable for machine learning

**Key Features:**
- **Data Sanitization**: Handles datetime conversion, sorts chronologically, manages binary flags
- **Calendar Features**: Extracts temporal patterns including cyclical encodings and Fourier harmonics
- **Rolling Statistics**: Creates moving averages, standard deviations, min/max values over multiple windows
- **Feature Interactions**: Generates meaningful combinations like demand × temperature, calendar × weather
- **Lag Features**: Creates historical lookback features with configurable lag periods
- **Data Cleaning**: Handles infinite values and missing data with multiple strategies

**Preprocessing Steps:**
1. Sanitize data and establish datetime index
2. Add comprehensive calendar features with cyclical encoding
3. Calculate rolling statistics for key variables
4. Create interaction terms between important features
5. Generate lagged features for temporal dependencies
6. Clean and finalize dataset

**Configuration Options:**
- Customizable lag periods for different variables
- Flexible rolling window sizes
- Optional Fourier harmonics for seasonal patterns
- Choice to keep or exclude potentially leaky columns
- Multiple missing data handling strategies

### 2. Data Pipeline (data.py)

**Purpose**: Creates supervised datasets with flexible target configurations and handles data splitting

**Target Modes:**
- **Single Target**: Predict just RRP (Regional Reference Price)
- **All RRP**: Predict all RRP-related columns simultaneously
- **Custom**: Specify exactly which columns to predict

**Key Capabilities:**
- **Multi-Target Support**: Handle single or multiple prediction targets
- **Sequence Creation**: Generate 3D sequences for transformer models
- **Dual Preparation**: Creates both 2D tabular and 3D sequential datasets
- **Chronological Splitting**: Proper time series train/validation/test splits
- **Feature Scaling**: StandardScaler fitted only on training data
- **Data Persistence**: Save and load processed datasets

**Data Flow:**
1. Load preprocessed data from CSV
2. Create supervised dataset with future targets
3. Generate sequences for transformer models (optional)
4. Split data chronologically into train/val/test
5. Scale features using training statistics
6. Save processed data for model training

**Configuration:**
- Flexible target selection modes
- Configurable prediction horizons
- Adjustable train/validation/test ratios
- Sequence length control for transformers
- Feature exclusion options

### 3. Neural Network Models (model.py)

**Purpose**: Multiple neural network architectures for comprehensive model comparison

**Available Architectures:**

**Standard Network**
- Traditional feed-forward architecture
- Configurable hidden layers with batch normalization
- Dropout regularization
- Best for: Baseline comparisons, interpretable results

**Residual Network**
- Skip connections between layers
- Handles vanishing gradient problems
- Deeper network capabilities
- Best for: Complex pattern recognition, deep learning benefits

**MLP Complex**
- Expanded multi-layer perceptron
- Three sub-layers per block (expand → compress → expand)
- Heavy regularization
- Best for: Rich feature interactions, robust predictions

**Transformer**
- Conv1D feature mixing
- Multi-head attention mechanisms
- Handles sequential dependencies
- Best for: Long-range temporal patterns, sequence modeling

**Model Features:**
- **Multi-Target Prediction**: All architectures support multiple outputs
- **Automatic Comparison**: Side-by-side performance evaluation
- **Smart Data Routing**: 2D data for standard models, 3D for transformers
- **Comprehensive Metrics**: RMSE, MAE, R², AIC, parameter counts
- **Model Persistence**: Save/load trained models with configurations

**Training Features:**
- Early stopping with patience
- Learning rate reduction on plateau
- Model checkpointing for best weights
- Configurable batch sizes and epochs
- Reproducible results with seed control

## Usage Workflow

### Step 1: Preprocess Raw Data
```python
from preprocessing import preprocess_data

# Transform raw data into ML-ready features
processed_df = preprocess_data(
    input_csv="raw_energy_data.csv",
    output_csv="preprocessed_data.csv",
    keep_leaky=False  # Exclude potentially leaky columns
)
```

### Step 2: Prepare Training Data
```python
from data import DataConfig, SupervisedDataset

# Configure target prediction
config = DataConfig(
    target_mode="single",  # or "all_rrp" or "custom"
    target_col="RRP",
    horizon=1,  # Predict 1 step ahead
    sequence_length=24  # For transformer models
)

# Prepare datasets
dataset = SupervisedDataset(config)
data_2d, data_3d = dataset.prepare_for_multiple_models("preprocessed_data.csv")
```

### Step 3: Train Multiple Models
```python
from model import ModelConfig, train_multiple_models

# Configure model training
model_config = ModelConfig(
    hidden_layers=(256, 128, 64),
    learning_rate=1e-3,
    epochs=100,
    dropout_rate=0.2
)

# Train all architectures
results = train_multiple_models(
    train_data=data_2d[0],
    val_data=data_2d[1], 
    test_data=data_2d[2],
    base_config=model_config,
    data_3d=data_3d,  # For transformer
    model_types=["standard", "residual", "mlp_complex", "transformer"]
)
```

### Step 4: Model Comparison
The system automatically provides comprehensive model comparison:
- Performance metrics across all architectures
- Training/validation/test scores
- Model complexity (parameter counts)
- AIC scores for model selection

## Configuration Options

### Preprocessing Configuration
- **Lag Periods**: Customize lookback windows for each variable
- **Rolling Windows**: Set statistical calculation periods
- **Calendar Features**: Enable/disable Fourier harmonics
- **Interactions**: Control feature combination generation
- **Missing Data**: Choose handling strategy

### Data Pipeline Configuration
- **Target Selection**: Single, multiple, or custom targets
- **Prediction Horizon**: How far ahead to predict
- **Data Splits**: Train/validation/test ratios
- **Sequences**: Length for transformer models
- **Scaling**: Feature normalization options

### Model Configuration
- **Architecture**: Choose from 4 different network types
- **Layer Sizes**: Customize hidden layer dimensions
- **Regularization**: Dropout rates and batch normalization
- **Training**: Learning rates, epochs, early stopping
- **Optimization**: Adam optimizer with learning rate scheduling

## Output and Results

### Model Artifacts
- Trained model weights (best_model.keras)
- Model configurations (model_config.json)
- Training histories
- Feature scaling parameters

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **AIC**: Akaike Information Criterion
- **Parameter Count**: Model complexity measure

### Data Artifacts
- Processed training/validation/test splits
- Feature names and scaling parameters
- Target column specifications
- Both 2D and 3D data formats

## System Advantages

### Flexibility
- Multiple target prediction modes
- Four different neural network architectures
- Configurable feature engineering pipeline
- Adaptable to different energy market contexts

### Robustness
- Proper time series validation
- Comprehensive feature engineering
- Multiple regularization techniques
- Automatic model comparison

### Scalability
- Efficient data processing
- Parallel model training capability
- Memory-efficient sequence handling
- Extensible architecture design

### Reliability
- Reproducible results with seed control
- Comprehensive error handling
- Data validation at each step
- Model persistence and recovery

## Best Practices

### Data Preparation
- Ensure chronological ordering of time series data
- Validate date column format and completeness
- Check for data gaps or irregularities
- Consider domain-specific feature requirements

### Model Selection
- Start with standard architecture for baseline
- Use transformer models for strong seasonal patterns
- Apply residual networks for complex relationships
- Try MLP complex for rich feature interactions

### Validation Strategy
- Maintain strict chronological splits
- Use validation set for hyperparameter tuning
- Reserve test set for final model evaluation
- Consider walk-forward validation for production

### Performance Monitoring
- Track both training and validation metrics
- Monitor for overfitting with early stopping
- Compare AIC scores for model selection
- Validate predictions on held-out periods