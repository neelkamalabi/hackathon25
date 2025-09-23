# Brand Power Prediction Pipeline

A machine learning pipeline for predicting brand power using various features including meaning, difference, salience, and premium metrics across different countries, brands, and time periods.

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline to predict brand power values based on:
- **Categorical Features**: Country, Brand
- **Numerical Features**: Meaning, Difference, Salience, Premium
- **Time Features**: Year, Quarter

The pipeline uses Ridge regression with proper data preprocessing, feature engineering, and model evaluation techniques.

## 📁 Project Structure

```
hackathon25/
├── main.py                    # Main pipeline execution script
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Configuration parameters
├── src/                      # Source code modules
│   ├── data/
│   │   ├── loader.py         # Data loading utilities
│   │   └── preprocessor.py   # Data preprocessing pipeline
│   ├── features/
│   │   └── engineering.py    # Feature engineering
│   ├── models/
│   │   ├── predictor.py      # Model training and prediction
│   │   └── evaluator.py      # Model evaluation metrics
│   └── utils/
│       ├── config.py         # Configuration management
│       └── outliers.py       # Outlier detection and handling
├── lte_participants_data/    # Training and test datasets
├── outputs/                  # Generated predictions
├── logs/                     # Application logs
└── tests/                    # Unit tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone/Download the repository**
   ```bash
   cd hackathon25
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**
   ```bash
   python main.py
   ```

## 📊 Input Data Format

The pipeline expects training data with the following columns:
- `year`: Year of the observation
- `quarter`: Quarter number (1-4)
- `country`: Country identifier
- `brand`: Brand identifier
- `meaning`: Meaning metric value
- `difference`: Difference metric value
- `salience`: Salience metric value
- `premium`: Premium metric value
- `power`: Target variable (brand power)

## 📈 Output Format

The pipeline generates predictions with the following columns:
- `Year`: Year of prediction
- `Quarter`: Quarter of prediction
- `Country`: Country identifier
- `Brand`: Brand identifier
- `Predicted Power`: Predicted brand power value

Output files are saved in the `outputs/` directory with timestamp naming convention:
`brand_power_predictions_YYYYMMDD_HHMMSS.csv`

## ⚙️ Configuration

The pipeline behavior can be customized through `config/config.yaml`:

### Data Configuration
- `train_path`: Path to training data
- `test_path`: Path to test data
- `output_path`: Directory for saving predictions

### Model Configuration
- `type`: Model type (currently supports "ridge")
- `ridge.alpha`: Regularization parameter for Ridge regression

### Preprocessing Configuration
- `apply_outlier_capping`: Enable/disable outlier capping
- `apply_scaling`: Enable/disable feature scaling
- `validation_split`: Train/validation split ratio
- `random_state`: Random seed for reproducibility
- `power_min_value`: Minimum allowed power value

### Feature Configuration
- `categorical`: List of categorical feature columns
- `numerical`: List of numerical feature columns
- `time_features`: List of time-based feature columns
- `target`: Target column name

## 🔧 Pipeline Components

### 1. Data Loading (`src/data/loader.py`)
- Loads training and test datasets
- Handles missing values and data type conversions

### 2. Data Preprocessing (`src/data/preprocessor.py`)
- Categorical encoding using target encoding
- Outlier detection and capping
- Feature scaling (optional)
- Train/validation splitting

### 3. Feature Engineering (`src/features/engineering.py`)
- Time-based feature creation
- Categorical feature encoding
- Feature interaction terms (if configured)

### 4. Model Training (`src/models/predictor.py`)
- Ridge regression model training
- Cross-validation for hyperparameter tuning
- Model persistence and loading

### 5. Model Evaluation (`src/models/evaluator.py`)
- RMSE calculation
- Trend hit rate analysis
- Performance scoring system (PPS)

## 📋 Key Features

- **Robust Preprocessing**: Handles outliers, missing values, and categorical encoding
- **Configurable Pipeline**: Easy customization through YAML configuration
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Modular Design**: Clean separation of concerns with reusable components
- **Validation Framework**: Built-in model validation and evaluation metrics


## 📝 Logging

The pipeline generates detailed logs in the `logs/` directory:
- Application logs: `logs/app.log`
- Timestamped execution logs for debugging

## 🔍 Model Performance

The pipeline includes comprehensive evaluation metrics:
- **RMSE Skill**: Root Mean Square Error skill score
- **Trend Hit Rate**: Accuracy of trend direction prediction
- **Performance Score**: Weighted combination of evaluation metrics

## 🛠️ Customization

### Adding New Models
1. Implement model class in `src/models/`
2. Update configuration in `config.yaml`
3. Modify predictor to handle new model type

### Adding New Features
1. Update feature engineering in `src/features/engineering.py`
2. Configure new features in `config.yaml`
3. Update preprocessing pipeline if needed

## 🏆 Results

The pipeline outputs timestamped prediction files ready for submission with properly formatted column names matching competition requirements.

---

*Last updated: September 23, 2025*
