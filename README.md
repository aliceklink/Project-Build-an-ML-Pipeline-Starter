# Build an ML Pipeline for Short-Term Rental Prices in NYC

A complete machine learning pipeline for predicting short-term rental prices in New York City using MLflow, Weights & Biases, and scikit-learn.

## Project Overview

This project implements an end-to-end ML pipeline that:
- Downloads and preprocesses Airbnb rental data
- Performs data validation and quality checks
- Trains a Random Forest regression model
- Evaluates model performance on test data
- Tracks experiments and artifacts using W&B

## Submission Links

**Required for Project Submission:**

- **Weights & Biases Project**: https://wandb.ai/aklink2-western-governors-university/nyc_airbnb
- **GitHub Repository**: https://github.com/aliceklink/Project-Build-an-ML-Pipeline-Starter

## Project Links

- **Weights & Biases Project**: [nyc_airbnb](https://wandb.ai/aklink2-western-governors-university/nyc_airbnb)
- **GitHub Repository**: [Project-Build-an-ML-Pipeline-Starter](https://github.com/aliceklink/Project-Build-an-ML-Pipeline-Starter)

## Pipeline Architecture

The pipeline consists of the following steps:

1. **Data Download** (`download`): Fetches raw data samples
2. **Data Cleaning** (`basic_cleaning`): Removes outliers and filters geographic boundaries
3. **Data Validation** (`data_check`): Performs statistical tests and quality checks
4. **Data Splitting** (`data_split`): Creates train/validation/test splits
5. **Model Training** (`train_random_forest`): Trains Random Forest with optimized hyperparameters
6. **Model Testing** (`test_regression_model`): Evaluates final model performance

## Model Performance

### Best Model Results
- **Model**: Random Forest (solar-firefly-24)
- **Validation MAE**: 32.70
- **Validation R²**: 0.558
- **Hyperparameters**:
  - n_estimators: 200
  - max_depth: 50
  - min_samples_split: 4
  - min_samples_leaf: 3

## Quick Start

### Prerequisites
- Python 3.10
- Conda
- MLflow
- Weights & Biases account

### Running the Complete Pipeline

```bash
# Run the latest release
mlflow run https://github.com/aliceklink/Project-Build-an-ML-Pipeline-Starter.git \
           -v 1.0.0 \
           -P hydra_options="etl.sample='sample1.csv'"

# Run with different data
mlflow run https://github.com/aliceklink/Project-Build-an-ML-Pipeline-Starter.git \
           -v 1.0.0 \
           -P hydra_options="etl.sample='sample2.csv'"
```

### Running Individual Steps

```bash
# Run only specific steps
mlflow run . -P steps=download,basic_cleaning

# Run with custom parameters
mlflow run . -P hydra_options="modeling.random_forest.n_estimators=300"
```

## Configuration

Key configuration parameters in `config.yaml`:

```yaml
etl:
  min_price: 10        # Minimum price filter
  max_price: 350       # Maximum price filter

modeling:
  test_size: 0.2       # Test set proportion
  val_size: 0.2        # Validation set proportion
  random_seed: 42      # Reproducibility seed
  
  random_forest:
    n_estimators: 200  # Optimized value
    max_depth: 50      # Optimized value
```

## Data Processing Features

### Geographic Filtering
The pipeline includes robust geographic boundary filtering to ensure data quality:
- Longitude: -74.25 to -73.50
- Latitude: 40.5 to 41.2

### Feature Engineering
- **Text Features**: TF-IDF vectorization of property names (max 5 features)
- **Categorical Features**: One-hot encoding for neighborhood groups and room types
- **Numerical Features**: Mean imputation for missing values

## Testing and Validation

The pipeline includes comprehensive data validation:
- Row count verification
- Price range validation
- Statistical distribution checks (KL divergence)
- Geographic boundary verification

## Releases

### Version 1.0.0 (Current)
- Complete ML pipeline implementation
- Optimized hyperparameters
- Geographic boundary filtering
- Comprehensive data validation

## Future Improvements and Considerations

### 1. Data and Feature Engineering
- **Enhanced Geographic Features**: 
  - Add distance to subway stations, airports, landmarks
  - Include neighborhood crime statistics and walkability scores
  - Seasonal and temporal features (holidays, events, weather data)

- **Advanced Text Processing**:
  - Sentiment analysis of property descriptions
  - Advanced NLP features (named entity recognition, topic modeling)
  - Image analysis of property photos for amenity detection

- **External Data Integration**:
  - Real estate market trends and pricing indices
  - Tourism and event data affecting demand
  - Economic indicators and demographic data

### 2. Model Improvements
- **Algorithm Diversification**:
  - Gradient boosting models (XGBoost, LightGBM, CatBoost)
  - Neural networks for complex pattern recognition
  - Ensemble methods combining multiple algorithms

- **Advanced Techniques**:
  - Time series modeling for seasonal price patterns
  - Multi-task learning for different property types
  - Bayesian optimization for hyperparameter tuning

### 3. Pipeline Enhancement
- **Real-time Processing**:
  - Stream processing for live price updates
  - Online learning capabilities for model adaptation
  - Real-time monitoring and alerting systems

- **Scalability and Performance**:
  - Distributed computing with Spark or Dask
  - GPU acceleration for model training
  - Model serving optimization and caching

### 4. Monitoring and Maintenance
- **Model Monitoring**:
  - Data drift detection and alerting
  - Model performance degradation tracking
  - Automated retraining triggers

- **A/B Testing Framework**:
  - Controlled model rollouts
  - Performance comparison between model versions
  - Business impact measurement

### 5. Business Intelligence
- **Interpretability**:
  - SHAP values for feature importance
  - LIME for local explanations
  - Model interpretation dashboards

- **Business Metrics**:
  - Revenue impact tracking
  - Price optimization recommendations
  - Market segment analysis

### 6. Data Quality and Governance
- **Enhanced Validation**:
  - Automated outlier detection
  - Data lineage tracking
  - Schema evolution management

- **Privacy and Compliance**:
  - Data anonymization techniques
  - GDPR compliance features
  - Audit trail implementation

### 7. User Experience
- **Interactive Tools**:
  - Web-based prediction interface
  - Mobile application for hosts
  - Real-time pricing recommendations

- **Reporting and Analytics**:
  - Automated business reports
  - Interactive dashboards
  - Market trend analysis

## Technical Debt and Improvements
- Implement proper logging throughout the pipeline
- Add comprehensive unit and integration tests
- Containerize the entire pipeline with Docker
- Implement CI/CD for automated testing and deployment
- Add configuration validation and error handling
- Optimize memory usage for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


## License

[License](LICENSE.txt)
