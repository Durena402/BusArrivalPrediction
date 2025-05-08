# Rutgers Bus Arrival Time Prediction

## Setup Instructions

### Environment Setup

1. Ensure you have Python installed (Python 3.8+ recommended)
2. Run the setup script to create a Python virtual environment and optionally generate synthetic data:
   ```
   chmod +x ./setup_env.sh
   ./setup_env.sh
   ```
3. This script will:
   - Create and activate a Python virtual environment
   - Install all required dependencies
   - Ask if you want to generate synthetic data immediately
   - Provide instructions for generating data later if needed

### Important Rules

**ALWAYS run project code with the Python environment activated:**
```
source rbat_env/bin/activate
```

## Project Overview

This project aims to improve bus arrival time predictions by integrating multiple data sources:

- Bus GPS tracking data
- Weather conditions
- Traffic patterns
- Passenger boarding/alighting behavior

Using supervised learning models, we predict arrival times more accurately than existing methods.

## Research Focus

This project compares multiple machine learning approaches to bus arrival prediction:

- **Traditional Machine Learning**: Linear Regression and Random Forest models
- **Feature Engineering**: Temporal, weather, traffic, and historical features
- **Comparative Analysis**: Evaluation of model performance, training efficiency, and prediction accuracy

The research aims to determine which modeling approach best captures the complex relationships between weather, traffic, passenger patterns, and bus arrival times.

## Data Generation

The project uses synthetic data to simulate real-world bus operations. If you didn't generate the dataset during setup, you can do so with:

```
python3 syntheticDataset.py
```

This will create two files:

- `bus_gps_tracking_data.csv`: GPS tracking points for buses
- `bus_stop_level_data.csv`: Stop-level arrival/departure data

### Synthetic Data Features

The generated dataset includes:

- Realistic transit loops with multiple stops
- Weather events that persist over time periods and affect all buses
- Traffic conditions correlated with weather
- Passenger boarding/alighting based on time of day
- Route detours and temporary stop closures
- Simulated delays based on various factors
- 24/7 bus operations with time-appropriate passenger loads

## Project Implementation

All project code is contained in a single comprehensive Jupyter notebook:

```
jupyter notebook main.ipynb
```

The notebook contains:

1. **Data Preparation Pipeline**: 
   - Loading GPS and stop-level data
   - Feature engineering (temporal, distance, weather/traffic, historical)
   - Data preprocessing and splitting
   - The `BusDataPrep` class implements all data preparation steps

2. **Linear Regression Model**:
   - Implementation via `BusLinearRegressionModel` class
   - Delay and arrival time prediction
   - Feature importance analysis
   - Performance visualization and evaluation

3. **Random Forest Model**:
   - Implementation via `BusRandomForestModel` class
   - Support for hyperparameter tuning
   - Feature importance analysis
   - Performance visualization and evaluation

4. **Model Comparison**:
   - Side-by-side performance metrics
   - Visualization of model predictions
   - Analysis of prediction accuracy
   - Improvement percentages between models

## Model Evaluation

The models are evaluated using multiple metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Percentage of predictions within different time windows (1, 2, 5, 10 minutes)

Comparative analysis shows:
- Random Forest outperforms Linear Regression across all metrics
- Weather and traffic conditions are significant predictors of delays
- Time of day and passenger activity also influence arrival predictions
- Feature importance analysis provides insights for future improvements

## Project Documentation

- `project-timeline.txt`: Development log following project phases
- `model_implementation_plan.txt`: Detailed steps for model implementation and comparison
- `main.ipynb`: Comprehensive Jupyter notebook with full analysis pipeline

## Authors

- Daneliz Urena
- Rohan Sharma
