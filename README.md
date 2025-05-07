# Rutgers Bus Arrival Time Prediction

A data science project to enhance the accuracy of Rutgers campus bus arrival time predictions using synthetic data and supervised learning techniques.

## Project Overview

This project aims to improve bus arrival time predictions by integrating multiple data sources:

- Bus GPS tracking data
- Weather conditions
- Campus events and schedules

Using supervised learning models, we predict arrival times more accurately than existing methods.

## Research Focus

This project compares multiple machine learning approaches to bus arrival prediction:

- **Neural Networks**: LSTM and GRU architectures for capturing temporal dependencies
- **Traditional Machine Learning**: Linear Regression and Random Forest models as baselines
- **Comparative Analysis**: Evaluation of model performance, training efficiency, and prediction accuracy

The research aims to determine which modeling approach best captures the complex relationships between weather, traffic, passenger patterns, and bus arrival times.

## Setup Instructions

### Environment Setup

1. Ensure you have Python installed (Python 3.8+ recommended)
2. Run the setup script to create a Python virtual environment:
   ```
   ./setup_env.sh
   ```
3. Activate the environment before running any project code:
   ```
   source rbat_env/bin/activate
   ```

### Important Rule

**ALWAYS run project code with the Python environment activated.**

## Data Generation

The project uses synthetic data to simulate real-world bus operations. To generate the dataset:

```
python syntheticDataset.py
```

This will create two files:

- `bus_gps_tracking_data.csv`: GPS tracking points for buses
- `bus_stop_level_data.csv`: Stop-level arrival/departure data

## Data Preparation

After generating synthetic data, prepare it for modeling by running:

```
python data_prep.py
```

This pipeline:

1. Engineers temporal, distance, weather/traffic, and historical features
2. Scales numerical features and one-hot encodes categorical variables
3. Handles missing values and prepares target variables
4. Creates train/validation/test splits (70%/15%/15%)
5. Prepares sequential data for LSTM/GRU models
6. Saves processed data to the `processed_data/` directory

## Model Development

The project follows a structured approach to model development:

1. **Data Preparation**: Feature engineering, normalization, and train/test splitting
2. **Baseline Models**: Implementation of Linear Regression and Random Forest
3. **Neural Networks**: LSTM and GRU architectures using TensorFlow
4. **Comparative Analysis**: Evaluation using MAE, RMSE, and other metrics
5. **Visualization**: Performance comparison and prediction analysis

See `model_implementation_plan.txt` for detailed implementation steps.

## Project Structure

- `syntheticDataset.py`: Generates synthetic bus data with realistic route modeling
- `data_prep.py`: Data preparation pipeline with feature engineering
- `requirements.txt`: Package dependencies
- `setup_env.sh`: Environment setup script
- `project-timeline.txt`: Development log following project phases
- `model_implementation_plan.txt`: Detailed steps for model implementation and comparison

## Authors

- Daneliz Urena
- Rohan Sharma
