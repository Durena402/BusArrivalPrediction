# Rutgers Bus Arrival Time Prediction

A data science project to enhance the accuracy of Rutgers campus bus arrival time predictions using synthetic data and supervised learning techniques.

## Project Overview

This project aims to improve bus arrival time predictions by integrating multiple data sources:

- Bus GPS tracking data
- Weather conditions
- Campus events and schedules

Using supervised learning models, we predict arrival times more accurately than existing methods.

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

## Project Structure

- `syntheticDataset.py`: Generates synthetic bus data with realistic route modeling
- `requirements.txt`: Package dependencies
- `setup_env.sh`: Environment setup script
- `project-timeline.txt`: Development log following project phases

## Authors

- Daneliz Urena
- Rohan Sharma
