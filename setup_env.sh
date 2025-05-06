#!/bin/bash

echo "Setting up Python environment for Rutgers Bus Arrival Time Prediction project..."

# Create virtual environment if it doesn't exist
if [ ! -d "rbat_env" ]; then
  echo "Creating virtual environment 'rbat_env'..."
  python -m venv rbat_env
else
  echo "Virtual environment 'rbat_env' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source rbat_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -q numpy pandas matplotlib scikit-learn jupyter seaborn

echo "Setup complete! Use 'source rbat_env/bin/activate' to activate the environment."
echo "IMPORTANT: Always run this project with the Python environment activated." 