#!/bin/bash

echo "Setting up Python environment for Rutgers Bus Arrival Time Prediction project..."

# Create virtual environment if it doesn't exist
if [ ! -d "rbat_env" ]; then
  echo "Creating Python virtual environment: rbat_env"
  python3 -m venv rbat_env
else
  echo "Using existing Python virtual environment: rbat_env"
fi

# Activate the environment
source rbat_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
echo "Installing required packages from requirements.txt"
pip install -r requirements.txt

echo "Environment setup complete!"
echo "IMPORTANT: Always run project code with the environment activated:"
echo "source rbat_env/bin/activate"

# Ask user if they want to generate synthetic data
echo ""
read -p "Would you like to generate synthetic data now? (y/n): " generate_data

if [[ $generate_data == "y" || $generate_data == "Y" ]]; then
  echo "Generating synthetic dataset. This may take a few minutes..."
  python3 syntheticDataset.py
  echo "Synthetic data generation complete!"
else
  echo ""
  echo "You can generate the synthetic dataset later by running:"
  echo "source rbat_env/bin/activate"
  echo "python3 syntheticDataset.py"
fi 