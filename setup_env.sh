#!/bin/bash

echo "Setting up Python environment for Rutgers Bus Arrival Time Prediction project..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment: .venv"
  python3 -m venv .venv
else
  echo "Using existing Python virtual environment: .venv"
fi

# Activate the environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
echo "Installing required packages from requirements.txt"
pip install -r requirements.txt

# Check if jupyter is installed, if not install it
if ! command -v jupyter &> /dev/null; then
    echo "Installing Jupyter..."
    pip install jupyter
fi

echo "Environment setup complete!"
echo "IMPORTANT: Always run project code with the environment activated:"
echo "source .venv/bin/activate"

# Ask user if they want to generate synthetic data
echo ""
read -p "Would you like to generate synthetic data now? (y/n): " generate_data

if [[ $generate_data == "y" || $generate_data == "Y" || $generate_data == "yes" || $generate_data == "Yes" || $generate_data == "YES" ]]; then
  echo "Generating synthetic dataset. This may take a few minutes..."
  python3 syntheticDataset.py
  echo "Synthetic data generation complete!"
else
  echo ""
  echo "You can generate the synthetic dataset later by running:"
  echo "source .venv/bin/activate"
  echo "python3 syntheticDataset.py"
fi 

echo ""
echo "-------------------------------------------------------------------------------------------------------"
echo "Would you like to start Jupyter Notebook server automatically? (y/n)"
echo "If yes, we'll start the server and provide you with the direct link to access the notebook."
echo "If no, we'll provide detailed instructions for manual startup."
read -p "Start Jupyter server automatically? (y/n): " start_jupyter

if [[ $start_jupyter == "y" || $start_jupyter == "Y" || $start_jupyter == "yes" || $start_jupyter == "Yes" || $start_jupyter == "YES" ]]; then
  echo "Starting Jupyter Notebook server..."
  # Start Jupyter in the background and capture the token
  jupyter notebook --no-browser > jupyter.log 2>&1 &
  JUPYTER_PID=$!
  
  # Wait a moment for the server to start
  sleep 3
  
  # Extract the token from the log file
  TOKEN=$(grep -o 'token=[^ ]*' jupyter.log | head -n 1)
  PORT=$(grep -o 'http://localhost:[0-9]*' jupyter.log | head -n 1 | grep -o '[0-9]*$')
  
  if [ -n "$TOKEN" ] && [ -n "$PORT" ]; then
    echo ""
    echo "Jupyter Notebook server is running!"
    echo "Click this link to open the notebook in your browser:"
    echo "http://localhost:$PORT/?$TOKEN"
    echo ""
    echo "To stop the server when you're done, run: kill $JUPYTER_PID"
  else
    echo "Failed to start Jupyter server. Please try the manual method below."
    kill $JUPYTER_PID 2>/dev/null
    start_jupyter="n"
  fi
fi

if [[ $start_jupyter != "y" && $start_jupyter != "Y" && $start_jupyter != "yes" && $start_jupyter != "Yes" && $start_jupyter != "YES" ]]; then
  echo ""
  echo "Next steps:"
  echo "1. To start working with the project, you have two options:"
  echo ""
  echo "   Option A - Run in your browser:"
  echo "   a. Open your web browser"
  echo "   b. Go to http://localhost:8888"
  echo "   c. Look for 'main.ipynb' in the file list"
  echo ""
  echo "   Option B - Run from terminal:"
  echo "   a. Open a new terminal"
  echo "   b. Run: source .venv/bin/activate"
  echo "   c. Run: jupyter notebook"
  echo "   d. Click on 'main.ipynb' in the browser window that opens"
  echo ""
  echo "2. For more information about the project, please read the README.md file"
  echo ""
  echo "Note: If you see a token/password prompt, you can find the token in the terminal output."
  echo "      Look for a line that starts with 'token=' in the terminal where you ran jupyter notebook."
fi

echo "-------------------------------------------------------------------------------------------------------"