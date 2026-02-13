#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing default packages..."
    pip install ultralytics opencv-python numpy psutil
fi

# Run YOLO setup script to verify installation and download model
echo "Setting up YOLO..."
python setup_yolo.py

echo "================================================="
echo "Setup complete!"
echo "To activate the environment, run: source $VENV_DIR/bin/activate"
echo "================================================="
