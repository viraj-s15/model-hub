#!/bin/bash
# Bash script to set up a Python environment with specific packages

# Step 1: Creating a Python virtual environment
echo "Step 1: Creating a Python virtual environment..."
python3 -m venv venv

# Step 2: Activating the virtual environment
echo "Step 2: Activating the virtual environment..."
source venv/bin/activate

# Step 3: Installing Python packages
echo "Step 3: Installing Python packages..."
echo " - Installing 'transformers', 'accelarate', and 'bitsandbytes'..."
pip3 install -q transformers accelarate bitsandbytes

# Step 4: Installing PyTorch and related packages with a custom index-url
echo "Step 4: Installing PyTorch and related packages with a custom index URL..."
echo " - Installing 'torch', 'torchvision', and 'torchaudio' with a custom index URL..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Step 5: Deactivating the virtual environment
echo "Step 5: Deactivating the virtual environment..."
deactivate

# Step 6: Optional - Provide a message indicating successful setup
echo "Python environment setup completed."

