#!/bin/bash
# Quick fix script to upgrade typing_extensions
# Sentinel was added in typing_extensions 4.5.0

set -e

CONDA_ENV_NAME="${1:-45pysaac}"

echo "Fixing typing_extensions in conda environment: $CONDA_ENV_NAME"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Check current version
echo "Checking current typing_extensions version..."
python3 -c "import typing_extensions; print('Current version:', typing_extensions.__version__)" 2>&1 || echo "typing_extensions not installed"

echo ""
echo "Upgrading typing_extensions to >=4.12.2 (required by openpi)..."
uv pip install --upgrade "typing-extensions>=4.12.2"

echo ""
echo "Verifying installation..."
python3 -c "from typing_extensions import Sentinel; print('✓ Sentinel import successful')"

echo ""
echo "✓ typing_extensions upgraded successfully!"

