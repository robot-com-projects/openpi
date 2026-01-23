#!/bin/bash
# Script to install openpi into a Python 3.10 conda environment
# This bypasses the Python 3.11 requirement for compatibility with Isaac Sim
# Usage: ./install_py310.sh <conda_env_name>

set -e

CONDA_ENV_NAME="${1:-45pysaac}"

echo "Installing openpi into Python 3.10 conda environment: $CONDA_ENV_NAME"
echo "Note: This bypasses the Python 3.11 requirement for Isaac Sim compatibility"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Get the conda environment path
CONDA_ENV_PATH=$(conda env list | grep "^${CONDA_ENV_NAME}" | awk '{print $2}')

if [ -z "$CONDA_ENV_PATH" ]; then
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Found conda environment at: $CONDA_ENV_PATH"
echo ""

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Get the Python executable path
PYTHON_EXE=$(which python)
echo "Using Python: $PYTHON_EXE"
echo ""

# Change to openpi directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Backup pyproject.toml
if [ ! -f "pyproject.toml.backup" ]; then
    echo "Backing up pyproject.toml..."
    cp pyproject.toml pyproject.toml.backup
fi

# Temporarily modify pyproject.toml to allow Python 3.10
echo "Temporarily modifying pyproject.toml to allow Python 3.10..."
sed -i.tmp 's/requires-python = ">=3.11,<3.12"/requires-python = ">=3.10,<3.12"/' pyproject.toml

# Clean up temp file
rm -f pyproject.toml.tmp

echo "Installing build dependencies first..."
# Install common build backends and tools needed for various packages
# - hatchling: for openpi and openpi-client
# - editables: for editable installs
# - scikit-build-core: for ruckig (dependency of i2rt)
# - nanobind: CMake dependency required by ruckig
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" hatchling editables scikit-build-core nanobind

echo ""
echo "Installing openpi dependencies using uv..."
echo ""

# Install dependencies using uv pip with --no-build-isolation since we installed hatchling
# This bypasses the build isolation and uses the installed hatchling
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" --no-build-isolation -e . || {
    echo ""
    echo "Installation with --no-build-isolation failed, trying without editable mode..."
    echo "Installing in regular mode..."
    
    # Try installing without editable mode
    GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" --no-build-isolation . || {
        echo "Installation failed. Trying to install dependencies manually..."
        
        # Restore pyproject.toml
        if [ -f "pyproject.toml.backup" ]; then
            mv pyproject.toml.backup pyproject.toml
        fi
        
        echo "Please install dependencies manually. Some packages may have compatibility issues with Python 3.10."
        exit 1
    }
    
    # If non-editable install worked, try editable again
    echo "Retrying editable install..."
    GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" --no-build-isolation -e .
}

# Restore original pyproject.toml
echo ""
echo "Restoring original pyproject.toml..."
if [ -f "pyproject.toml.backup" ]; then
    mv pyproject.toml.backup pyproject.toml
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  python -c 'import openpi; print(\"openpi installed at:\", openpi.__file__)'"
echo ""
echo "Note: Some features may not work correctly with Python 3.10. If you encounter issues,"
echo "consider using Python 3.11 in a separate environment for openpi development."

