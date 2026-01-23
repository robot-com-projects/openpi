# Installing openpi into a Conda Environment

This guide explains how to install openpi dependencies into an existing conda environment (e.g., `45pysaac`) that already has ROS and Isaac Sim.

## Prerequisites

1. **Python 3.11**: openpi requires Python 3.11 (not 3.10 or 3.12)
   ```bash
   # Check your Python version
   conda activate 45pysaac
   python --version  # Should be 3.11.x
   
   # If not, recreate the environment:
   # conda create -n 45pysaac python=3.11
   # conda activate 45pysaac
   # # Then reinstall ROS and Isaac Sim dependencies
   ```

2. **uv installed**: Install uv if you haven't already
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or: pip install uv
   ```

## Installation Methods

### Method 1: Using the Installation Script (Recommended)

```bash
cd /home/zetans/Documents/openpi
./install_to_conda.sh 45pysaac
```

### Method 2: Manual Installation

1. **Activate your conda environment:**
   ```bash
   conda activate 45pysaac
   ```

2. **Navigate to the openpi directory:**
   ```bash
   cd /home/zetans/Documents/openpi
   ```

3. **Install openpi and dependencies using uv:**
   ```bash
   # Get your Python executable path
   PYTHON_EXE=$(which python)
   
   # Install openpi in editable mode (this installs all dependencies from pyproject.toml)
   GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import openpi; print('openpi installed at:', openpi.__file__)"
   python -c "from openpi.policies import policy_config; print('Policies module loaded successfully')"
   ```

## Optional: Install Additional Dependency Groups

If you need simulation dependencies (mujoco, rerun):

```bash
conda activate 45pysaac
cd /home/zetans/Documents/openpi
PYTHON_EXE=$(which python)
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PYTHON_EXE" -e ".[sim]"
```

## Troubleshooting

### Issue: Python version mismatch
- **Error**: `requires-python = ">=3.11,<3.12"`
- **Solution**: Make sure your conda environment uses Python 3.11
  ```bash
  conda activate 45pysaac
  python --version  # Should show 3.11.x
  ```

### Issue: Dependency conflicts with ROS/Isaac Sim
- Some packages might conflict. You can try:
  1. Install openpi first, then ROS/Isaac Sim
  2. Or use `--no-deps` and install dependencies manually
  3. Or create a separate conda environment for openpi and use it alongside your ROS/Isaac Sim environment

### Issue: CUDA/JAX version conflicts
- openpi requires `jax[cuda12]==0.5.3` which needs CUDA 12
- If you have CUDA 11, you may need to adjust the JAX version or use CPU-only JAX

### Issue: Git dependencies not found
- Make sure submodules are initialized:
  ```bash
  cd /home/zetans/Documents/openpi
  git submodule update --init --recursive
  ```

## Notes

- The `GIT_LFS_SKIP_SMUDGE=1` flag skips downloading large Git LFS files during installation (you can download them later if needed)
- The `-e` flag installs in editable mode, so changes to the code are immediately available
- `uv pip install` respects the `pyproject.toml` file and will install all dependencies listed there

