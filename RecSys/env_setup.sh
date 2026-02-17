#!/usr/bin/env bash
set -e

# Script to create a conda environment matching README.md requirements
# Usage: bash env_setup.sh

ENV_NAME=core_env
PYTHON=3.7
CUDA=10.1

echo "Creating conda env '$ENV_NAME' with Python $PYTHON and CUDA $CUDA..."
conda create -n "$ENV_NAME" python="$PYTHON" cudatoolkit="$CUDA" -y

# Activate the environment in a non-interactive script
# This uses conda's shell hook so activation works inside scripts
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch 1.7.1 (CUDA $CUDA) from pytorch channel..."
conda install pytorch==1.7.1 cudatoolkit=$CUDA -c pytorch -y

echo "Upgrading pip and installing RecBole..."
python -m pip install --upgrade pip
python -m pip install recbole==1.0.1

# Optional: common extras (uncomment if you want them)
# python -m pip install tqdm numpy scipy scikit-learn

echo "Environment '$ENV_NAME' created. Verification commands:"
echo "conda activate $ENV_NAME"
echo "python -V"
echo "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "python -c 'import recbole; print(recbole.__version__)'"

echo "Done. To run the project's training use:"
echo "conda activate $ENV_NAME && python main.py --model trm --dataset diginetica"
