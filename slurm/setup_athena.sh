#!/bin/bash -l
# Run this on Athena login node to set up the environment
# Usage: bash setup_athena.sh

set -e

module add GCCcore/13.2.0 Python/3.11.5

# Create venv if not exists
if [ ! -d "$SCRATCH/venvs/hackathon" ]; then
    mkdir -p $SCRATCH/venvs
    python3 -m venv $SCRATCH/venvs/hackathon
fi

source $SCRATCH/venvs/hackathon/bin/activate

# Install all dependencies
pip --no-cache-dir --require-virtualenv install \
    pandas numpy requests python-dotenv fastparquet pyarrow \
    scikit-learn rdkit lightgbm \
    rank_bm25 jsonlines \
    2>&1 | tail -5

echo "=== Setup complete ==="
echo "Venv: $SCRATCH/venvs/hackathon"
python3 --version
