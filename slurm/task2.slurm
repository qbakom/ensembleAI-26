#!/bin/bash -l
#SBATCH -J task2-context
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=32GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module add GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/hackathon/bin/activate

cd $SCRATCH/ensembleAI-26/task2/EnsembleAI2026-starter-kit

echo "=== Task 2: Code Context Retrieval ==="
echo "Node: $(hostname)"
date

# First check if python-dataset.zip needs extracting
DATA_DIR="$SCRATCH/ensembleAI-26/task2"
if [ -f "$DATA_DIR/python-dataset.zip" ] && [ ! -d "$DATA_DIR/extracted" ]; then
    echo "Extracting python-dataset.zip..."
    mkdir -p "$DATA_DIR/extracted"
    cd "$DATA_DIR/extracted"
    unzip -o "$DATA_DIR/python-dataset.zip"
    cd $SCRATCH/ensembleAI-26/task2/EnsembleAI2026-starter-kit
fi

# Determine where archives are
# They could be in extracted/ or directly in task2/
if [ -d "$DATA_DIR/extracted" ]; then
    ARCHIVE_DIR="$DATA_DIR/extracted"
    # Check if archives are in a subdirectory
    if ls "$ARCHIVE_DIR"/*.zip 1>/dev/null 2>&1; then
        echo "Archives found in $ARCHIVE_DIR"
    else
        # Try one level deeper
        SUBDIR=$(ls -d "$ARCHIVE_DIR"/*/ 2>/dev/null | head -1)
        if [ -n "$SUBDIR" ]; then
            ARCHIVE_DIR="$SUBDIR"
            echo "Archives found in $ARCHIVE_DIR"
        fi
    fi
else
    ARCHIVE_DIR="$DATA_DIR"
fi

echo "Using archives from: $ARCHIVE_DIR"
echo "Input file: $DATA_DIR/python-test.jsonl"

# Run solve_task2.py with test dataset
python3 -u solve_task2.py \
    --input "$DATA_DIR/python-test.jsonl" \
    --data-dir "$ARCHIVE_DIR" \
    --output predictions/python-test-smart.jsonl \
    2>&1

echo "=== Task 2 prediction done ==="
date

# Submit test predictions
if [ -f predictions/python-test-smart.jsonl ]; then
    echo "Submitting test..."
    cd $SCRATCH/ensembleAI-26
    python3 submit_solution.py task2 \
        $SCRATCH/ensembleAI-26/task2/EnsembleAI2026-starter-kit/predictions/python-test-smart.jsonl \
        jsonl_file test
fi

echo "=== Task 2 complete ==="
date
