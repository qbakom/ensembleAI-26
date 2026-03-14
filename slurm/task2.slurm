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

echo "=== Task 2: Chunk-based Context Collection ==="
echo "Node: $(hostname)"
date

# Paths
REPO_DIR="$SCRATCH/ensembleAI-26"
TASK2_DIR="$REPO_DIR/task2"
SOLUTION_DIR="$TASK2_DIR/solution"
STARTER_DIR="$TASK2_DIR/EnsembleAI2026-starter-kit"

# Install rank_bm25 if not present
pip install rank_bm25 2>/dev/null

# Check if python-dataset.zip needs extracting
if [ -f "$TASK2_DIR/python-dataset.zip" ] && [ ! -d "$TASK2_DIR/python-dataset" ]; then
    echo "Extracting python-dataset.zip..."
    cd "$TASK2_DIR"
    unzip -o python-dataset.zip -d python-dataset
fi

# Find where the zip archives are
ARCHIVE_DIR=""
if [ -d "$TASK2_DIR/python-dataset" ]; then
    # Check for zips directly or in subdirectory
    if ls "$TASK2_DIR/python-dataset/"*.zip 1>/dev/null 2>&1; then
        ARCHIVE_DIR="$TASK2_DIR/python-dataset"
    else
        # Try one level deeper
        for d in "$TASK2_DIR/python-dataset/"*/; do
            if ls "$d"*.zip 1>/dev/null 2>&1; then
                ARCHIVE_DIR="$d"
                break
            fi
        done
    fi
fi

# Fallback: use starter kit data dir
if [ -z "$ARCHIVE_DIR" ]; then
    ARCHIVE_DIR="$STARTER_DIR/data"
fi

echo "Archives directory: $ARCHIVE_DIR"
echo "Input file: $TASK2_DIR/python-test.jsonl"

# Also check for prepare_data.sh in starter kit
if [ -d "$STARTER_DIR" ] && [ -f "$STARTER_DIR/prepare_data.sh" ]; then
    cd "$STARTER_DIR"
    # Copy test data if needed
    if [ -f "$TASK2_DIR/python-test.jsonl" ] && [ ! -f "$STARTER_DIR/data/python-public.jsonl" ]; then
        cp "$TASK2_DIR/python-test.jsonl" "$STARTER_DIR/data/python-public.jsonl"
    fi
fi

cd "$SOLUTION_DIR"
mkdir -p predictions

# Run chunk-based context collection
python3 -u solve_task2.py \
    --input "$TASK2_DIR/python-test.jsonl" \
    --data-dir "$ARCHIVE_DIR" \
    --output predictions/python-public-smart.jsonl \
    --max-context-chars 60000 \
    --max-chunks 30 \
    2>&1

echo "=== Prediction done ==="
date

# Submit with stage=public
if [ -f predictions/python-public-smart.jsonl ]; then
    echo "Submitting to server (stage=public)..."
    cd "$REPO_DIR"
    python3 submit_solution.py task2 \
        "$SOLUTION_DIR/predictions/python-public-smart.jsonl" \
        jsonl_file public
    echo "Submitted! Check status in ~20-30 min with get_task_status.py"
fi

echo "=== Task 2 complete ==="
date
