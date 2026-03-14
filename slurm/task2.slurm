#!/bin/bash -l
#SBATCH -J task2-context
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module add GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/hackathon/bin/activate

cd $SCRATCH/EnsembleAI2026-starter-kit

echo "=== Task 2: Code Context Retrieval ==="
echo "Node: $(hostname)"
date

# Generate predictions for practice
echo "--- Practice stage ---"
python3 -u smart_context.py --stage practice --lang python 2>&1

# Generate predictions for public
echo "--- Public stage ---"
python3 -u smart_context.py --stage public --lang python 2>&1

echo "=== Task 2 done ==="
date

# Auto-submit
if [ -f predictions/python-practice-smart.jsonl ]; then
    echo "Submitting practice..."
    cd $SCRATCH/ensembleAI-26
    python3 submit_solution.py task2 $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-practice-smart.jsonl jsonl_file practice
fi

sleep 610  # Wait for rate limit (600s)

if [ -f $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-public-smart.jsonl ]; then
    echo "Submitting public..."
    cd $SCRATCH/ensembleAI-26
    python3 submit_solution.py task2 $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-public-smart.jsonl jsonl_file public
fi
