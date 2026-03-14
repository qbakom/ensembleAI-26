#!/bin/bash -l
#SBATCH -J submit-all
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=4GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module add GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/hackathon/bin/activate

cd $SCRATCH/ensembleAI-26

echo "=== Submitting all tasks ==="
date

# Task 1
if [ -f task1/chebi_submission.parquet ]; then
    echo "--- Task 1 ---"
    python3 submit_solution.py task1 task1/chebi_submission.parquet parquet_file
    echo "Waiting 450s for rate limit..."
    sleep 455
fi

# Task 2 practice
if [ -f $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-practice-smart.jsonl ]; then
    echo "--- Task 2 practice ---"
    python3 submit_solution.py task2 $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-practice-smart.jsonl jsonl_file practice
    echo "Waiting 600s for rate limit..."
    sleep 605
fi

# Task 2 public
if [ -f $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-public-smart.jsonl ]; then
    echo "--- Task 2 public ---"
    python3 submit_solution.py task2 $SCRATCH/EnsembleAI2026-starter-kit/predictions/python-public-smart.jsonl jsonl_file public
    echo "Waiting 2700s for rate limit..."
    sleep 2705
fi

# Task 3
if [ -f task3/data/out/load_submission.csv ]; then
    echo "--- Task 3 ---"
    python3 submit_solution.py task3 task3/data/out/load_submission.csv csv_file
    echo "Waiting 450s for rate limit..."
    sleep 455
fi

echo "=== All submissions done ==="
date
