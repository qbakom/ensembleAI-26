#!/bin/bash -l
#SBATCH -J task1-chebi
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=32GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module add GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/hackathon/bin/activate

cd $SCRATCH/ensembleAI-26/task1

echo "=== Task 1: ChEBI Classification ==="
echo "Node: $(hostname), CPUs: $SLURM_NTASKS_PER_NODE"
date

python3 -u solve_task1_v7.py --n-jobs 16 2>&1

echo "=== Task 1 done ==="
date

# Auto-submit if output exists
if [ -f chebi_submission.parquet ]; then
    echo "Submitting..."
    cd $SCRATCH/ensembleAI-26
    python3 submit_solution.py task1 task1/chebi_submission.parquet parquet_file
fi
