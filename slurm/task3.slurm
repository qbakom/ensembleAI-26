#!/bin/bash -l
#SBATCH -J task3-heatpump
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

cd $SCRATCH/ensembleAI-26/task3

echo "=== Task 3: Heat Pump Forecasting ==="
echo "Node: $(hostname)"
date

python3 -u solve_task3.py 2>&1

echo "=== Task 3 done ==="
date

# Auto-submit if output exists
if [ -f data/out/load_submission.csv ]; then
    echo "Submitting..."
    cd $SCRATCH/ensembleAI-26
    python3 submit_solution.py task3 task3/data/out/load_submission.csv csv_file
fi
