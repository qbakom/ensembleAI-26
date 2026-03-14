#!/bin/bash -l
# Run from Athena login node to submit task3 directly (no SLURM needed for just a POST request)
module add GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/hackathon/bin/activate
cd $SCRATCH/ensembleAI-26
python3 submit_solution.py task3 task3/data/out/load_submission.csv csv_file
