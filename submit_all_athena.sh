#!/bin/bash
# Master Athena submission script
# Run this on Athena login node: bash submit_all_athena.sh

set -e

SCRATCH_DIR="$SCRATCH/hackathon"
mkdir -p "$SCRATCH_DIR"

echo "=== Setting up on Athena ==="
echo "SCRATCH=$SCRATCH"
echo "Working dir: $SCRATCH_DIR"

# Copy solution files from home to scratch
cp -r ~/hackathon_solutions/* "$SCRATCH_DIR/" 2>/dev/null || true

# Create the SLURM batch script
cat > "$SCRATCH_DIR/submit_job.slurm" << 'SLURM_END'
#!/bin/bash -l
#SBATCH -J hackathon-submit
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --time=01:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=32GB
#SBATCH --output=submit_%J.out

module add GCCcore/13.2.0 Python/3.11.5

# Setup venv on scratch
VENV_DIR="$SCRATCH/venvs/hackathon"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    mkdir -p "$SCRATCH/venvs"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip --no-cache-dir --require-virtualenv install requests python-dotenv 2>&1 | tail -3

cd "$SCRATCH/hackathon"

export TEAM_TOKEN="${TEAM_TOKEN:-b2c6083ba78b4039a6db64a4bb5e07ca}"
export SERVER_URL="http://149.156.182.9:6060"

echo ""
echo "=== Submitting Task 1 ==="
if [ -f "task1/chebi_submission.parquet" ]; then
    python3 submit_solution.py task1 task1/chebi_submission.parquet parquet_file
else
    echo "SKIP: task1/chebi_submission.parquet not found"
fi

echo ""
echo "=== Submitting Task 2 (practice) ==="
if [ -f "task2/python-practice-smart.jsonl" ]; then
    python3 submit_solution.py task2 task2/python-practice-smart.jsonl jsonl_file practice
else
    echo "SKIP: task2 practice predictions not found"
fi

echo ""
echo "=== Submitting Task 2 (public) ==="
if [ -f "task2/python-public-smart.jsonl" ]; then
    python3 submit_solution.py task2 task2/python-public-smart.jsonl jsonl_file public
else
    echo "SKIP: task2 public predictions not found"
fi

echo ""
echo "=== Submitting Task 3 ==="
if [ -f "task3/load_submission.csv" ]; then
    python3 submit_solution.py task3 task3/load_submission.csv csv_file
else
    echo "SKIP: task3/load_submission.csv not found"
fi

echo ""
echo "=== Submitting Task 4 ==="
if [ -f "task4/ecg_submission.npz" ]; then
    python3 submit_solution.py task4 task4/ecg_submission.npz npz_file
else
    echo "SKIP: task4/ecg_submission.npz not found"
fi

echo ""
echo "=== All submissions done ==="
SLURM_END

echo "SLURM script created: $SCRATCH_DIR/submit_job.slurm"
echo ""
echo "To submit: cd $SCRATCH_DIR && sbatch submit_job.slurm"
echo "To check: squeue --me"
echo "To see output: cat $SCRATCH_DIR/submit_*.out"
