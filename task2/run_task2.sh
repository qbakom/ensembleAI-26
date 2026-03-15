#!/bin/bash -l
#SBATCH -J task2-ast-bm25
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=16GB
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err

set -e

module add GCCcore/13.2.0 Python/3.11.5

VENV="$SCRATCH/venvs/hackathon"
if [ ! -d "$VENV" ]; then
    echo "Tworzę venv..."
    mkdir -p "$SCRATCH/venvs"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip --no-cache-dir --require-virtualenv install -q jsonlines rank_bm25

cd /net/tscratch/people/tutorial243/EnsembleAI2026-starter-kit

echo "=== Task 2: AST+BM25 pipeline ==="
echo "Node: $(hostname), Python: $(python3 --version)"
date

SCRATCH=/net/tscratch/people/tutorial243
INPUT="$SCRATCH/EnsembleAI2026-starter-kit/data/new_public/python-test.jsonl"
REPOS="$SCRATCH/EnsembleAI2026-starter-kit/data/new_public/repositories-python-public/python-dataset"
OUTPUT="$SCRATCH/EnsembleAI2026-starter-kit/predictions/python-public-new-ast-bm25.jsonl"

python3 -u solve_task2_ast_bm25.py \
    --input "$INPUT" \
    --repos-dir "$REPOS" \
    --output "$OUTPUT"

echo "=== Generowanie gotowe, wysyłam... ==="
date

python3 -c "
import requests
with open('$OUTPUT', 'rb') as f:
    r = requests.post(
        'http://149.156.182.9:6060/task2',
        files={'jsonl_file': f},
        data={'stage': 'public'},
        headers={'X-API-Token': 'b2c6083ba78b4039a6db64a4bb5e07ca'}
    )
print('Status:', r.status_code)
print('Response:', r.json())
"

echo "=== Done ==="
date
