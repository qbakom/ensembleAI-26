# Instrukcje dla Claude Code na Athenie

## Kontekst
Hackathon EnsembleAI 2026, 3 taski do submitowania. Repo jest w `$SCRATCH/ensembleAI-26` (branch `dev`).
Serwer submitów: `http://149.156.182.9:6060`, token: `b2c6083ba78b4039a6db64a4bb5e07ca`.
Rate limit: **600 sekund** między submitami — nie submituj szybciej.

## KROK 1: Setup środowiska

```bash
cd $SCRATCH/ensembleAI-26
git pull origin dev

# Jeśli venv nie istnieje:
mkdir -p $SCRATCH/venvs
python3 -m venv $SCRATCH/venvs/hackathon
source $SCRATCH/venvs/hackathon/bin/activate
pip install pandas pyarrow scikit-learn lightgbm rdkit rank_bm25 requests python-dotenv
```

Jeśli venv już istnieje — tylko aktywuj i doinstaluj brakujące:
```bash
source $SCRATCH/venvs/hackathon/bin/activate
pip install lightgbm rank_bm25
```

## KROK 2: Sprawdź czy pliki danych są na miejscu

```bash
# Task 1 — powinny być w repo
ls -la $SCRATCH/ensembleAI-26/task1/chebi_dataset_train.parquet
ls -la $SCRATCH/ensembleAI-26/task1/chebi_dataset_test_empty.parquet
ls -la $SCRATCH/ensembleAI-26/task1/chebi_classes.obo

# Task 2 — potrzebny python-dataset.zip i python-test.jsonl
ls -la $SCRATCH/ensembleAI-26/task2/python-test.jsonl
ls -la $SCRATCH/ensembleAI-26/task2/python-dataset.zip

# Task 3 — dane powinny być w task3/data/
ls -la $SCRATCH/ensembleAI-26/task3/data/
```

Jeśli czegoś brakuje — powiedz userowi.

## KROK 3: Odpal joby SLURM (wszystkie 3 na raz)

```bash
cd $SCRATCH/ensembleAI-26
sbatch slurm/task2.slurm
sbatch slurm/task1.slurm
sbatch slurm/task3.slurm
```

Sprawdź czy joby weszły do kolejki:
```bash
squeue -u $USER
```

## KROK 4: Monitoruj joby

```bash
# Sprawdź status
squeue -u $USER

# Podglądaj logi na żywo (jak job ruszy):
tail -f $SCRATCH/ensembleAI-26/task2-context_*.out
tail -f $SCRATCH/ensembleAI-26/task1-chebi_*.out
tail -f $SCRATCH/ensembleAI-26/task3-heatpump_*.out

# Po zakończeniu — sprawdź czy outputy istnieją:
ls -la $SCRATCH/ensembleAI-26/task2/solution/predictions/python-public-smart.jsonl
ls -la $SCRATCH/ensembleAI-26/task1/chebi_submission.parquet
ls -la $SCRATCH/ensembleAI-26/task3/data/out/load_submission.csv
```

## KROK 5: Ręczny submit (jeśli auto-submit nie zadziałał)

Każdy SLURM ma auto-submit na końcu, ale gdyby nie zadziałał:

```bash
source $SCRATCH/venvs/hackathon/bin/activate
cd $SCRATCH/ensembleAI-26

# TASK 2 (priorytet — najlepiej przetestowane)
python3 submit_solution.py task2 task2/solution/predictions/python-public-smart.jsonl jsonl_file public

# POCZEKAJ 600 SEKUND (10 minut)

# TASK 1
python3 submit_solution.py task1 task1/chebi_submission.parquet parquet_file

# POCZEKAJ 600 SEKUND

# TASK 3
python3 submit_solution.py task3 task3/data/out/load_submission.csv csv_file
```

## KROK 6: Sprawdź wyniki submitów

Z outputu submita dostaniesz `request_id`. Użyj go:

```bash
cd $SCRATCH/ensembleAI-26
python3 shared/get_task_status.py --request-id <REQUEST_ID>
```

## Debugowanie

Jeśli job SLURM failuje — sprawdź error logi:
```bash
cat $SCRATCH/ensembleAI-26/task1-chebi_*.err
cat $SCRATCH/ensembleAI-26/task2-context_*.err
cat $SCRATCH/ensembleAI-26/task3-heatpump_*.err
```

Typowe problemy:
- `ModuleNotFoundError` → doinstaluj pakiet w venvie
- `FileNotFoundError` → sprawdź czy dane są na $SCRATCH
- `ConnectionRefusedError` przy submicie → upewnij się że jesteś na Athenie (serwer jest tylko w sieci Cyfronetu)

## Co submitujemy — podsumowanie

| Task | Co robi solver | Plik wynikowy | Komenda submit |
|------|---------------|---------------|----------------|
| **Task 2** | AST chunk-based context collection, BM25+TF-IDF+Jaccard ranking | `task2/solution/predictions/python-public-smart.jsonl` | `python3 submit_solution.py task2 <plik> jsonl_file public` |
| **Task 1** | LightGBM + Morgan+MACCS+RDKit fingerprints + DAG post-processing | `task1/chebi_submission.parquet` | `python3 submit_solution.py task1 <plik> parquet_file` |
| **Task 3** | Per-device seasonal patterns + temperature correlation | `task3/data/out/load_submission.csv` | `python3 submit_solution.py task3 <plik> csv_file` |
