# EnsembleAI 2026 - Submission Instructions

## Prerequisites
Submissions MUST be sent from Athena (http://149.156.182.6/).
Login: tutorial243 / Password: pybtujumra

## Files Ready for Submission

### Task 1: ChEBI Ontology Classification
- **File**: `task1/chebi_submission.parquet`
- **Endpoint**: POST `http://149.156.182.9:6060/task1`
- **Key**: `parquet_file`
- **Status**: Training in progress (LightGBM ~500 classifiers)

### Task 2: Code Context Retrieval
- **Practice file**: `task2/EnsembleAI2026-starter-kit/predictions/python-practice-multisignal.jsonl` (47 predictions)
- **Public file**: `task2/EnsembleAI2026-starter-kit/predictions/python-public-multisignal.jsonl` (247 predictions)
- **Endpoint**: POST `http://149.156.182.9:6060/task2`
- **Key**: `jsonl_file`, form data: `stage=practice` or `stage=public`
- **Status**: READY

### Task 3: Heat Pump Load Forecasting
- **v1 file**: `task3/data/out/load_submission.csv` (3600 rows)
- **v2 file**: `task3/data/out/load_submission_v2.csv` (with temperature model - pending)
- **Endpoint**: POST `http://149.156.182.9:6060/task3`
- **Key**: `csv_file`
- **Status**: v1 READY, v2 processing

### Task 4: ECG Digitization
- **File**: `task4/data/out/ecg_submission.npz` (baseline - synthetic signals)
- **Endpoint**: POST `http://149.156.182.9:6060/task4`
- **Key**: `npz_file`
- **Status**: Baseline ready (dataset download blocked by Google Drive quota)

## Quick Submit Script
Copy files to Athena, then run:
```bash
cd ensembleAI-26
source .venv/bin/activate  # or install deps: pip install requests python-dotenv
python submit_all.py
```

Or submit individual tasks:
```bash
python submit_all.py 2    # Just Task 2
python submit_all.py 1 3  # Tasks 1 and 3
```

## Check Status
```python
import requests
headers = {"X-API-Token": "f624a0295f9b4b359395e12a12ca0f2d"}
r = requests.get("http://149.156.182.9:6060/status/<request_id>", headers=headers)
print(r.json())
```
