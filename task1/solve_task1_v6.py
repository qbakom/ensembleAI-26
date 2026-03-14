"""
Task 1 v6: ChEBI Multi-Label Classification - Resume-capable
Saves checkpoints every 50 classifiers, can resume from where it left off.
"""
import os, sys, time, warnings, traceback, pickle
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.linear_model import SGDClassifier

CHECKPOINT = '/tmp/task1_checkpoint.pkl'

def main():
    t0 = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Task 1 v6: ChEBI (SGD, checkpoint-capable)", flush=True)

    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')
    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    n_classes = len(label_cols)
    print(f"Train={len(train_df)}, Test={len(test_df)}, Classes={n_classes}", flush=True)

    n_bits = 1024
    print("Computing fingerprints...", flush=True)

    X_train = np.zeros((len(train_df), n_bits), dtype=np.float32)
    for i, smi in enumerate(train_df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            ConvertToNumpyArray(fp, X_train[i])

    X_test = np.zeros((len(test_df), n_bits), dtype=np.float32)
    for i, smi in enumerate(test_df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            ConvertToNumpyArray(fp, X_test[i])

    y_train = train_df[label_cols].values.astype(np.int8)
    print(f"Fingerprints done in {time.time()-t0:.1f}s", flush=True)

    # Try to resume from checkpoint
    start_i = 0
    predictions = np.zeros((len(test_df), n_classes), dtype=np.int8)
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, 'rb') as f:
                ckpt = pickle.load(f)
            start_i = ckpt['i']
            predictions = ckpt['predictions']
            print(f"Resumed from checkpoint at {start_i}/{n_classes}", flush=True)
        except:
            print("Checkpoint corrupt, starting fresh", flush=True)

    print(f"Training classifiers {start_i}-{n_classes}...", flush=True)

    for i in range(start_i, n_classes):
        pos = int(y_train[:, i].sum())
        neg = len(y_train) - pos

        if pos == 0:
            continue
        if neg == 0:
            predictions[:, i] = 1
            continue

        clf = SGDClassifier(
            loss='log_loss',
            max_iter=50,
            random_state=42,
            tol=1e-3,
            class_weight={0: 1.0, 1: neg / max(pos, 1)},
            n_jobs=1,
        )
        clf.fit(X_train, y_train[:, i])
        predictions[:, i] = clf.predict(X_test).astype(np.int8)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_classes} ({time.time()-t0:.0f}s)", flush=True)

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({'i': i + 1, 'predictions': predictions}, f)

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    submission = test_df[['mol_id', 'SMILES']].copy()
    for j, col in enumerate(label_cols):
        submission[col] = predictions[:, j].astype(int)

    output_path = 'chebi_submission.parquet'
    submission.to_parquet(output_path, index=False)

    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns)
    assert len(submission) == len(example)

    print(f"Saved {output_path}, shape={submission.shape}", flush=True)
    print(f"Positive rate: {predictions.mean():.4f}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
