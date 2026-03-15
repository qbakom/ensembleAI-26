"""
Task 1 v4: ChEBI Multi-Label Classification
Simple, reliable: ECFP4 fingerprints + SGDClassifier
"""
import os, sys, time, warnings, traceback
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.linear_model import SGDClassifier

def main():
    t0 = time.time()
    print("Task 1 v4: ChEBI Classification", flush=True)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')
    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    n_classes = len(label_cols)
    print(f"Train={len(train_df)}, Test={len(test_df)}, Classes={n_classes}", flush=True)

    # ECFP4 only (2048 bits) - enough for good performance
    n_bits = 2048
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

    # Train 500 classifiers
    print("Training classifiers...", flush=True)
    predictions = np.zeros((len(test_df), n_classes), dtype=np.int8)

    for i in range(n_classes):
        try:
            pos = int(y_train[:, i].sum())
            neg = len(y_train) - pos

            if pos == 0:
                predictions[:, i] = 0
                continue
            if pos == len(y_train):
                predictions[:, i] = 1
                continue

            clf = SGDClassifier(
                loss='modified_huber',
                max_iter=100,
                random_state=42,
                tol=1e-3,
                class_weight={0: 1.0, 1: neg / max(pos, 1)},
                n_jobs=1,
            )
            clf.fit(X_train, y_train[:, i])

            proba = clf.predict_proba(X_test)
            if proba.shape[1] == 2:
                predictions[:, i] = (proba[:, 1] >= 0.5).astype(np.int8)
            else:
                predictions[:, i] = clf.predict(X_test).astype(np.int8)

        except Exception as e:
            print(f"  ERROR class {i}: {e}", flush=True)
            predictions[:, i] = 0

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_classes} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    # Save submission
    submission = test_df[['mol_id', 'SMILES']].copy()
    for i, col in enumerate(label_cols):
        submission[col] = predictions[:, i].astype(int)

    output_path = 'chebi_submission.parquet'
    submission.to_parquet(output_path, index=False)

    # Verify
    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns)
    assert len(submission) == len(example)

    pos_rate = predictions.mean()
    print(f"Saved {output_path}, shape={submission.shape}", flush=True)
    print(f"Positive rate: {pos_rate:.4f}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
