"""
Task 1 Ridge: Ultra-fast ChEBI Classification using RidgeClassifier
~5 min total for 500 classes
"""
import os, time, warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.linear_model import RidgeClassifier

def main():
    t0 = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Task 1 Ridge: ChEBI Classification", flush=True)

    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')
    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    print(f"Train={len(train_df)}, Test={len(test_df)}, Classes={len(label_cols)}", flush=True)

    print("Computing fingerprints...", flush=True)
    n = len(train_df)
    m = len(test_df)
    X_train = np.zeros((n, 2215), dtype=np.float32)
    X_test = np.zeros((m, 2215), dtype=np.float32)

    for i, smi in enumerate(train_df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ConvertToNumpyArray(fp, X_train[i, :2048])
            mfp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167, dtype=np.float32)
            ConvertToNumpyArray(mfp, arr)
            X_train[i, 2048:] = arr

    for i, smi in enumerate(test_df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ConvertToNumpyArray(fp, X_test[i, :2048])
            mfp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167, dtype=np.float32)
            ConvertToNumpyArray(mfp, arr)
            X_test[i, 2048:] = arr

    y_train = train_df[label_cols].values
    print(f"Fingerprints done in {time.time()-t0:.1f}s", flush=True)

    predictions = np.zeros((m, len(label_cols)), dtype=np.int64)
    print("Training classifiers...", flush=True)

    for i in range(len(label_cols)):
        pos = int(y_train[:, i].sum())
        if pos == 0:
            predictions[:, i] = 0
            continue
        if pos == len(y_train):
            predictions[:, i] = 1
            continue

        clf = RidgeClassifier(alpha=1.0, class_weight='balanced')
        clf.fit(X_train, y_train[:, i])
        predictions[:, i] = clf.predict(X_test)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(label_cols)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    # Hierarchy enforcement
    try:
        from solve_task1 import build_class_hierarchy, enforce_hierarchy_predictions
        idx_parents = build_class_hierarchy('chebi_classes.obo', 'chebi_class_definitions.csv')
        if idx_parents:
            predictions, n_fixes = enforce_hierarchy_predictions(predictions, idx_parents)
            print(f"Hierarchy: {n_fixes} fixes", flush=True)
    except Exception as e:
        print(f"Hierarchy skipped: {e}", flush=True)

    submission = test_df[['mol_id', 'SMILES']].copy()
    for i, col in enumerate(label_cols):
        submission[col] = predictions[:, i]

    output_path = 'chebi_submission_ridge.parquet'
    submission.to_parquet(output_path, index=False)

    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns)
    assert len(submission) == len(example)
    print(f"Saved {output_path}, shape={submission.shape}, time={time.time()-t0:.0f}s", flush=True)
    print(f"Positive rate: {predictions.mean():.4f}", flush=True)

if __name__ == "__main__":
    main()
