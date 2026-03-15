"""
Task 1 v3: Fast ChEBI Classification using LogisticRegression with saga solver
Much faster than SGD with modified_huber, similar quality
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.linear_model import RidgeClassifier

def smiles_to_fp(smiles_list):
    n = len(smiles_list)
    fps = np.zeros((n, 2215), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ConvertToNumpyArray(ecfp, fps[i, :2048])
            maccs = MACCSkeys.GenMACCSKeys(mol)
            arr167 = np.zeros(167, dtype=np.float32)
            ConvertToNumpyArray(maccs, arr167)
            fps[i, 2048:] = arr167
    return fps

def main():
    t0 = time.time()
    print("Task 1 v3: ChEBI Multi-Label Classification (LogisticRegression)", flush=True)

    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')
    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Classes: {len(label_cols)}", flush=True)

    print("Computing fingerprints...", flush=True)
    X_train = smiles_to_fp(train_df['SMILES'].tolist())
    X_test = smiles_to_fp(test_df['SMILES'].tolist())
    y_train = train_df[label_cols].values
    print(f"Fingerprints done in {time.time()-t0:.1f}s", flush=True)

    predictions = np.zeros((len(test_df), len(label_cols)), dtype=np.int64)

    print("Training classifiers...", flush=True)
    for i in range(len(label_cols)):
        pos = int(y_train[:, i].sum())
        neg = len(y_train) - pos

        if pos == 0:
            predictions[:, i] = 0
            continue
        if pos == len(y_train):
            predictions[:, i] = 1
            continue

        # RidgeClassifier is extremely fast for binary classification
        clf = RidgeClassifier(
            alpha=1.0,
            class_weight='balanced',
        )
        clf.fit(X_train, y_train[:, i])
        predictions[:, i] = clf.predict(X_test)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(label_cols)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    # Try hierarchy enforcement
    try:
        from solve_task1 import build_class_hierarchy, enforce_hierarchy_predictions
        idx_parents = build_class_hierarchy('chebi_classes.obo', 'chebi_class_definitions.csv')
        if idx_parents:
            predictions, n_fixes = enforce_hierarchy_predictions(predictions, idx_parents)
            print(f"Hierarchy enforcement: {n_fixes} fixes applied", flush=True)
    except Exception as e:
        print(f"Hierarchy enforcement skipped: {e}", flush=True)

    # Build submission
    submission = test_df[['mol_id', 'SMILES']].copy()
    for i, col in enumerate(label_cols):
        submission[col] = predictions[:, i]

    output_path = 'chebi_submission.parquet'
    submission.to_parquet(output_path, index=False)

    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns)
    assert len(submission) == len(example)
    print(f"Saved {output_path}, shape={submission.shape}, time={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
