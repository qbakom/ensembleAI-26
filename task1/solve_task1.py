"""
Task 1: ChEBI Ontology Prediction
- Hierarchical multi-label classification for molecules
- Input: SMILES strings -> 500 binary class predictions
- Metric: macro-averaged F1 score
- Strategy: ECFP + MACCS fingerprints + LightGBM per-class (fast)
"""

import io
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from dotenv import load_dotenv

load_dotenv()


def smiles_to_features(smiles_list):
    """Convert SMILES to ECFP4 + MACCS fingerprints."""
    n = len(smiles_list)
    ecfp = np.zeros((n, 2048), dtype=np.float32)
    maccs = np.zeros((n, 167), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ConvertToNumpyArray(fp, ecfp[i])
            mfp = MACCSkeys.GenMACCSKeys(mol)
            ConvertToNumpyArray(mfp, maccs[i])

    return np.hstack([ecfp, maccs])


def parse_obo_hierarchy(obo_path):
    """Parse OBO file to get class hierarchy for consistency enforcement."""
    # Read the OBO file and extract class IDs and is_a relationships
    class_ids = []
    parents_map = {}  # chebi_id -> [parent_chebi_ids]
    current_id = None

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: CHEBI:'):
                current_id = line.split('id: ')[1].strip()
            elif line.startswith('is_a: CHEBI:') and current_id:
                parent = line.split('is_a: ')[1].strip().split(' !')[0].strip()
                if current_id not in parents_map:
                    parents_map[current_id] = []
                parents_map[current_id].append(parent)

    return parents_map


def build_class_hierarchy(obo_path, class_definitions_path):
    """Build mapping from class indices to parent class indices using the OBO and definitions."""
    # Read class definitions to get ChEBI ID -> class index mapping
    class_defs = pd.read_csv(class_definitions_path)

    # Build chebi_id -> class_index mapping
    chebi_to_idx = {}
    for _, row in class_defs.iterrows():
        chebi_id = str(row.iloc[0]).strip()
        class_idx = int(row.iloc[1]) if len(class_defs.columns) > 1 else None
        if 'CHEBI:' in chebi_id:
            chebi_to_idx[chebi_id] = class_idx

    # Parse OBO hierarchy
    parents_map = parse_obo_hierarchy(obo_path)

    # Build index-based parent mapping
    idx_parents = {}  # class_idx -> [parent_class_idx]
    for chebi_id, parents in parents_map.items():
        if chebi_id in chebi_to_idx:
            child_idx = chebi_to_idx[chebi_id]
            if child_idx is not None:
                parent_indices = []
                for p in parents:
                    if p in chebi_to_idx and chebi_to_idx[p] is not None:
                        parent_indices.append(chebi_to_idx[p])
                if parent_indices:
                    idx_parents[child_idx] = parent_indices

    return idx_parents


def enforce_hierarchy_predictions(predictions, idx_parents):
    """If a child class is predicted as 1, ensure all ancestors are also 1."""
    # Build full ancestor map (transitive closure)
    def get_all_ancestors(idx, parents_map, visited=None):
        if visited is None:
            visited = set()
        if idx in visited:
            return set()
        visited.add(idx)
        ancestors = set()
        if idx in parents_map:
            for p in parents_map[idx]:
                ancestors.add(p)
                ancestors.update(get_all_ancestors(p, parents_map, visited))
        return ancestors

    n_classes = predictions.shape[1]
    ancestor_map = {}
    for idx in range(n_classes):
        ancestors = get_all_ancestors(idx, idx_parents)
        if ancestors:
            ancestor_map[idx] = [a for a in ancestors if a < n_classes]

    # Enforce: if child=1, all ancestors must be 1
    fixed = predictions.copy()
    for child_idx, ancestor_indices in ancestor_map.items():
        mask = fixed[:, child_idx] == 1
        if mask.any():
            for anc in ancestor_indices:
                fixed[mask, anc] = 1

    n_fixes = (fixed != predictions).sum()
    return fixed, n_fixes


def main():
    t0 = time.time()
    print("=" * 60)
    print("Task 1: ChEBI Ontology Multi-Label Classification")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')

    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    print(f"Train: {len(train_df)} samples, {len(label_cols)} classes")
    print(f"Test: {len(test_df)} samples")

    # Extract features
    print("\nExtracting features...")
    X_train = smiles_to_features(train_df['SMILES'].tolist())
    y_train = train_df[label_cols].values
    X_test = smiles_to_features(test_df['SMILES'].tolist())
    print(f"Features: {X_train.shape[1]} dims, took {time.time()-t0:.1f}s")

    # Train per-class LightGBM models
    print("\nTraining 500 LightGBM classifiers...")
    predictions = np.zeros((len(test_df), len(label_cols)), dtype=np.int64)

    for i, col in enumerate(label_cols):
        pos_count = int(y_train[:, i].sum())
        neg_count = len(y_train) - pos_count

        if pos_count == 0:
            predictions[:, i] = 0
            continue
        if pos_count == len(y_train):
            predictions[:, i] = 1
            continue

        scale_pos = neg_count / max(pos_count, 1)

        clf = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            num_leaves=63,
            learning_rate=0.1,
            n_estimators=150,
            scale_pos_weight=max(scale_pos, 0.001),
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train[:, i])
        proba = clf.predict_proba(X_test)[:, 1]
        predictions[:, i] = (proba >= 0.5).astype(np.int64)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(label_cols)} done ({time.time()-t0:.0f}s)")

    print(f"\nTraining done in {time.time()-t0:.0f}s")

    # Try to enforce hierarchy
    try:
        idx_parents = build_class_hierarchy('chebi_classes.obo', 'chebi_class_definitions.csv')
        if idx_parents:
            predictions, n_fixes = enforce_hierarchy_predictions(predictions, idx_parents)
            print(f"Hierarchy enforcement: {n_fixes} fixes applied")
    except Exception as e:
        print(f"Hierarchy enforcement skipped: {e}")

    # Build submission
    print("\nBuilding submission...")
    submission = test_df[['mol_id', 'SMILES']].copy()
    for i, col in enumerate(label_cols):
        submission[col] = predictions[:, i]

    output_path = 'chebi_submission.parquet'
    submission.to_parquet(output_path, index=False)

    # Verify format
    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns), "Column mismatch!"
    assert len(submission) == len(example), "Row count mismatch!"
    print(f"Submission saved: {output_path} ({submission.shape})")
    print(f"Total time: {time.time()-t0:.0f}s")

    return output_path


if __name__ == "__main__":
    main()
