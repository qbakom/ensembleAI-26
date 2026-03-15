"""
Task 1 v7: ChEBI Multi-Label Classification
LightGBM + Morgan(2048) + MACCS(166) + RDKit(2048) fingerprints
DAG hierarchical consistency post-processing
Cross-validation for local F1 estimation
"""
import os, sys, time, warnings, traceback, argparse
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import lightgbm as lgb


def compute_fingerprints(smiles_list, morgan_bits=2048, rdkit_bits=2048):
    """Compute combined fingerprints: Morgan + MACCS + RDKit."""
    n = len(smiles_list)
    # Morgan: 2048, MACCS: 167, RDKit: 2048 => 4263 features
    total_bits = morgan_bits + 167 + rdkit_bits
    X = np.zeros((n, total_bits), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        offset = 0

        # Morgan fingerprint (circular, radius=2)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits)
        arr = np.zeros(morgan_bits, dtype=np.float32)
        ConvertToNumpyArray(fp_morgan, arr)
        X[i, offset:offset + morgan_bits] = arr
        offset += morgan_bits

        # MACCS keys (166+1 bits)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        arr_maccs = np.zeros(167, dtype=np.float32)
        ConvertToNumpyArray(fp_maccs, arr_maccs)
        X[i, offset:offset + 167] = arr_maccs
        offset += 167

        # RDKit topological fingerprint
        fp_rdk = RDKFingerprint(mol, fpSize=rdkit_bits)
        arr_rdk = np.zeros(rdkit_bits, dtype=np.float32)
        ConvertToNumpyArray(fp_rdk, arr_rdk)
        X[i, offset:offset + rdkit_bits] = arr_rdk

        if (i + 1) % 5000 == 0:
            print(f"  Fingerprints: {i+1}/{n}", flush=True)

    print(f"  Fingerprints: {n}/{n} done. Shape: {X.shape}", flush=True)
    return X


def parse_obo_hierarchy(obo_path):
    """Parse ChEBI OBO file to get parent-child relationships."""
    parents = {}  # child_id -> set of parent_ids
    current_id = None

    if not os.path.exists(obo_path):
        print(f"WARNING: OBO file not found at {obo_path}, skipping DAG", flush=True)
        return {}

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: '):
                current_id = line[4:]
            elif line.startswith('is_a: '):
                parent_id = line[6:].split('!')[0].strip()
                if current_id:
                    if current_id not in parents:
                        parents[current_id] = set()
                    parents[current_id].add(parent_id)

    return parents


def get_class_to_chebi_mapping(label_cols):
    """Map class_XXXXX column names to CHEBI:XXXXX IDs."""
    mapping = {}
    for col in label_cols:
        chebi_id = col.replace('class_', 'CHEBI:')
        mapping[col] = chebi_id
    return mapping


def dag_postprocess(predictions, label_cols, obo_path):
    """
    Enforce hierarchical consistency: if a child is predicted positive,
    all ancestors must also be positive.
    """
    parents_map = parse_obo_hierarchy(obo_path)
    if not parents_map:
        return predictions

    col_to_chebi = get_class_to_chebi_mapping(label_cols)
    chebi_to_idx = {col_to_chebi[col]: i for i, col in enumerate(label_cols)}

    # Build transitive closure of ancestors
    def get_all_ancestors(chebi_id, visited=None):
        if visited is None:
            visited = set()
        if chebi_id in visited:
            return visited
        visited.add(chebi_id)
        for parent in parents_map.get(chebi_id, []):
            get_all_ancestors(parent, visited)
        return visited

    # For each class, find all ancestor indices
    ancestor_indices = {}
    for col in label_cols:
        chebi_id = col_to_chebi[col]
        ancestors = get_all_ancestors(chebi_id)
        ancestors.discard(chebi_id)  # remove self
        ancestor_idx = [chebi_to_idx[a] for a in ancestors if a in chebi_to_idx]
        if ancestor_idx:
            ancestor_indices[label_cols.index(col)] = ancestor_idx

    # Apply: if child=1, set all ancestors=1
    fixed = predictions.copy()
    n_fixes = 0
    for child_idx, anc_idxs in ancestor_indices.items():
        child_positive = fixed[:, child_idx] == 1
        for anc_idx in anc_idxs:
            needs_fix = child_positive & (fixed[:, anc_idx] == 0)
            n_fixes += needs_fix.sum()
            fixed[needs_fix, anc_idx] = 1

    print(f"  DAG post-processing: {n_fixes} fixes applied", flush=True)
    return fixed


def train_lightgbm_multilabel(X_train, y_train, X_test, label_cols, n_jobs=8):
    """Train one LightGBM per label."""
    n_classes = y_train.shape[1]
    predictions = np.zeros((X_test.shape[0], n_classes), dtype=np.int8)
    probabilities = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)

    for i in range(n_classes):
        pos = int(y_train[:, i].sum())
        neg = len(y_train) - pos

        if pos == 0:
            continue
        if neg == 0:
            predictions[:, i] = 1
            probabilities[:, i] = 1.0
            continue

        scale = neg / max(pos, 1)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_threads': n_jobs,
            'learning_rate': 0.1,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': max(5, pos // 20),
            'scale_pos_weight': scale,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train[:, i])

        prob = clf.predict_proba(X_test)[:, 1]
        probabilities[:, i] = prob
        predictions[:, i] = (prob >= 0.5).astype(np.int8)

        if (i + 1) % 25 == 0:
            print(f"  LightGBM: {i+1}/{n_classes}", flush=True)

    return predictions, probabilities


def cross_validate(X, y, label_cols, n_folds=3, n_jobs=8):
    """Quick cross-validation to estimate macro F1."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        preds, _ = train_lightgbm_multilabel(X_tr, y_tr, X_val, label_cols, n_jobs)

        # Apply DAG post-processing
        obo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chebi_classes.obo')
        preds = dag_postprocess(preds, label_cols, obo_path)

        # Macro F1
        f1s = []
        for j in range(y.shape[1]):
            if y_val[:, j].sum() > 0:
                f1s.append(f1_score(y_val[:, j], preds[:, j], zero_division=0))
        macro_f1 = np.mean(f1s) if f1s else 0.0
        fold_f1s.append(macro_f1)
        print(f"  Fold {fold+1}/{n_folds}: macro F1 = {macro_f1:.4f}", flush=True)

    mean_f1 = np.mean(fold_f1s)
    print(f"  CV mean macro F1: {mean_f1:.4f} (+/- {np.std(fold_f1s):.4f})", flush=True)
    return mean_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', action='store_true', help='Run cross-validation before training')
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--no-dag', action='store_true', help='Skip DAG post-processing')
    args = parser.parse_args()

    t0 = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Task 1 v7: ChEBI (LightGBM + Morgan+MACCS+RDKit + DAG)", flush=True)

    # Load data
    train_df = pd.read_parquet('chebi_dataset_train.parquet')
    test_df = pd.read_parquet('chebi_dataset_test_empty.parquet')
    label_cols = [c for c in train_df.columns if c.startswith('class_')]
    n_classes = len(label_cols)
    print(f"Train={len(train_df)}, Test={len(test_df)}, Classes={n_classes}", flush=True)

    # Compute fingerprints
    print("Computing training fingerprints...", flush=True)
    X_train = compute_fingerprints(train_df['SMILES'].tolist())
    print(f"Computing test fingerprints...", flush=True)
    X_test = compute_fingerprints(test_df['SMILES'].tolist())
    y_train = train_df[label_cols].values.astype(np.int8)
    print(f"Fingerprints done in {time.time()-t0:.1f}s", flush=True)

    # Optional cross-validation
    if args.cv:
        print(f"\n=== Cross-validation ({args.cv_folds} folds) ===", flush=True)
        cross_validate(X_train, y_train, label_cols, args.cv_folds, args.n_jobs)

    # Full training
    print(f"\n=== Full training ===", flush=True)
    predictions, probabilities = train_lightgbm_multilabel(
        X_train, y_train, X_test, label_cols, args.n_jobs
    )
    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    # DAG post-processing
    if not args.no_dag:
        print("\n=== DAG post-processing ===", flush=True)
        obo_path = 'chebi_classes.obo'
        predictions = dag_postprocess(predictions, label_cols, obo_path)

    # Save submission
    submission = test_df[['mol_id', 'SMILES']].copy()
    for j, col in enumerate(label_cols):
        submission[col] = predictions[:, j].astype(int)

    output_path = 'chebi_submission.parquet'
    submission.to_parquet(output_path, index=False)

    # Validate format
    example = pd.read_parquet('chebi_submission_example.parquet')
    assert list(submission.columns) == list(example.columns), "Column mismatch!"
    assert len(submission) == len(example), "Row count mismatch!"

    print(f"\nSaved {output_path}, shape={submission.shape}", flush=True)
    print(f"Positive rate: {predictions.mean():.4f}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
