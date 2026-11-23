"""Export the trained random forest to JSON so the browser can run it.

Run after retraining:
    python training/export_model.py

Writes frontend/src/model/model.json, which frontend/src/model/forest.js reads.
Each tree is stored as parallel arrays (the same layout scikit-learn uses):
    left/right  child node index, -1 at a leaf
    feature     which input value to compare (index into the 63 landmark values)
    threshold   go left when value <= threshold
    value       at a leaf, the class probabilities
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'ALX_model_local.joblib'
OUT_PATH = ROOT / 'frontend' / 'src' / 'model' / 'model.json'


def export_tree(tree):
    t = tree.tree_
    leaf = t.children_left == -1
    # Normalize leaf counts to probabilities, rounded to keep the file small.
    # Thresholds are NOT rounded: training points sit right next to them.
    values = t.value[:, 0, :]
    values = values / values.sum(axis=1, keepdims=True)
    return {
        'left': t.children_left.tolist(),
        'right': t.children_right.tolist(),
        'feature': [int(f) if not l else -1 for f, l in zip(t.feature, leaf)],
        'threshold': [float(x) if not l else 0 for x, l in zip(t.threshold, leaf)],
        'value': [[round(float(p), 4) for p in row] if l else 0 for row, l in zip(values, leaf)],
    }


def main():
    clf = joblib.load(MODEL_PATH)
    model = {
        'classes': [str(c) for c in clf.classes_],
        'nFeatures': int(clf.n_features_in_),
        'trees': [export_tree(est) for est in clf.estimators_],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(model, separators=(',', ':')))
    print(f'Wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB, {len(model["trees"])} trees)')


if __name__ == '__main__':
    main()
