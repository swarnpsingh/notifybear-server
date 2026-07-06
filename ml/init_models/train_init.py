import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ast
import pandas as pd
import shutil

from ml.model import NotificationClassifier

CSV_PATH = "training_features.csv"

df = pd.read_csv(CSV_PATH)

# Keep only labeled rows
df = df[df["label"].notna()].copy()

X = []
y = []

for _, row in df.iterrows():
    features = row["features"]

    if isinstance(features, str):
        features = ast.literal_eval(features)

    X.append(features)

    # Same behavior as your server
    y.append(int(float(row["label"])))

clf = NotificationClassifier()

clf.train(X, y)

print("Python model:", clf.model_path)
print("ONNX model:", clf.onnx_path)

# Replace your init model
shutil.copy(clf.onnx_path, "ml/init_models/init.onnx")

print("init.onnx updated successfully!")