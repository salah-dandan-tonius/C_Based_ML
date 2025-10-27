import os
import sys

build_dir = os.path.abspath("Build")
sys.path.insert(0, build_dir)

import pandas as pd
import numpy as np
import mymodule           # pybind11 C model
import joblib
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_dir = "Data/Models"
parquet_file = "Data/Parquet/orion-pipeline-2024-08-11.00.parquet"

df = pd.read_parquet(parquet_file)
X_raw = df.drop(['EventType', 'SourceIP', 'TCP', 'ICMP', 'Country'], axis=1, errors='ignore')
Y_true = df['EventType'].to_numpy()

preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
X_proc = preprocessor.transform(X_raw)
X_proc = np.ascontiguousarray(X_proc, dtype=np.float64)

c_model_funcs = {
    "DecisionTree": mymodule.predict_decision_tree,
    "RandomForest": mymodule.predict_random_forest,
    "LogisticRegression": mymodule.predict_logistic_regression
}

for name, func in c_model_funcs.items():
    classes = joblib.load(os.path.join(model_dir, f"{name}_classes.joblib"))
    output_dim = len(classes)
    input_dim = X_proc.shape[1]

    c_preds = func(X_proc, input_dim, output_dim)  # call the specific method
    pred_idx = np.argmax(c_preds, axis=1)
    Y_pred = classes[pred_idx]

    c_path = os.path.join(model_dir, f"{name}_m2c.c")
    disk_size = os.path.getsize(c_path)
    print(f"C {name} metrics:")
    print(f"Accuracy: {accuracy_score(Y_true, Y_pred):.4f}")
    print(f"Precision: {precision_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"F1: {f1_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"Disk size (bytes, C source file): {disk_size}")
    print("-----")

for name in ["DecisionTree", "RandomForest", "LogisticRegression"]:
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    sess = ort.InferenceSession(onnx_path)

    input_feed = {}
    for col in X_raw.columns:
        arr = X_raw[col].to_numpy().reshape(-1, 1)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32)    # ONNX expects float
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        input_feed[col] = arr

    onnx_preds = sess.run(None, input_feed)[0]

    pred_idx = np.argmax(onnx_preds, axis=1)
    classes = joblib.load(os.path.join(model_dir, f"{name}_classes.joblib"))
    Y_pred = classes[pred_idx]

    print(f"ONNX {name} metrics:")
    print(f"Accuracy: {accuracy_score(Y_true, Y_pred):.4f}")
    print(f"Precision: {precision_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"F1: {f1_score(Y_true, Y_pred, average='weighted'):.4f}")
    print(f"Disk size (bytes): {os.path.getsize(onnx_path)}")
    print("-----")

