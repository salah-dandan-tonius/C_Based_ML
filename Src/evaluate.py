# USED TO EVALUATE ONNX MODELS

import os
import onnxruntime as ort
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")
df = df.drop(['SourceIP', 'TCP', 'ICMP', 'Country'], axis=1, errors='ignore')

X = df.drop('EventType', axis=1)
y = df['EventType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def decode_onnx_predictions(y_pred, y_train):
    """
    Decode ONNX model output to original labels.
    y_pred: numpy array output from ONNX
    y_train: original target series to get classes
    """
    # Case 1: probabilities
    if y_pred.dtype.kind == 'f' and y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
        classes = np.array(y_train.unique())
        return classes[y_pred_idx]
    
    # Case 2: integer indices
    elif np.issubdtype(y_pred.dtype, np.integer):
        classes = np.array(y_train.unique())
        return classes[y_pred]
    
    # Case 3: already string labels
    else:
        return y_pred.ravel()


model_dir = "Data/Models"
results = []

def prepare_inputs(df):
    inputs = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            inputs[col] = df[col].astype(str).to_numpy().reshape(-1, 1)
        else:
            inputs[col] = df[col].to_numpy().astype(np.float32).reshape(-1, 1)
    return inputs

def memory_usage(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    result = func(*args, **kwargs)
    mem_after = process.memory_info().rss
    return result, mem_after - mem_before

# Evaluate all ONNX models
for model_file in os.listdir(model_dir):
    if model_file.endswith(".onnx"):
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace(".onnx", "")
        
        disk_size = os.path.getsize(model_path)
        
        session, mem_used = memory_usage(ort.InferenceSession, model_path)
        
        X_test_inputs = prepare_inputs(X_test)
        y_pred = session.run(None, X_test_inputs)[0]
        
        if y_pred.dtype.kind == 'f' and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        y_pred = decode_onnx_predictions(y_pred, y)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Model": model_name,
            "Disk_Size_Bytes": disk_size,
            "Memory_Usage_Bytes": mem_used,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })

# Save to csv
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(model_dir, "onnx_model_metrics.csv"), index=False)
print("Metrics saved to:", os.path.join(model_dir, "onnx_model_metrics.csv"))
print(results_df)
