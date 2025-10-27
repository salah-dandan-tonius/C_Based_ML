import os
import ctypes
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")
df = df.drop(['SourceIP', 'TCP', 'ICMP', 'Country'], axis=1, errors='ignore')

from sklearn.model_selection import train_test_split
X = df.drop('EventType', axis=1)
y = df['EventType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_cols = ['Port', 'Traffic', 'Packets', 'Bytes', 'UniqueDests',
                'UniqueDest24s', 'Lat', 'Long', 'ASN']

def decode_m2c_predictions(y_pred, y_train):
    """Convert numeric m2cgen output to closest class labels."""
    classes = np.array(sorted(y_train.unique()))
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    if np.all((y_pred >= -1) & (y_pred < len(classes) + 1)):
        y_pred_idx = np.clip(np.round(y_pred).astype(int), 0, len(classes) - 1)
    else:
        centers = np.arange(len(classes))
        y_pred_idx = np.array([
            int(np.argmin(np.abs(centers - val))) for val in y_pred
        ])

    return classes[y_pred_idx]

def memory_usage(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    result = func(*args, **kwargs)
    mem_after = process.memory_info().rss
    return result, mem_after - mem_before

def predict_row(lib, row_values):
    row_array = (ctypes.c_double * len(row_values))(*row_values)
    out_array = (ctypes.c_double * 1)()
    lib.score(row_array, out_array)
    return out_array[0]

def decode_m2c_predictions(y_pred, y_train):
    """Map numeric m2cgen output to original labels."""
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred_int = y_pred.astype(int)
        classes = np.array(y_train.unique())
        return decode_m2c_predictions(y_pred, y_train)
    else:
        return y_pred.ravel()

model_dir = "Data/Models"
results = []

for model_file in os.listdir(model_dir):
    if model_file.endswith("_m2c.so"):
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace("_m2c.so", "")

        disk_size = os.path.getsize(model_path)

        lib, mem_used = memory_usage(ctypes.CDLL, model_path) # REMOVE THE MEM USED
        lib.score.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.score.restype = None

        X_test_values = X_test[feature_cols].to_numpy().astype(np.float64)

        y_pred = np.array([predict_row(lib, row) for row in X_test_values])

        y_pred_labels = decode_m2c_predictions(y_pred, y)

        acc = accuracy_score(y_test, y_pred_labels)
        prec = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)

        results.append({
            "Model": model_name,
            "Disk_Size_Bytes": disk_size,
            #"Memory_Usage_Bytes": mem_used, mem_used shouldn't be there
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })
        print(f"Evaluated {model_name}: Accuracy={acc:.4f}")

# save to csv
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(model_dir, "c_model_metrics.csv"), index=False)
print("Metrics saved to:", os.path.join(model_dir, "c_model_metrics.csv"))
print(results_df)
