import pandas as pd
import numpy as np
import mymodule
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load raw data
df = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")
X_raw = df.drop(['EventType', 'SourceIP', 'TCP', 'ICMP', 'Country'], axis=1, errors='ignore')
y_true = df['EventType'].to_numpy()

# Identify numeric and categorical columns
numeric_features = X_raw.select_dtypes(include='number').columns.tolist()
categorical_features = X_raw.select_dtypes(exclude='number').columns.tolist()

# Recreate preprocessing pipeline
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# Fit the preprocessor on the raw X
preprocessor.fit(X_raw)
X_proc = preprocessor.transform(X_raw)
X_proc = np.ascontiguousarray(X_proc, dtype=np.float64)

# Determine model dimensions
input_dim = X_proc.shape[1]
# You must match this to your m2cgen model's number of outputs
# For DecisionTreeClassifier example:
model = DecisionTreeClassifier()
model.fit(X_proc, df['EventType'])  # just to get classes_
classes = model.classes_
output_dim = len(classes)

# Run the C model
pred_matrix = mymodule.predict(X_proc, input_dim, output_dim)

# Convert to predicted labels
pred_idx = np.argmax(pred_matrix, axis=1)
y_pred = classes[pred_idx]

# Evaluation
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y_true, y_pred))
