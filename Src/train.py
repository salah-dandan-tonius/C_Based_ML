import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import m2cgen as m2c

model_dir = "Data/Models"
os.makedirs(model_dir, exist_ok=True)

df = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")
df = df.drop(['SourceIP', 'TCP', 'ICMP', 'Country'], axis=1, errors='ignore')

X = df.drop('EventType', axis=1)
y = df['EventType']

numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Creates a list of data types that represent one model input, since ONNX needs that. For example,
# if the inputs are {Port, Bytes, EventType}, then the variable initial_types = {int, int, str} because
# because Port is an integer, Bytes is an integer, and EventType is a string (for example, "TCP Backscatter")
initial_types = []
for col in X.columns:
    if X[col].dtype == 'object':
        initial_types.append((col, StringTensorType([None, 1])))
    else:
        initial_types.append((col, FloatTensorType([None, 1])))

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # save onnx
    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        onnx_path = os.path.join(model_dir, f"{name}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"{name} model saved as {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed for {name}: {e}")
    
    # save m2cgen code
    code = m2c.export_to_c(model)
    c_path = os.path.join(model_dir, f"{name}_m2c.c")
    with open(c_path, "w") as f:
        f.write(code)
    print(f"{name} model exported to {c_path} using m2cgen")
