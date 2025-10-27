import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import m2cgen as m2c
import joblib

model_dir = "Data/Models"
os.makedirs(model_dir, exist_ok=True)

df = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")
X = df.drop('EventType', axis=1)
Y = df['EventType']

numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()

num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

preprocessor.fit(X)
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

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

    pipeline.fit(X_train, Y_train)

    # Save ONNX
    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        onnx_path = os.path.join(model_dir, f"{name}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"{name} saved as ONNX: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed for {name}: {e}")

    # Save m2cgen C
    code = m2c.export_to_c(model)
    func_name = f"score_{name}"
    code = code.replace("void score(", f"void {func_name}(")
    
    c_path = os.path.join(model_dir, f"{name}_m2c.c")
    with open(c_path, "w") as f:
        f.write(code)
    print(f"{name} exported to C: {c_path}")

    # Save class labels for C model mapping
    joblib.dump(model.classes_, os.path.join(model_dir, f"{name}_classes.joblib"))
