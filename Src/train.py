import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

df: pd.DataFrame = pd.read_parquet("Data/Parquet/orion-pipeline-2024-08-11.00.parquet")

numeric_features = df.select_dtypes(include='number').columns.tolist()
categorical_features = df.select_dtypes(exclude='number').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
])

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

target = 'EventType'
X = df.drop(columns=[target])
y = df[target]

clf.fit(X, y)
preds = clf.predict(X)

print(preds)

