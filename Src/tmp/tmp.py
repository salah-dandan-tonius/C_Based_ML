import numpy as np
import lightgbm as lgb
from skl2onnx import convert_sklearn
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

# Random dummy dataset
X = np.random.rand(100, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.rand(100)) > 1.5  # simple pattern

# LightGBM dataset
train_data = lgb.Dataset(X, label=y)
params = {
    'objective': 'binary',
    'verbosity': -1
}
# Train the LightGBM model
model = lgb.train(params, train_data, num_boost_round=10)

# Convert to ONNX
initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_lightgbm(model, initial_types=initial_type)

with open("lgbm.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model saved")

session = ort.InferenceSession(
    "lgbm.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

X_test = np.random.rand(5, 4).astype(np.float32)
preds = session.run([output_name], {input_name: X_test})[0]

print("Sample predictions:", preds)