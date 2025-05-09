import os

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression

os.makedirs("model_repository/sklearn_linear/1", exist_ok=True)

X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

model = LinearRegression()
model.fit(X, y)

model_path = "model_repository/sklearn_linear/1/model.joblib"
joblib.dump(model, model_path)
