import os

import joblib
import numpy as np


def predict_direct(model_path, values):
    """
    Load the model and make predictions directly without Triton
    """
    model = joblib.load(model_path)

    input_data = np.array(values, dtype=np.float32)
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)

    predictions = model.predict(input_data)

    return predictions


def main():
    model_path = "model_repository/sklearn_linear/1/model.joblib"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run export_model.py first.")
        return

    test_values = np.array([[6], [7], [8], [9], [10]], dtype=np.float32)

    print("Input values:", test_values.flatten())

    predictions = predict_direct(model_path, test_values)

    print("Predictions:", predictions.flatten())

    expected = test_values * 2
    print("Expected values:", expected.flatten())

    mse = np.mean((predictions - expected.flatten()) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

    model = joblib.load(model_path)
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")


if __name__ == "__main__":
    main()
