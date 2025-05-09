import numpy as np
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)
from tritonclient.utils import np_to_triton_dtype


def get_client():
    return InferenceServerClient(url="0.0.0.0:8600")


def predict_linear(values):
    """
    Send a prediction request to Triton Inference Server
    """
    triton_client = get_client()

    input_data = np.array(values, dtype=np.float32)

    inputs = []
    inputs.append(
        InferInput(
            name="INPUT",
            shape=input_data.shape,
            datatype=np_to_triton_dtype(input_data.dtype),
        )
    )

    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(InferRequestedOutput("OUTPUT"))

    response = triton_client.infer(
        model_name="sklearn_linear", inputs=inputs, outputs=outputs
    )

    output_data = response.as_numpy("OUTPUT")

    return output_data


def main():
    test_values = np.array([[6], [7], [8], [9], [10]], dtype=np.float32)

    print("Input values:", test_values.flatten())

    predictions = predict_linear(test_values)

    print("Predictions:", predictions.flatten())

    expected = test_values * 2
    print("Expected values:", expected.flatten())

    mse = np.mean((predictions - expected.flatten()) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")


if __name__ == "__main__":
    main()
