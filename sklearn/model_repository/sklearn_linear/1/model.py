import os

import joblib
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "model.joblib"
        )
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("Loaded sklearn model successfully")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()

            # Reshape if necessary (ensuring 2D array for sklearn)
            if input_data.ndim == 1:
                input_data = input_data.reshape(-1, 1)

            # Make prediction
            y_pred = self.model.predict(input_data)

            # Convert prediction to Triton's expected format
            output_tensor = pb_utils.Tensor("OUTPUT", y_pred.astype(np.float32))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
