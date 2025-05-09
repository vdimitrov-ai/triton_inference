# Scikit-Learn Model with Triton Server

This project demonstrates how to deploy a scikit-learn LinearRegression model using NVIDIA Triton Inference Server.

## Requirements

- Docker
- Docker Compose
- NVIDIA GPU with CUDA support (optional for this simple model)
- NVIDIA Container Toolkit (nvidia-docker)

## Environment Setup

1. Create a Python virtual environment for the client:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training and Exporting the Model

Run the export script to train a simple LinearRegression model and prepare the files for Triton:

```bash
python export_model.py
```

This script:
1. Trains a linear regression model (y = 2x) on sample data
2. Saves the model to `model_repository/sklearn_linear/1/model.joblib`
3. Creates model configuration file and Python backend script

## Running Triton Server

Run Triton Server using Docker Compose:

```bash
docker-compose up --build
```

This command will build the Docker image and start the Triton Server with all the necessary configurations.

## Testing the Model

### Direct Testing (without Triton)

To test the model directly, bypassing Triton:

```bash
source .venv/bin/activate
python run_model.py
```

### Inference Using the Triton Client

After starting the Triton Server, you can use the client to send requests:

```bash
source .venv/bin/activate
python client.py
```

The client will send sample values to the server and display the predictions.

## Project Structure

- `model_repository/` - directory with models for Triton
  - `sklearn_linear/` - sklearn model directory
    - `config.pbtxt` - model configuration for Triton
    - `/1/` - model version directory
      - `model.joblib` - serialized scikit-learn model
      - `model.py` - Python backend for Triton
- `client.py` - Python client for sending requests to Triton
- `export_model.py` - script for training and exporting the model
- `run_model.py` - script for direct model inference (without Triton)
- `Dockerfile` - file for building the Triton Server Docker image
- `docker-compose.yaml` - Docker Compose configuration file
- `requirements.txt` - Python dependencies

## Troubleshooting

1. Error "No module named 'sklearn'" - make sure scikit-learn is installed in the Docker image.

2. Error "No module named 'geventhttpclient'" - install tritonclient with HTTP support:
   ```bash
   pip install tritonclient[http] geventhttpclient
   ```

3. If client fails to connect to Triton server, check that the server is running correctly with:
   ```bash
   docker ps
   ``` 