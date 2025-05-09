# Huggingface Transformer with Triton Server

This project demonstrates how to run a pre-trained RuBERT model using NVIDIA Triton Inference Server.

## Requirements

- Docker
- Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (nvidia-docker)

## Environment Setup

1. Create a Python virtual environment for the client:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_dev.txt
```

2. Make sure all necessary tokenizer files, including `config.json`, are in the `assets/rubert-tokenizer` directory.

## Export Model (optional)

If you need to export the model from Huggingface to ONNX format:

```bash
python export_model.py
```

## Convert Model to TensorRT (optional)

To improve inference performance, you can convert the ONNX model to TensorRT:

```bash
docker run -it --rm --gpus all -v $(pwd):/models \
nvcr.io/nvidia/tensorrt:24.03-py3 \
trtexec --onnx=/models/model_repository/onnx-rubert/1/model.onnx \
        --saveEngine=/models/model_repository/trt-fp16-rubert/1/model.plan \
        --minShapes=INPUT_IDS:1x16,ATTENTION_MASK:1x16 \
        --optShapes=INPUT_IDS:8x16,ATTENTION_MASK:8x16 \
        --maxShapes=INPUT_IDS:16x16,ATTENTION_MASK:16x16 \
        --profilingVerbosity=detailed \
        --builderOptimizationLevel=3
```

## Running Triton Server

Run Triton Server using Docker Compose:

```bash
docker-compose up --build
```

This command will build the Docker image and start the Triton Server with all the necessary configurations.

## Inference Using the Client

After starting the Triton Server, you can use the client to send requests:

```bash
source .venv/bin/activate
python client.py
```

The client will send several text examples to the server and output a tensor of distances between the embeddings of these texts.

## Project Structure

- `model_repository/` - directory with models for Triton
  - `onnx-rubert/` - ONNX version of the model
  - `trt-fp16-rubert/` - TensorRT version of the model
  - `python-tokenizer/` - Python backend for tokenization
  - `ensemble-onnx/` - ensemble combining tokenizer and model
- `assets/` - resources for models
  - `rubert-tokenizer/` - tokenizer files
- `client.py` - Python client for sending requests to Triton
- `export_model.py` - script for exporting the model to ONNX format
- `Dockerfile` - file for building the Triton Server Docker image
- `docker-compose.yaml` - Docker Compose configuration file
- `requirements.txt` - Python dependencies

## Troubleshooting

1. Error "No module named 'transformers'" - make sure the transformers library is installed in the Docker image.

2. Error "No module named 'geventhttpclient'" - install tritonclient with HTTP support:
   ```bash
   pip install tritonclient[http] geventhttpclient
   ```

3. Error "engine plan file is not compatible with this version of TensorRT" - regenerate the TensorRT model with a version compatible with your Triton Server. 