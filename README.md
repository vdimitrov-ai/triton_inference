# NVIDIA Triton Inference Server Test Repository

This repository was created to demonstrate the capabilities of NVIDIA Triton Inference Server - a platform for deploying and serving machine learning models.

## Repository Contents

The repository presents examples of using Triton Server with different types of models:

### [Huggingface](./huggingface)

An example of deploying a transformer model (RuBERT) using Triton Server. Includes model export to ONNX format, conversion to TensorRT, and a client example.

Detailed description and instructions are available in the [huggingface folder README.md](./huggingface/README.md).

### [Sklearn](./sklearn)

An example of deploying a scikit-learn linear regression model using Triton Server. Demonstrates a simple workflow for training, saving, and serving the model.

Detailed description and instructions are available in the [sklearn folder README.md](./sklearn/README.md).

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for some examples)
- NVIDIA Container Toolkit (nvidia-docker)

## Purpose

This repository is intended to introduce the capabilities of Triton Server and demonstrate the basic principles of its operation with various machine learning frameworks.
