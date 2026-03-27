# CIFAR-10 Image Classifier API (FastAPI + Docker)

FastAPI backend with a pretrained ResNet18 model for CIFAR-10 image classification. Containerized with Docker for easy deployment.

## Features
- FastAPI endpoint `/predict` for image upload and classification
- Uses the model trained in Task 2 (`cifar10_resnet18.pth`)
- Data preprocessing (resize + normalization)
- Returns predicted class + confidence score
- Docker support for local and cloud deployment

## Files
- `app.py` → FastAPI application
- `requirements.txt` → Python dependencies
- `Dockerfile` → Docker container instructions

## How to Run Locally

### Option 1: Without Docker (FastAPI only)
1. Clone the repo:
   ```bash
   git clone https://github.com/janardankaki1125/cifar10-fastapi-docker.git
   cd cifar10-fastapi-docker
   pip install -r requirements.txt
   uvicorn app:app --reload
   
