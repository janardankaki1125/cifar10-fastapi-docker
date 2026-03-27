# CIFAR-10 Image Classifier API (FastAPI + Docker)

FastAPI backend with ResNet18 model for CIFAR-10 image classification. Containerized using Docker.

## Features
- `/predict` endpoint to upload an image and get prediction
- Uses pretrained ResNet18 with transfer learning (from Task 2)
- Returns class name + confidence score
- Docker support

## Files
- `app.py` → FastAPI code
- `requirements.txt`
- `Dockerfile`

## Important: Model File
The trained model `cifar10_resnet18.pth` is **not** included in this repo because of size limit.

**How to get the model:**
1. Open this notebook in Google Colab:
   https://github.com/janardankaki1125/deep-learning-image-classifier/blob/main/deep_learning_image_classifier.ipynb
2. Run all cells (it will train and save the model).
3. Download `cifar10_resnet18.pth` from the Files panel in Colab.
4. Place it in this folder before running the API or building Docker.

## How to Run

### Without Docker
```bash
git clone https://github.com/janardankaki1125/cifar10-fastapi-docker.git
cd cifar10-fastapi-docker
pip install -r requirements.txt
uvicorn app:app --reload
## With Docker
docker build -t cifar10-api .
docker run -p 8000:8000 cifar10-api
