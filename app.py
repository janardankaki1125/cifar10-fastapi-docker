from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
from PIL import Image
import uvicorn

app = FastAPI(title="CIFAR-10 Image Classifier API")

# Load the model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("cifar10_resnet18.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Class names for CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Image transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.get("/")
def home():
    return {"message": "CIFAR-10 Image Classifier API is running! Send image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        
        predicted_class = classes[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "status": "success"
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
