from fastapi import FastAPI
from contextlib import asynccontextmanager
from App.utils import load_model, preprocess_image
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch.nn.functional as F

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield



app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
)

@app.get('/')
def health_check():
    return "Health Check: Success"

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = preprocess_image(image_data)

    prediction = model(image)
    confidence , _ = F.softmax(prediction).max(dim=1)
    confidence = confidence.item() * 100
    prediction = prediction.argmax(dim=1).item()
    return {"Prediction" : {prediction}, "Confidence" : {confidence}}