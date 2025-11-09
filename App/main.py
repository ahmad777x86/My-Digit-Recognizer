from fastapi import FastAPI
from contextlib import asynccontextmanager
from App.utils import load_model, preprocess_image
from fastapi import UploadFile, File

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield


app = FastAPI(lifespan=lifespan)

@app.get('/')
def health_check():
    return "Health Check: Success"

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = preprocess_image(image_data)

    prediction = model(image)
    prediction = prediction.argmax(dim=1).item()
    return f"Prediction : {prediction}"