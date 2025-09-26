from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Send to TensorFlow Serving
    data = json.dumps({"instances": img_array.tolist()})
    headers = {"content-type": "application/json"}
    response = requests.post("http://localhost:8501/v1/models/my_model:predict", data=data, headers=headers)

    result = response.json()
    return {
        "predictions": result["predictions"]
    }
