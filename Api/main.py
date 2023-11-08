from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from io import BytesIO
from PIL import Image

import tensorflow as tf
import keras

MODEL: keras.models.Sequential = keras.models.load_model('../model_potato_disease.h5')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5500/test.html",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello world"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_arr = read_file_as_image(await file.read())

    pred = MODEL.predict(tf.expand_dims(img_arr, 0))

    confidence = np.max(pred[0])

    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    print({
         'class': pred_class,
         'confidence': confidence
     })
    return {
         'class': pred_class,
         'confidence': float(confidence)
     }


def read_file_as_image(bytes) -> np.array:
    img = Image.open(BytesIO(bytes))

    return np.array(img)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000, reload=False)
