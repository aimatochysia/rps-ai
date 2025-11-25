import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("rps_ai.onnx")
input_name = session.get_inputs()[0].name

IMG_SIZE = 640 

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})

    return {"detections": outputs[0].tolist()}
