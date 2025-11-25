import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image

app = FastAPI()
session = ort.InferenceSession("rps_ai.onnx")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = np.array(img).transpose(2,0,1) / 255.0
    img = img[None].astype(np.float32)
    outputs = session.run(None, {"input": img})
    return {"detections": outputs[0].tolist()}
