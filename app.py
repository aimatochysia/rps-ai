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

CLASS_NAMES = ["rock", "paper", "scissors"]
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640

input_name = session.get_inputs()[0].name


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    outputs = session.run(None, {input_name: img_array})

    predictions = outputs[0][0]

    predictions = predictions.T

    detections = []

    for pred in predictions:
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]

        class_id = np.argmax(class_scores)
        confidence = float(class_scores[class_id])

        if confidence >= CONFIDENCE_THRESHOLD:
            x1 = float(x_center - width / 2)
            y1 = float(y_center - height / 2)
            x2 = float(x_center + width / 2)
            y2 = float(y_center + height / 2)

            detections.append({
                "class": CLASS_NAMES[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:2]
    return {"detections": detections}
