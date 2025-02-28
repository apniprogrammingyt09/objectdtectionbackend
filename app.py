from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model (Ensure 12x.pt exists)
model_path = "12x.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please place it in the project directory.")

model = YOLO(model_path)


def process_frame(frame):
    results = model(frame)
    predictions = []
    object_count = {}

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            predictions.append({
                "class": class_name,
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

            # Count objects
            object_count[class_name] = object_count.get(class_name, 0) + 1

    return predictions, object_count


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        await websocket.send_json({"error": "Could not open webcam"})
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predictions, object_count = process_frame(frame)

            # Draw bounding boxes on the frame
            for pred in predictions:
                x1, y1, x2, y2 = map(int, pred["bbox"])
                label = f"{pred['class']} ({pred['confidence']:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame to send over WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send frame and object count to frontend
            await websocket.send_json({"frame": frame_base64, "object_count": object_count})

    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        cap.release()
        await websocket.close()


@app.get("/")
def home():
    return {"message": "Real-Time Object Detection API using 12x.pt"}
