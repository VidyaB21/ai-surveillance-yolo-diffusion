import requests
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------------
# HUGGING FACE API
# -------------------------------
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Accept": "image/png"
}

def generate_image(prompt):
    payload = {
        "inputs": prompt
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)


    if response.status_code == 200:
        with open("outputs/generated_scene.png", "wb") as f:
            f.write(response.content)
        print("✅ Image generated successfully!")
    else:
        print("❌ Error Code:", response.status_code)
        print(response.text)

# -------------------------------
# LOAD YOLO
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# VIDEO INPUT
# -------------------------------
cap = cv2.VideoCapture("C:/Users/Vidya Bag/Videos/food video row.mp4")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 60th frame (reduce API calls)
    if frame_count % 60 != 0:
        continue

    results = model(frame)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append(label)

    if len(detected_objects) == 0:
        continue

    # -------------------------------
    # GENERATE PROMPT
    # -------------------------------
    objects_text = ", ".join(set(detected_objects))

    prompt = f"A CCTV scene showing {objects_text}, realistic, night vision, surveillance style"

    print("Generated Prompt:", prompt)

    # -------------------------------
    # GENERATE IMAGE
    # -------------------------------
    generate_image(prompt)

cap.release()