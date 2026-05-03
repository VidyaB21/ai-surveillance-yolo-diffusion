from ultralytics import YOLO
import cv2
from diffusers import StableDiffusionPipeline
import torch
import os
import sys


def run_pipeline(video_path):
    print("▶ Using video:", video_path)

    # -------------------------------
    # LOAD MODELS
    # -------------------------------
    yolo_model = YOLO("yolov8n.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # -------------------------------
    # VIDEO INPUT
    # -------------------------------
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Cannot open video")
        return

    # -------------------------------
    # OUTPUT SETUP
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    object_positions = {}
    frame_count = 0

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_count += 1

        results = yolo_model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        detected_objects = []
        activity = "normal"

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy
            ids = results[0].boxes.id
            classes = results[0].boxes.cls

            for box, obj_id, cls in zip(boxes, ids, classes):
                obj_id = int(obj_id)
                cls = int(cls)
                label = yolo_model.names[cls]

                detected_objects.append(label)

                if label == "person":
                    x1, y1, x2, y2 = map(int, box)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if obj_id in object_positions:
                        px, py = object_positions[obj_id]
                        speed = abs(cx - px) + abs(cy - py)

                        if speed > 25:
                            activity = "running (suspicious)"
                            cv2.putText(
                                annotated_frame,
                                "SUSPICIOUS ACTIVITY",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3
                            )

                    object_positions[obj_id] = (cx, cy)

        # -------------------------------
        # GENERATE IMAGE EVERY 150 FRAMES
        # -------------------------------
        if frame_count % 150 == 0 and detected_objects:
            objects_text = ", ".join(set(detected_objects))

            prompt = f"A CCTV footage showing {objects_text}, {activity}, low light, security camera view"

            print("\n🧠 Generating scene...")
            print(prompt)

            image = pipe(prompt, num_inference_steps=20).images[0]

            filename = f"outputs/generated_{frame_count}.png"
            image.save(filename)

            print("✅ Saved:", filename)

            with open("outputs/log.txt", "a") as f:
                f.write(f"{frame_count}: {objects_text} | {activity}\n")

        cv2.imshow("AI Surveillance", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/default.mp4"

    run_pipeline(video_path)