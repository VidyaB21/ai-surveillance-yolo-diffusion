from ultralytics import YOLO
import cv2


# LOAD MODEL

model = YOLO("yolov8n.pt")

# VIDEO INPUT (CHANGE PATH IF NEEDED)

video_path = "C:/Users/Vidya Bag/Videos/food video row.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()


# GET VIDEO PROPERTIES (IMPORTANT)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VIDEO WRITER (SAVE OUTPUT)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("outputs/tracking_output.avi", fourcc, fps, (frame_width, frame_height))


# STORE OBJECT POSITIONS

object_positions = {}


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()

    # Check if detections exist
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        for box, obj_id, cls in zip(boxes, ids, classes):

            obj_id = int(obj_id)
            cls = int(cls)
            label = model.names[cls]

         
            # FILTER ONLY PERSON
           
            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, box)

            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

          
            # MOVEMENT DETECTION
           
            if obj_id in object_positions:
                prev_x, prev_y = object_positions[obj_id]

                dx = center_x - prev_x
                dy = center_y - prev_y

                speed = abs(dx) + abs(dy)

               
                # SPEED ALERT
              
                if speed > 25:
                    print(f"⚠️ Person {obj_id} moving fast!")

                    # Draw alert text
                    cv2.putText(
                        annotated_frame,
                        f"FAST ID {obj_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            # Save current position
            object_positions[obj_id] = (center_x, center_y)

    # SAVE OUTPUT VIDEO
  
    out.write(annotated_frame)

  
    # DISPLAY WINDOW

    cv2.imshow("AI Surveillance - Tracking", annotated_frame)

    # Exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


# RELEASE EVERYTHING

cap.release()
out.release()
cv2.destroyAllWindows()