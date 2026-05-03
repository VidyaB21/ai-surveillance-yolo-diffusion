from ultralytics import YOLO
import cv2

# Load YOLO model (auto downloads)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(r"C:\Users\Vidya Bag\Videos\food video row.mp4")

# Check camera
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Create video writer (save output)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("outputs/output.avi", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save output
    out.write(annotated_frame)

    # Show output window
    cv2.imshow("YOLO Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()