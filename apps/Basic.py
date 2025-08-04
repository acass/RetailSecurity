import cv2
from ultralytics import YOLO
import time

# Load pretrained YOLOv8 model
model = YOLO("../models/yolov8n.pt")  # Replace with custom model if needed

# Open video stream (0 = webcam, or provide IP cam URL)
# cap = cv2.VideoCapture("rtsp://172.16.225.166:8554/stream")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("Retail Surveillance", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
