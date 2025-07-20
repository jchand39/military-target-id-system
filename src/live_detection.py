from ultralytics import YOLO
import cv2

# Load your models
aircraft_model = YOLO("yolov12/runs/detect/mil-aircraft-detect6/weights/best.pt")
coco_model = YOLO("yolov12n.pt")  # Or yolov12s.pt

# Open webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "❌ Could not open webcam"

print("🎯 Dual-model detection running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run both models
    aircraft_results = aircraft_model(frame, conf=0.5, verbose=False)[0]
    coco_results = coco_model(frame, conf=0.5, verbose=False)[0]

    # Clone frame for drawing
    annotated_frame = frame.copy()

    # Draw aircraft detections
    if aircraft_results.boxes is not None:
        for box in aircraft_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = aircraft_model.names[cls]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"[MIL] {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw COCO detections
    if coco_results.boxes is not None:
        for box in coco_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = coco_model.names[cls]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"[CIV] {label}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show annotated frame
    cv2.imshow("YOLOv12 Dual Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



