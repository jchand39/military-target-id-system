from ultralytics import YOLO
import cv2

# Load models
aircraft_model = YOLO("yolov12/runs/detect/mil-aircraft-detect6/weights/best.pt")
coco_model = YOLO("yolov12n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

def draw_results(frame, results, label_prefix="", position="top"):
    boxes = results[0].boxes
    names = results[0].names

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{label_prefix}{names[cls_id]} {conf:.2f}"

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        color = (int(cls_id * 15) % 255, int(cls_id * 50) % 255, int(cls_id * 85) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Position the label
        if position == "top":
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        else:  # "bottom"
            text_y = y2 + 20 if y2 + 20 < frame.shape[0] - 10 else y2 - 10

        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results_aircraft = aircraft_model.predict(source=frame, conf=0.25, verbose=False)
    results_coco = coco_model.predict(source=frame, conf=0.25, verbose=False)

    # Draw military boxes on top, COCO boxes below
    frame = draw_results(frame, results_aircraft, label_prefix="[MIL] ", position="top")
    frame = draw_results(frame, results_coco, label_prefix="[CIV] ", position="bottom")

    # Show frame
    cv2.imshow("Military + COCO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




