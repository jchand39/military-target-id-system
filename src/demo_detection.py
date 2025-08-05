from ultralytics import YOLO
import cv2
import sys
sys.path.insert(0, '/content/project/yolov12')

# Load your custom model
aircraft_model = YOLO("yolov12/runs/detect/mil-aircraft-detect6/weights/best.pt")

# Path to your input video
video_path = "demo.mp4"  # Make sure this file is in the same directory

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit(1)

# Optional: Save output video
save_output = True
output_path = "demo_output.mp4"
out = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

        text_y = y1 - 10 if position == "top" and y1 - 10 > 10 else y2 + 20
        text_y = min(max(10, text_y), frame.shape[0] - 10)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference with MIL model only
    results = aircraft_model.predict(source=frame, conf=0.25, verbose=False)

    # Draw results
    frame = draw_results(frame, results, label_prefix="[MIL] ", position="top")

    # Show and optionally save
    cv2.imshow("Military Aircraft Detection", frame)
    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

