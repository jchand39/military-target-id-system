# live_detect.py
from ultralytics import YOLO

def main():
    print("🔍 Initializing Military Aircraft Detection...")
    model = YOLO("yolov12/runs/detect/mil-aircraft-detect6/weights/best.pt")
    
    # Real-time detection
    results = model(source=0, conf=0.25, show=True, stream=True)
    
    for r in results:
        print([model.names[int(cls)] for cls in r.boxes.cls])  # Print class names of detections

if __name__ == "__main__":
    main()


