from ultralytics import YOLO

def main():
    model = YOLO("yolov12n.pt")
    model.train(
        data="datasets/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="mil-aircraft-detect"
    )

    trained_model = YOLO("yolov12/runs/detect/mil-aircraft-detect/weights/best.pt")
    results = trained_model(source=0, show=True, conf=0.6, save=True)

if __name__ == "__main__":
    main()
