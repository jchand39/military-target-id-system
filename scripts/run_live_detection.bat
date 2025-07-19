@echo off
yolo task=detect mode=predict ^
  model=yolov12/runs/detect/mil-aircraft-detect6/weights/best.pt ^
  source=0 conf=0.25 show=True
pause
