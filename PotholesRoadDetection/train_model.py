from ultralytics import YOLO

model = YOLO('models_yolo\yolov8m-seg.pt')

if __name__ == '__main__': 
   # Training.
   results = model.train(
      data='dataset\data.yaml', 
      imgsz=1056,
      epochs=40,
      batch=4,
      name='yolov8m_seg(1056_40_8)',
      device=0)
      