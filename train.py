# Setup to train
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8 model
model = YOLO('yolov8x.pt')

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="dataset_biensoxe_all/data.yaml", epochs=50, imgsz=640, device=0)