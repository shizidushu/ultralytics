from ultralytics import settings
from ultralytics import YOLO
import torch
import os

current_folder = os.path.dirname(__file__)
datasets_dir = '/mnt/d/0-Datasets/ultralytics_datasets_dir/datasets'
weights_dir = os.path.join(current_folder, "weights")
runs_dir = os.path.join(current_folder, "runs")

settings.update({'datasets_dir': datasets_dir})
settings.update({'weights_dir': weights_dir})
settings.update({'runs_dir': runs_dir})
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

resume = False
if not resume:
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
else:
    model = YOLO('runs/detect/train/weights/last.pt')

device_count = torch.cuda.device_count()
model.train(data='coco.yaml', imgsz=640, epochs=200, resume=resume, batch = 32 * device_count, device=list(range(0,device_count)))
