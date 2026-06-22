import torch
import insightface
from ultralytics import YOLO
import mediapipe as mp

# Test GPU
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Test InsightFace
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640,640))
print("InsightFace ready")

# Test YOLO
yolo = YOLO("yolov8n.pt")
print("YOLO ready")