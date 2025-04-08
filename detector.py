import cv2
import numpy as np
import torch
from torchvision import transforms
import os

class Detector:
    def __init__(self, yolo_model, feature_model, device):
        """Initialize the Detector with YOLO and feature models."""
        self.yolo_model = yolo_model
        self.feature_model = feature_model
        self.device = device
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def detect(self, frame):
        """Detect people in the frame and extract features."""
        results = self.yolo_model(frame, classes=[0], conf=0.3, iou=0.4)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                features = self.get_features(roi)
                if features is not None:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], features))
        return detections

    def get_features(self, roi):
        """Extract features from a region of interest (ROI)."""
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_LANCZOS4)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = torch.from_numpy(roi).float().permute(2, 0, 1) / 255.0
        roi = self.normalize(roi).unsqueeze(0)
        with torch.no_grad():
            return self.feature_model(roi).squeeze().cpu().numpy()