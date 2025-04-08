import torch
from ultralytics import YOLO
import torchreid
import os

def load_yolo_model(model_path):
    """Load the YOLO model from the specified path."""
    return YOLO(model_path)

def load_feature_model(model_path, num_classes=751):
    """Load the feature extraction model and set it to evaluation mode."""
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=num_classes)
    torchreid.utils.load_pretrained_weights(model, model_path)
    model.eval()
    return model