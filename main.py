import os
from models import load_yolo_model, load_feature_model
from detector import Detector
from tracker import Tracker
from analyzer import TrackAnalyzer
from video_processor import VideoProcessor

# Set device
device = "cpu"

# Load models
yolo_model = load_yolo_model("yolov8m.pt")
feature_model = load_feature_model('./models/osnet_x1_0_same.pth')

# Initialize components
detector = Detector(yolo_model, feature_model, device)
tracker = Tracker()
analyzer = TrackAnalyzer()

# Set up video processor
video_path = "2.mp4"
output_folder = os.path.expanduser("./Output_8")
os.makedirs(output_folder, exist_ok=True)
video_processor = VideoProcessor(video_path, output_folder, detector, tracker, analyzer)

# Process video
video_processor.process()

# Generate reports
video_processor.generate_reports()

print(f"\nProcessing complete!")
print(f"Video saved to: {os.path.join(output_folder, 'output-behavior-analysis.mp4')}")
print(f"Behavior report saved to: {os.path.join(output_folder, 'behavior_report.txt')}")
print(f"Interaction plots saved in: {os.path.join(output_folder, 'interaction_plots')}")