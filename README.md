# 🎯 Computer Vision Based Crowd Scene Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-FFD700?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

A powerful, core machine learning pipeline designed to analyze crowd behaviors, detect individuals, and track movements in video streams using state-of-the-art Computer Vision algorithms. This repository forms the foundational ML engine that integrates seamlessly with the [Frontend](https://github.com/atharvaishere/Frontend_for_CV_Based_Crowd_Scene_Analysis) and [Backend](https://github.com/atharvaishere/Backend_CV_Analysis) APIs.

## ✨ Key Features

- **Real-Time Detection:** Utilizes Ultralytics YOLOv8 (Nano, Small, and Medium models supported) for high-speed, accurate human detection.
- **Robust Tracking:** Advanced tracking logic (`tracker.py`) to maintain object identities across consecutive frames.
- **Behavior Analytics:** Analyzes crowd density, flow, and structural patterns (`analyzer.py`).
- **Video Processing Pipeline:** Efficient frame-by-frame video extraction and processing (`video_processor.py`).
- **Containerized Environment:** Fully Dockerized setup for consistent execution across different platforms.

## 📂 Core Modules

- `main.py`: Entry point for executing the analysis pipeline.
- `detector.py`: Handles YOLO model initialization and bounding-box inferences.
- `tracker.py`: Implements object tracking logic for continuous ID assignment.
- `analyzer.py`: Generates analytics and insights based on the detection data.
- `video_processor.py`: Responsible for reading, manipulating, and outputting video streams.
- `models.py`: Pydantic/Data models representing internal structures.

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Deep Learning Framework:** Ultralytics YOLOv8
- **Computer Vision:** OpenCV (`cv2`)
- **Deployment:** Docker

## 🚀 Getting Started

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atharvaishere/Computer_Vision_Based_Crowd_Scene_Analysis.git
   cd Computer_Vision_Based_Crowd_Scene_Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python main.py
   ```
   *(Ensure you have input videos or camera streams configured as per your `main.py` setup)*

### Docker Setup

To run the application inside an isolated Docker container:

```bash
# Build the Docker image
docker build -t cv-crowd-analyzer .

# Run the container
docker run -p 8000:8000 cv-crowd-analyzer
```

---
*Developed by [Atharva Shrivastava](https://github.com/atharvaishere).*
