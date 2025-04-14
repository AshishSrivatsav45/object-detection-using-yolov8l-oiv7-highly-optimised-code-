# Object Detection using YOLOv8l-OIV7 with ESP32-CAM

This project implements a real-time object detection system that streams video from an ESP32-CAM, detects objects using a YOLOv8 model trained on OpenImages v7, and aggregates & debounces detection announcements. Detected objects (with spatial info such as left, center, or right) are sent to a local Flask server, which can then trigger audio feedback on connected devices.

## Features

- **Real-Time Object Detection:** Uses YOLOv8l-OIV7 for high-accuracy detections on a live video stream.
- **Optimized & Highly Configurable:** Frame skipping and a debouncing mechanism prevent false repeats and reduce processing load.
- **Aggregation of Detections:** Multiple detections are aggregated (e.g., "3 person on the left") to avoid overwhelming output.
- **Server Communication:** Integrates with a Flask server for remote display or audio announcements.
- **Easy to Customize:** Adjust detection frequency, debounce interval, or ignored labels as needed.

## Requirements

- Python 3.8 or higher
- [OpenCV](https://opencv.org/) for video processing
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [Requests](https://docs.python-requests.org/) for HTTP communication
- [Flask](https://flask.palletsprojects.com/) for the local server

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository
