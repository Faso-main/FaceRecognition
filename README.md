# YOLO Video Processing Application

This application utilizes the YOLO (You Only Look Once) object detection model to process video feeds in real time. It can be used to detect various objects and query a database for further actions based on the detected items.

## Features

- Real-time object detection using YOLO.
- Configurable model parameters including video source and confidence threshold.
- Database querying to check for detected items against a local list.
- User-friendly output displaying detected items and bounding boxes.

## Requirements

- Python 3.6+
- Dependencies:
  - [`opencv-python`](https://pypi.org/project/opencv-python/)
  - [`numpy`](https://numpy.org/)
  - [`ultralytics`](https://pypi.org/project/ultralytics/) (for YOLO)
  - Any custom module you use for database interaction (e.g., `db.main`)

You can install the required packages with:
```bash
pip install opencv-python numpy ultralytics
