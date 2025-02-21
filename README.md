# Automatic Fruit & Vegetable Detection

This project provides a real‑time object detection system that identifies fruits and vegetables from your webcam feed. It leverages the power of the YOLOv8 model (specifically the YOLOv8s variant) from the [Ultralytics](https://github.com/ultralytics/ultralytics) library combined with [OpenCV](https://opencv.org/) to capture video from your camera, process each frame, and display only produce detections (e.g., apple, banana, orange, broccoli, carrot).

> **Note:**  
> The default YOLOv8 models are trained on the COCO dataset, which only includes a limited set of fruit and vegetable classes. To detect a broader range of produce, you might consider fine‑tuning a custom model on a dedicated fruit and vegetable dataset.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Real‑Time Detection:**  
  Processes your webcam feed in real‑time and annotates detections with bounding boxes and labels.

- **Model Upgrade:**  
  Uses the YOLOv8s model for improved accuracy compared to the nano model, while still maintaining near real‑time performance on typical laptop hardware.

- **Selective Detection:**  
  The application processes detections from the full model output and then filters results to display only fruits and vegetables (currently, apple, banana, orange, broccoli, and carrot).

---

## Prerequisites

- **Python:** Version 3.7 or later is recommended.
- **Operating System:** Windows, macOS, or Linux.
- **Webcam:** A functional camera is required for capturing real‑time video.
- **Hardware:** For optimal performance with YOLOv8s, a GPU is advantageous—but many laptops can still process the stream in near real‑time using CPU.

---

## Installation

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <repository-url>
cd automaticFruitVegetableDetection
```

### 2. Create a Virtual Environment

Isolate all dependencies by creating and activating a virtual environment.

- **On macOS/Linux:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies

With the virtual environment activated, install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- **ultralytics:** Provides pre‑trained YOLOv8 models.
- **opencv-python:** Captures webcam video and handles real‑time frame display.

---

## Project Structure

A quick overview of the repository files:

- `fruit_detection.py`  
  The main script which captures the webcam feed, performs YOLOv8 real‑time fruit and vegetable detection, and displays the annotated results.

- `requirements.txt`  
  Lists all dependencies for the project.

- `.gitignore`  
  Contains rules for files and folders (like virtual environments, caches, and editor-specific settings) that Git should ignore.

- `README.md`  
  This file, explaining how to install, run, and troubleshoot the project.

---

## Running the Project

After you have set up the environment and installed all dependencies, run the detection script:

```bash
python fruit_detection.py
```

### What to Expect:

1. **Model Loading:**  
   The script loads the YOLOv8s model for detection.

2. **Webcam Initialization:**  
   Your webcam feed will be captured and a window will open displaying the live feed.

3. **Detection & Filtering:**  
   The model processes each frame. It performs full inference and then filters the detections to only highlight fruits and vegetables (currently, apple, banana, orange, broccoli, and carrot).

4. **Real‑Time Annotated Feed:**  
   Bounding boxes and labels (with confidence scores) are drawn on the frame for each valid detection.

5. **Exit:**  
   Press `q` while the window is in focus to close the application.

---

## Troubleshooting

### Webcam Not Displaying or Errors

- **Camera Permissions:**  
  On macOS, ensure that your Terminal or IDE is allowed to access the camera via **System Preferences** → **Security & Privacy** → **Privacy** → **Camera**.

- **Test Webcam Independently:**  
  Run a minimal OpenCV webcam test:

  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
      print("Cannot open camera")
      exit()
  while True:
      ret, frame = cap.read()
      if not ret:
          print("Can't receive frame. Exiting...")
          break
      cv2.imshow("Camera Test", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()
  ```

- **Dependencies:**  
  If issues arise during installation, ensure `pip` is up-to-date:

  ```bash
  pip install --upgrade pip
  ```

### Performance Considerations

- **Model Choice:**  
  YOLOv8s is used for its balance between speed and accuracy. If you have a powerful GPU and need even better detection accuracy, consider modifying the script to use YOLOv8m.
- **Hardware:**  
  On devices without a dedicated GPU, inference might be slightly slower. Adjust the resolution of the webcam feed (using OpenCV's `cv2.resize`) if necessary.

---

## Advanced Usage

- **Custom Detection Classes:**  
  The current script filters detections to only include a selected set of fruits and vegetables. For additional classes, you would need to either modify the allowed list or train/fine‑tune a custom model on a more comprehensive dataset.

- **Exploring Newer YOLO Models:**  
  Recent models (e.g., YOLOv9 and YOLOv10) offer increased accuracy at the expense of increased computational requirements. Experiment with different versions if your hardware allows.

---

## License

*(Include license information if applicable.)*

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – The library for state‑of‑the‑art object detection models.
- [OpenCV](https://opencv.org/) – The library used for real-time image processing.

For questions or to report issues, please open an issue in the repository.