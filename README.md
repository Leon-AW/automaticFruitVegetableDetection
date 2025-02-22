# Automatic Fruit & Vegetable Detection

This project provides a real‑time object detection system that captures video from your webcam, processes each frame using the YOLOv8 model, and displays annotated detections with bounding boxes and labels. This updated version now only detects and displays objects that belong to specific classes corresponding to fruits and vegetables from the COCO dataset. The detected classes are:

- **Fruits:**
  - Banana
  - Apple
  - Orange
  
- **Vegetables:**
  - Broccoli
  - Carrot

> **Note:**  
> The YOLOv8 models are trained on the COCO dataset. The detected fruit and vegetable classes are a subset of the available classes. For a more customized application, you can modify the script further to include or exclude classes as needed.

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
  Captures your webcam feed in real‑time and annotates detections with bounding boxes and labels.

- **Targeted Detection:**  
  This updated implementation restricts detections to specific classes corresponding to fruits and vegetables. Detected objects include:
  - **Fruits:** Banana, Apple, Orange
  - **Vegetables:** Broccoli, Carrot

- **Model Upgrade:**  
  Utilizes the YOLOv8s model, ensuring a balance between speed and accuracy.

---

## Prerequisites

- **Python:** Version 3.7 or later.
- **Operating System:** Windows, macOS, or Linux.
- **Webcam:** A functioning camera is required.
- **Hardware:** A GPU is advantageous for optimal performance, though the CPU will suffice for near real‑time detection on many laptops.

---

## Installation

### 1. Clone the Repository

Clone the project repository to your local machine and change to the project directory:

```bash
git clone <repository-url>
cd automaticFruitVegetableDetection
```

Replace `<repository-url>` with the actual URL of your Git repository.

### 2. Create a Virtual Environment

#### For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows:

Open Command Prompt or PowerShell and run:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- **ultralytics:** Provides pre‑trained YOLOv8 models.
- **opencv-python:** Captures and displays the webcam feed.

---

## Project Structure

- `fruit_detection.py`  
  The main script that captures the webcam feed, performs real‑time detection using the YOLOv8s model, and displays annotated detections only for the specified fruits and vegetables (Banana, Apple, Orange, Broccoli, and Carrot).

- `requirements.txt`  
  Contains the list of dependencies.

- `.gitignore`  
  Specifies files and folders that should be ignored by Git.

- `README.md`  
  This file, providing detailed instructions for installation and usage.

---

## Running the Project

After completing the installation steps, run the detection script with:

```bash
python fruit_detection.py
```

### What to Expect:

1. **Model Loading:**  
   The script loads the YOLOv8s model.

2. **Webcam Initialization:**  
   Your webcam feed is captured and displayed in a new window.

3. **Detection & Annotation:**  
   The model processes each frame and only detects objects corresponding to the selected classes:
   - **Fruits:** Banana, Apple, Orange
   - **Vegetables:** Broccoli, Carrot
   
4. **Exit:**  
   Press `q` while the window is in focus to close the application.

---

## Troubleshooting

### Webcam Issues

- **Camera Permissions:**  
  - **macOS:** Ensure that your Terminal or your preferred IDE has permission to access the camera. Go to **System Preferences** → **Security & Privacy** → **Privacy** → **Camera**, and verify that your application is allowed.
  - **Windows:** If you experience issues, try running your command prompt or PowerShell with administrator rights.

- **Testing Your Webcam:**  
  Run a minimal OpenCV script to confirm that your webcam is working correctly.

### Windows-Specific Notes

- **Environment Activation:**  
  Use the command `venv\Scripts\activate` to activate your virtual environment.
- **Dependency Installation:**  
  Ensure that your `pip` is up-to-date by running:
  ```bash
  python -m pip install --upgrade pip
  ```
- **Administrator Privileges:**  
  If you encounter permission-related errors, consider launching your command prompt or PowerShell as an administrator.

---

## Advanced Usage

- **Custom Detection Focus:**  
  The detection filtering is implemented in `fruit_detection.py` by passing specific class indices to the model. To modify which classes are detected, adjust the list of allowed indices in the script.

- **Exploring Alternative Models:**  
  For enhanced performance or accuracy, experiment with other YOLO variants such as YOLOv8m, YOLOv9, or YOLOv10, provided your hardware can support them.

---

## License

MIT License

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – The library for state‑of‑the‑art object detection models.
- [OpenCV](https://opencv.org/) – The library used for real‑time image processing.

For questions or contributions, please open an issue or submit a pull request.