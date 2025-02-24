# Automatic Fruit & Vegetable Detection

This project provides a real‑time system for detecting fruits and vegetables from your webcam feed. It offers two detection strategies:

- **YOLOv8 Model:**  
  Uses the state‑of‑the‑art YOLOv8s model from Ultralytics. Trained on the COCO dataset, this model outputs bounding boxes and labels but is restricted to a subset of objects—specifically Banana, Apple, Orange, Broccoli, and Carrot.

- **Self-Trained Model:**  
  A custom fine‑tuned EfficientNet‑B3 model trained specifically for fruit and vegetable classification. The training was performed on a fruits dataset (downloaded via KaggleHub) using the `train_model.py` script. This model leverages transfer learning from ImageNet, replaces the classifier head with dropout, batch normalization, and a linear layer, and achieves very high accuracy (about 99.71% on the validation set). It is loaded from the file `fine_tuned_efficientnet_b3.pth` and performs a whole‑frame classification, overlaying the predicted class and its corresponding confidence score on the video stream.

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
- [Self‑Trained Model Details](#self‑trained-model-details)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Real‑Time Detection:** Captures your webcam feed and processes frames in real‑time.
- **Dual Model Options:**  
  Choose between the YOLOv8 detection model (with bounding box outputs) and our self‑trained EfficientNet‑B3 model (full frame classification).
- **Custom Detection Focus:**  
  - The YOLOv8 model is configured to detect only specific fruits and vegetables (Banana, Apple, Orange, Broccoli, and Carrot).  
  - The self‑trained model, meanwhile, is adapted to classify a broader range of fruit and vegetable classes.
- **State‑of‑the‑Art Models:**  
  Utilizes pre‑trained models and transfer learning to ensure high accuracy and performance.

---

## Prerequisites

- **Python:** Version 3.7 or later.
- **Operating System:** Windows, macOS, or Linux.
- **Webcam:** A functioning camera is required.
- **Hardware:** A GPU is recommended for optimal performance; however, the CPU can suffice for near real‑time detection.

---

## Installation

### 1. Clone the Repository

Clone the project repository and change to the project directory:

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

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

---

## Project Structure

- **fruit_detection.py**  
  The main script that captures the webcam feed and performs real‑time detection/classification. Use the command-line argument `--model_type` to choose between the YOLOv8 and self‑trained models.

- **train_model.py**  
  The training script for fine‑tuning the EfficientNet‑B3 model on a fruits dataset. It downloads data via KaggleHub, applies data augmentation, and trains with early stopping. The best model weights are saved as `fine_tuned_efficientnet_b3.pth`.

- **classes.yaml**  
  Contains the definitions for fruit and vegetable classes used during training and detection.

- **requirements.txt**  
  Lists the necessary dependencies.

- **.gitignore**  
  Specifies files and directories to be excluded from version control.

- **README.md**  
  This documentation file.

---

## Running the Project

After completing the installation steps, run the detection script. By default, the YOLOv8 model is used:

```bash
python fruit_detection.py
```

To use the self‑trained EfficientNet‑B3 model instead, run:

```bash
python fruit_detection.py --model_type trained
```

### What to Expect

1. **Model Loading:**  
   - **YOLOv8:** Loads the pre‑trained YOLOv8s model from Ultralytics.  
   - **Self‑Trained Model:** Loads the fine‑tuned EfficientNet‑B3 model from `fine_tuned_efficientnet_b3.pth`. (Ensure that Git LFS is properly configured so you receive the actual weight file, not a pointer.)

2. **Webcam Initialization:**  
   Initiates your webcam feed and processes frames in real‑time.

3. **Detection & Annotation:**  
   - **YOLOv8:** Outputs bounding boxes and labels for detected fruits and vegetables.
   - **Self‑Trained Model:** Performs image-level classification, overlaying the predicted class and its confidence score on each frame.

4. **Exit:**  
   Press `q` when the window is focused to close the application.

---

## Self‑Trained Model Details

The self‑trained model is based on EfficientNet‑B3 and has been fine‑tuned to classify fruit and vegetable images with high accuracy. Below are some key details:

- **Architecture:**  
  The model uses an EfficientNet‑B3 backbone (pre‑trained on ImageNet) with a custom classifier head comprising:
  - Dropout (0.3)
  - Batch Normalization
  - A Linear layer that maps features to the number of classes

- **Training Process:**  
  - **Data:** Images are downloaded using KaggleHub and organized into training and test splits. The final number of classes is determined by the dataset or the `classes.yaml` file.
  - **Augmentation:** Data augmentation techniques include random resized cropping, horizontal flipping, color jitter, and normalization.
  - **Optimization:** Training employs mixed precision (using torch.amp.autocast), AdamW optimizer, and a learning rate scheduler with early stopping.
  - **Performance:** The model achieved an accuracy of approximately 99.71% on the validation set.
  - **Checkpointing:** The best performing weights are saved in `fine_tuned_efficientnet_b3.pth`.

- **Usage:**  
  Launch the detector with the self‑trained model option using the `--model_type trained` flag. The model applies pre‑processing (resizing, normalization) on each frame before classifying and annotating it with the predicted class label and confidence score.

---

## Troubleshooting

### Webcam Issues

- **Camera Permissions:**  
  - **macOS:** Ensure that your Terminal or preferred IDE is allowed access to the camera. (System Preferences → Security & Privacy → Privacy → Camera)  
  - **Windows:** Run your command prompt or PowerShell as an administrator if permission issues arise.

- **Testing Your Webcam:**  
  Use a minimal OpenCV script to verify that your webcam is functioning correctly.

### Git LFS and Model Loading

If loading `fine_tuned_efficientnet_b3.pth` results in errors, verify that Git LFS is installed and properly configured in your environment. In your active shell, ensure the Git LFS binary is available in your PATH and run:

```bash
git lfs install
git lfs pull
```

This ensures that the weight file is downloaded in its binary form rather than as a pointer file.

---

## Advanced Usage

- **Custom Detection Focus:**  
  Edit `fruit_detection.py` to modify the allowed object class indices for the YOLOv8 model if you wish to detect different subsets of objects.

- **Model Experimentation:**  
  Feel free to retrain the self‑trained model with additional data or altered hyperparameters. Adjust the training script in `train_model.py` to experiment with performance improvements.

---

## License

MIT License

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – For providing pre‑trained object detection models.
- [OpenCV](https://opencv.org/) – For real‑time image processing.
- [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) – The backbone of our fine‑tuned model.
- [KaggleHub](https://github.com/martinarjovsky/kagglehub) – For simplified dataset downloads.