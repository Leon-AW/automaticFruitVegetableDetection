import cv2
from ultralytics import YOLO
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import yaml
import numpy as np

def load_efficientnet(model_path, num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Modified loading mechanism
    try:
        # Try loading with map_location
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            # Alternative loading method
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    model.eval()
    return model

def main():
    # Parse command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["yolov8s", "trained"], default="yolov8s",
                        help="Choose the model to use: 'yolov8s' or 'trained'")
    args = parser.parse_args()

    # Load class names from YAML for the trained model
    if args.model_type == "trained":
        with open("classes.yaml", "r") as f:
            classes_data = yaml.safe_load(f)
        all_classes = list(classes_data["fruit_and_vegetables"].keys())
        num_classes = len(all_classes)
        
        # Load and prepare the EfficientNet model
        model = load_efficientnet("fine_tuned_efficientnet_b3.pth", num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Load YOLO model
        model = YOLO("ultralytics/yolov8s.pt")
        allowed_class_indices = [46, 47, 49, 50, 51]

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.model_type == "trained":
            # Process frame for EfficientNet
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Transform the image
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = all_classes[predicted.item()]
                conf_value = confidence.item()

            # Draw results on frame
            if conf_value > 0.5:  # Confidence threshold
                h, w = frame.shape[:2]
                # Draw black rectangle at bottom
                cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
                # Draw text
                text = f"Detected: {predicted_class} ({conf_value:.2f})"
                cv2.putText(frame, text, (10, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            annotated_frame = frame

        else:
            # Process frame for YOLO
            results = model(frame, classes=allowed_class_indices)
            annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("Fruit and Vegetable Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 