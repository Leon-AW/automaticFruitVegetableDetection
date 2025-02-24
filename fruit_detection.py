import cv2
from ultralytics import YOLO
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import os
from torchvision.datasets import ImageFolder
import kagglehub
from pathlib import Path

def load_efficientnet(model_path, num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    
    # Match the classifier structure used during training
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, num_classes)
    )
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                new_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict)
            else:
                new_state_dict = {}
                for key, value in checkpoint.items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["yolov8s", "trained"], default="yolov8s",
                        help="Choose the model to use: 'yolov8s' or 'trained'")
    args = parser.parse_args()

    if args.model_type == "trained":
        dataset_dir = kagglehub.dataset_download("moltean/fruits")
        train_dir = os.path.join(dataset_dir, 
                                 "fruits-360_dataset_100x100",
                                 "fruits-360",
                                 "Training")
        
        temp_dataset = ImageFolder(root=train_dir)
        all_classes = temp_dataset.classes
        num_classes = len(all_classes)
        print(f"Number of classes from dataset: {num_classes}")
        
        MODEL_PATH = Path("models/efficientnet_b3_fruits.pth")
        
        model = load_efficientnet(MODEL_PATH, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        model = YOLO("ultralytics/yolov8s.pt")
        allowed_class_indices = [46, 47, 49, 50, 51]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.model_type == "trained":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = all_classes[predicted.item()]
                conf_value = confidence.item()

            if conf_value > 0.5:
                h, w = frame.shape[:2]
                text = f"{predicted_class} ({conf_value:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                rect_top = 30
                rect_left = 30
                rect_bottom = rect_top + text_height + 20
                rect_right = rect_left + text_width + 20
                
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                              (rect_left, rect_top), 
                              (rect_right, rect_bottom), 
                              (0, 128, 0),
                              -1)
                
                alpha = 0.7
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                cv2.putText(frame, 
                            text,
                            (rect_left + 10, rect_bottom - 10),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness)
            
            annotated_frame = frame

        else:
            results = model(frame, classes=allowed_class_indices)
            annotated_frame = results[0].plot()

        cv2.imshow("Fruit and Vegetable Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 