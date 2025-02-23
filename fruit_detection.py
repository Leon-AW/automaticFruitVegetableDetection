import cv2
from ultralytics import YOLO
import argparse

def main():
    # Parse a command-line argument to choose the model type.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["yolov8s", "trained"], default="yolov8s",
                        help="Choose the model to use: 'yolov8s' for the default pre-trained YOLOv8s model, "
                             "or 'trained' for the custom trained model.")
    args = parser.parse_args()

    # Based on the model type, determine which model file to load.
    if args.model_type == "yolov8s":
        model_path = "ultralytics/yolov8s.pt"
        use_custom_overlay = False  # No extra overlay for the base YOLO model
    else:
        # Replace the below path with the actual path to your custom trained fruit/vegetable detector.
        model_path = "path/to/your_trained_fruit_detection_model.pt"
        use_custom_overlay = True  # We'll add an extra text overlay for the custom model

    # Load the specified model.
    model = YOLO(model_path)
    # Print the model's class mapping (names) for verification.
    print("Model names:", model.names)

    # Define allowed class indices for fruits and vegetables if using the default YOLOv8s model.
    # (These indices come from the dataset on which YOLOv8s was trained.)
    allowed_class_indices = [46, 47, 49, 50, 51]

    # Initialize webcam capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if the frame is not captured properly.

        # Run inference on the current frame.
        # For the default YOLOv8s model, we apply a filter to only allow fruits/vegetables.
        if args.model_type == "yolov8s":
            results = model(frame, classes=allowed_class_indices)
        else:
            # Assume the custom trained model only detects fruits/vegetables (no filtering needed)
            results = model(frame)

        # Get an annotated frame from the YOLO results (bounding boxes, labels, etc.).
        annotated_frame = results[0].plot()

        # If we're using the custom trained model, overlay a clearly visible text box at the bottom
        # showing the detected fruit or vegetable names.
        if use_custom_overlay:
            detected_classes = set()
            if results[0].boxes is not None:
                try:
                    # results[0].boxes.cls is a tensor of detected class indices.
                    cls_array = results[0].boxes.cls.cpu().numpy()
                    for cls in cls_array:
                        detected_classes.add(str(model.names[int(cls)]))
                except Exception as e:
                    print("Error extracting class information:", e)
            detected_text = "Detected: " + (", ".join(detected_classes) if detected_classes else "None")

            # Determine frame dimensions.
            h, w = annotated_frame.shape[:2]
            rect_height = 50  # Height of the overlay rectangle.
            # Draw a filled rectangle across the bottom of the frame.
            cv2.rectangle(annotated_frame, (0, h - rect_height), (w, h), (0, 0, 0), -1)
            # Put the detection text on top of the rectangle.
            cv2.putText(annotated_frame, detected_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting annotated frame.
        cv2.imshow("Fruit and Vegetable Detection", annotated_frame)

        # Press 'q' to quit the real-time detection loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 