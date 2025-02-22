import cv2
from ultralytics import YOLO

def main():
    # 1. Load a pre-trained YOLOv8 model.
    model = YOLO("ultralytics/yolov8s.pt")

    # Define allowed class indices for fruits and vegetables.
    # Included classes:
    #   46: banana (fruit)
    #   47: apple (fruit)
    #   49: orange (fruit)
    #   50: broccoli (vegetable)
    #   51: carrot (vegetable)
    allowed_class_indices = [46, 47, 49, 50, 51]
    
    # You can check the full class mapping with:
    # print(model.names)

    # 2. Initialize webcam capture (0 = default camera).
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if there's an issue with the webcam feed

        # 3. Run inference on the current frame with detection filtering.
        results = model(frame, classes=allowed_class_indices)

        # 4. Get an annotated frame from YOLO results (bounding boxes, labels).
        annotated_frame = results[0].plot()

        # 5. Display the annotated frame in a window.
        cv2.imshow("Fruit and Vegetable Detection", annotated_frame)

        # Press 'q' to quit the real-time detection loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Release resources.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 