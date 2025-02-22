from ultralytics import YOLO
import yaml
import os

def load_configs(dataset_config_path="fruit_veg_detection_finetuning/configs/dataset.yaml",
                 hyperparams_config_path="fruit_veg_detection_finetuning/configs/hyperparams.yaml"):
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    with open(hyperparams_config_path, 'r') as f:
        hyperparams = yaml.safe_load(f)
    return dataset_config, hyperparams

def train_model(dataset_config_path="fruit_veg_detection_finetuning/configs/dataset.yaml",
                hyperparams_config_path="fruit_veg_detection_finetuning/configs/hyperparams.yaml"):
    # Load configurations
    dataset_config, hyperparams = load_configs(dataset_config_path, hyperparams_config_path)
    
    # Get baseline model and hardware device from hyperparams.
    baseline_model = hyperparams.get("baseline_model", "ultralytics/yolov8s.pt")
    # Specify the device to run training on; e.g. "0" for a single GPU, "0,1" for multiple GPUs.
    device = hyperparams.get("device", "0")
    print(f"Using baseline model: {baseline_model} on device: {device}")

    # Initialize the YOLO model with pre‑trained weights.
    model = YOLO(baseline_model)
    
    # Train the model using the dataset configuration and hyperparameters.
    print("Starting training with progress monitoring on GPU(s)...")
    results = model.train(
        data=dataset_config_path,
        epochs=hyperparams.get("epochs", 50),
        batch=hyperparams.get("batch_size", 16),
        imgsz=hyperparams.get("imgsz", 640),
        lr0=hyperparams.get("learning_rate", 0.001),
        device=device,
        save=True,
        verbose=True,  # Enable verbose output with progress information
        # Optionally, you can add other GPU-related parameters (e.g., workers) if needed:
        workers=hyperparams.get("workers", 8)
    )
    
    # Optionally, move the best model checkpoint to your models folder.
    models_dir = "fruit_veg_detection_finetuning/models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pt")
    model.save(weights=model_path)
    print(f"Training complete. Saved the best model to {model_path}")
    return results

def main():
    train_model()

if __name__ == "__main__":
    main() 