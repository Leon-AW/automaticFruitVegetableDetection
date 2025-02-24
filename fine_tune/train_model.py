import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader
import kagglehub
import multiprocessing
import json
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from pathlib import Path
from contextlib import nullcontext

# =============================================================================
# Main training function using the TF preprocessed data
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall and f1-score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions and true labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_predictions, all_labels

def download_dataset():
    """
    Downloads the latest version of the fruits dataset using kagglehub.
    """
    path = kagglehub.dataset_download("moltean/fruits")
    print("Path to dataset files:", path)
    return path

def get_optimal_batch_size(model, device):
    """Calculate optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 8  # Default CPU batch size
    
    # Get GPU memory in GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Rough estimation: assume each image + model overhead takes ~1.5GB for batch_size=32
    # This is a conservative estimate that includes memory for gradients and optimizer states
    estimated_memory_per_32_batch = 1.5
    
    optimal_batch_size = int((gpu_memory / estimated_memory_per_32_batch) * 32)
    # Round down to nearest power of 2 for better performance
    optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
    
    # Set reasonable bounds
    optimal_batch_size = max(min(optimal_batch_size, 256), 16)
    return optimal_batch_size

def main():
    # Create absolute path to models directory in project root and ensure it exists
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "efficientnet_b3_fruits"
    save_path = models_dir / f"{model_name}.pth"
    metrics_path = models_dir / f"{model_name}_metrics.json"
    
    print(f"Model will be saved to: {save_path}")
    print(f"Metrics will be saved to: {metrics_path}")

    # Download the dataset from Kaggle Hub
    dataset_path = download_dataset()

    # Define the training and test directories based on the file structure.
    training_dir = os.path.join(dataset_path, "fruits-360_dataset_100x100", "fruits-360", "Training")
    test_dir = os.path.join(dataset_path, "fruits-360_dataset_100x100", "fruits-360", "Test")
    
    # Save one sample image from the training and test directories so you can inspect them.
    train_image_paths = glob.glob(os.path.join(training_dir, "*", "*.jpg"))
    test_image_paths = glob.glob(os.path.join(test_dir, "*", "*.jpg"))
    
    if train_image_paths:
        sample_train_img = Image.open(train_image_paths[0])
        sample_train_save_path = "sample_train.jpg"
        sample_train_img.save(sample_train_save_path)
        print("Saved a sample training image to:", sample_train_save_path)
    else:
        print("No training images found!")
        
    if test_image_paths:
        sample_test_img = Image.open(test_image_paths[0])
        sample_test_save_path = "sample_test.jpg"
        sample_test_img.save(sample_test_save_path)
        print("Saved a sample test image to:", sample_test_save_path)
    else:
        print("No test images found!")
        
    # Define a simple transform that converts the PIL image to a tensor.
    # (We're not adding resizing, normalization, or augmentation in this case.)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create datasets using ImageFolder (it builds the class mapping from folder names).
    train_dataset = ImageFolder(root=training_dir, transform=transform)
    val_dataset = ImageFolder(root=test_dir, transform=transform)
    
    # Show the class mapping.
    print("\nClass mapping:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"{idx}: {class_name}")
    
    # Setup device and DataLoader parameters with optimal settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load EfficientNet B3 and adjust its classifier to fit our number of classes.
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, len(train_dataset.classes))
    )
    model = model.to(device)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set optimal batch size based on GPU memory
        batch_size = get_optimal_batch_size(model, device)
        # Use number of CPU cores for DataLoader workers, leaving some cores free
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        # Enable pinned memory for faster GPU transfer
        pin_memory = True
    else:
        batch_size = 8
        num_workers = 0
        pin_memory = False
    
    print(f"Using batch size: {batch_size}")
    print(f"Using {num_workers} worker processes")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Set up gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    num_epochs = 10
    early_stopping_patience = 3
    no_improve_count = 0
    best_val_loss = float("inf")
    grad_accum_steps = 4  # Accumulate gradients over 4 batches
    max_grad_norm = 1.0  # For gradient clipping
    
    # Main training loop.
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=True)
        for i, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Skip this batch if we get NaN loss
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print(f"WARNING: NaN or Inf loss detected, skipping batch")
                continue
                
            # Backward pass
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    # Clip gradients to prevent explosion
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate after the training epoch.
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        nan_batches = 0
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Skip this batch if we get NaN loss
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    nan_batches += 1
                    continue
                    
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct_val / total_val:.2f}%'
                })
        
        if nan_batches > 0:
            print(f"WARNING: {nan_batches} validation batches had NaN loss and were skipped")
        
        if total_val > 0:  # Only calculate metrics if we have valid batches
            val_loss = val_loss / total_val
            val_acc = correct_val / total_val
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)
                print(f"Saved best model to {save_path}")
                
                # Evaluate to calculate detailed metrics
                y_pred, y_true = evaluate_model(model, val_loader, device)
                metrics = calculate_metrics(y_true, y_pred)
                metrics['val_loss'] = float(val_loss)
                metrics['val_accuracy'] = float(val_acc)
                
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                print(f"Saved metrics to {metrics_path}")
                
                print("\nBest model metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1_score']:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        else:
            print("WARNING: All validation batches had NaN loss, skipping this epoch for model saving")

    print("Training complete.")
    
    # Load the best model and compute the final metrics.
    print("\nComputing final metrics using best model...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    y_pred, y_true = evaluate_model(model, val_loader, device)
    final_metrics = calculate_metrics(y_true, y_pred)
    
    print("\nFinal Model Metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1-score: {final_metrics['f1_score']:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()