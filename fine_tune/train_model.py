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

def main():
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
    
    # Setup device and DataLoader parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    batch_size = 16 if torch.cuda.is_available() else 8
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load EfficientNet B3 and adjust its classifier to fit our number of classes.
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, len(train_dataset.classes))
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    num_epochs = 10
    early_stopping_patience = 3
    no_improve_count = 0
    best_val_loss = float("inf")
    grad_accum_steps = 4  # Accumulate gradients over 4 batches
    
    # Create directory to save models and metrics.
    os.makedirs('models', exist_ok=True)
    model_name = "efficientnet_b3_fruits"
    save_path = os.path.join("models", f"{model_name}.pth")
    metrics_path = os.path.join("models", f"{model_name}_metrics.json")
    
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
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
            
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
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct_val / total_val:.2f}%'
                })
        
        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Check for improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            
            # Save the best model.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"Saved best model to {save_path}")
            
            # Evaluate to calculate detailed metrics.
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