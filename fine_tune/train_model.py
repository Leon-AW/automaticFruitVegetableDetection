import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader
import kagglehub
import multiprocessing
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

# Early stopping class to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        return self.early_stop

# Function to visualize and save preprocessed images
def visualize_preprocessed_images(dataloader, class_names, save_dir="images/preprocessed"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images.cpu()  # Move to CPU for visualization
    
    # Create a grid of images
    grid_size = min(16, len(images))
    selected_indices = random.sample(range(len(images)), grid_size)
    
    # Function to denormalize images for visualization
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Plot the grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, idx in enumerate(selected_indices):
        ax = axes[i//4, i%4]
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        label_idx = labels[idx].item()
        ax.set_title(class_names[label_idx])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "preprocessed_samples.png"))
    print(f"Saved preprocessed images to {os.path.join(save_dir, 'preprocessed_samples.png')}")
    plt.close()

def main():
    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if torch.cuda.is_available() else 2  # More workers for GPU
    pin_memory = torch.cuda.is_available()  # Only pin memory if using GPU
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # --- Dataset & Transforms ---
    # Get the dataset directory from kagglehub
    dataset_dir = kagglehub.dataset_download("moltean/fruits")
    
    # Print the dataset directory structure to understand available data
    print(f"Dataset directory: {dataset_dir}")
    print("Contents of dataset directory:")
    for item in os.listdir(dataset_dir):
        print(f" - {item}")
        if os.path.isdir(os.path.join(dataset_dir, item)):
            for subitem in os.listdir(os.path.join(dataset_dir, item)):
                print(f"   - {subitem}")

    # Find main dataset directory (looking for the directory containing Training and Test folders)
    dataset_folders = []
    for root, dirs, files in os.walk(dataset_dir):
        if "Training" in dirs and "Test" in dirs:
            dataset_folders.append(root)
    
    if not dataset_folders:
        raise ValueError("Could not find Training and Test directories in the dataset")
    
    # Use the first found dataset directory (or choose the one with most images if multiple)
    main_dataset_dir = dataset_folders[0]
    print(f"\nUsing dataset from: {main_dataset_dir}")
    
    # Construct the full paths to Training and Test directories
    train_dir = os.path.join(main_dataset_dir, "Training")
    test_dir = os.path.join(main_dataset_dir, "Test")

    # Verify the directories exist
    if not os.path.isdir(train_dir):
        raise ValueError(f"Training directory {train_dir} does not exist.")
    if not os.path.isdir(test_dir):
        raise ValueError(f"Test directory {test_dir} does not exist.")

    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")

    # More aggressive data augmentation to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((300, 300)),  # Fixed size for both dimensions
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),  # Added vertical flip
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.02),  # Occasional grayscale
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Increased random erasing from 0.1 to 0.2
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((300, 300)),  # Fixed size for both dimensions
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset using the directories provided in the dataset.
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=test_dir, transform=val_transforms)
    
    # Get class information directly from the dataset
    all_classes = train_dataset.classes
    num_classes = len(all_classes)
    print(f"Number of classes detected: {num_classes}")
    print(f"Classes: {all_classes}")

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")

    # Adjust batch size based on available memory
    batch_size = 64 if torch.cuda.is_available() else 32
    
    # Data loaders with optimized settings for GPU/CPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Visualize preprocessed images before training
    visualize_preprocessed_images(train_loader, all_classes)

    # --- Model Setup ---
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # Increase dropout for better regularization
    model.classifier[0] = nn.Dropout(p=0.4, inplace=True)  # Increased from default 0.3
    
    # Replace the classifier head of EfficientNet-B3
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Enable parallel processing if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    
    # Increased weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    
    # Using CosineAnnealingLR for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=2)

    # --- Training Loop ---
    num_epochs = 20
    best_val_loss = float("inf")
    best_val_acc = 0.0
    save_path = "models/fine_tuned_efficientnet_b3.pth"
    results_dir = "images/results"
    os.makedirs(results_dir, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                         leave=True)
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                # Use mixed precision training on GPU
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training on CPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with current loss and accuracy
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc="Validation", leave=True)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if torch.cuda.is_available():
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                # Save predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct_val / total_val:.2f}%'
                })

        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step()
        
        # Create confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # Create confusion matrix (only for a subset of classes if there are too many)
            plt.figure(figsize=(12, 10))
            if num_classes > 20:
                # If too many classes, show a subset of most confused classes
                cm = confusion_matrix(all_labels, all_preds)
                error_indices = np.where((cm.diagonal() / cm.sum(axis=1)) < 0.9)[0]
                selected_indices = error_indices[:min(20, len(error_indices))]
                if len(selected_indices) < 5:  # If not enough errors, just show first 20 classes
                    selected_indices = range(min(20, num_classes))
                
                cm_subset = confusion_matrix([l for i, l in enumerate(all_labels) if l in selected_indices], 
                                            [p for i, p in enumerate(all_preds) if all_labels[i] in selected_indices])
                selected_class_names = [all_classes[i] for i in selected_indices]
                sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=selected_class_names, yticklabels=selected_class_names)
                plt.title(f'Confusion Matrix (Selected Classes) - Epoch {epoch+1}')
            else:
                cm = confusion_matrix(all_labels, all_preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=all_classes, yticklabels=all_classes)
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"{results_dir}/confusion_matrix_epoch_{epoch+1}.png")
            plt.close()
        
        # Plot training and validation metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/training_curves.png")
        plt.close()
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            # Save model state
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'num_classes': num_classes,
                'class_names': all_classes
            }
            torch.save(save_dict, save_path)
            print(f"Saved best model with accuracy: {best_val_acc:.4f}")
        
        # Check for early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered. Ending training.")
            break

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()