import os
from tqdm import tqdm
import yaml
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
from torch.cuda.amp import autocast, GradScaler

def main():
    # --- Load classes from YAML ---
    with open("classes.yaml", "r") as f:
        classes_data = yaml.safe_load(f)
    # The top-level key is "fruit_and_vegetables". 
    all_classes = list(classes_data["fruit_and_vegetables"].keys())
    num_classes = len(all_classes)
    print(f"Number of classes from YAML: {num_classes}")

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

    # Adapted path: Use the new dataset structure
    # Now the downloaded dataset will have:
    #   dataset_dir/
    #       fruits-360_dataset_100x100/
    #           fruits-360/
    #               Training/
    #               Test/
    #               Readme.md
    #               LICENSE
    train_dir = os.path.join(dataset_dir, 
                             "fruits-360_dataset_100x100",
                             "fruits-360",
                             "Training")
    test_dir = os.path.join(dataset_dir, 
                            "fruits-360_dataset_100x100",
                            "fruits-360",
                            "Test")

    # Verify the directories exist
    if not os.path.isdir(train_dir):
        raise ValueError(f"Training directory {train_dir} does not exist. Contents of dataset directory: {os.listdir(dataset_dir)}")
    if not os.path.isdir(test_dir):
        raise ValueError(f"Test directory {test_dir} does not exist. Contents of dataset directory: {os.listdir(dataset_dir)}")

    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")

    # EfficientNet-B3 recommends an input size of 300x300.
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset using the directories provided in the dataset.
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset   = ImageFolder(root=test_dir, transform=val_transforms)

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

    # --- Model Setup ---
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    scaler = GradScaler() if torch.cuda.is_available() else None

    # --- Training Loop ---
    num_epochs = 20
    best_val_loss = float("inf")
    save_path = "fine_tuned_efficientnet_b3.pth"

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=True)
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
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

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_pbar = tqdm(val_loader, desc="Validation", leave=True)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if torch.cuda.is_available():
                    with autocast():
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'class_names': all_classes
            }
            torch.save(save_dict, save_path)
            print("Saved best model.")

    print("Training complete.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()