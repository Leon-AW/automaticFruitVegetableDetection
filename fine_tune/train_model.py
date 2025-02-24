import os
import glob
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader, IterableDataset
import kagglehub
import multiprocessing
import urllib.request  # Needed for the patch
import tensorflow as tf
import fine_tune.data_pipeline as dp
import torchvision.transforms.functional as TF
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

# --- Patch torch.hub.download_url_to_file to include a progress bar ---
_original_download_url_to_file = torch.hub.download_url_to_file

def download_url_to_file_with_progress(url, dst, hash_prefix=None, progress=True):
    if not progress:
        return _original_download_url_to_file(url, dst, hash_prefix, progress)
    desc = os.path.basename(dst) or "download"
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
        last_block = [0]
        def reporthook(block_num, block_size, total_size):
            if total_size is not None:
                pbar.total = total_size
            pbar.update((block_num - last_block[0]) * block_size)
            last_block[0] = block_num
        urllib.request.urlretrieve(url, dst, reporthook=reporthook)

torch.hub.download_url_to_file = download_url_to_file_with_progress

# =============================================================================
# Define custom datasets to wrap the TF data pipeline
# =============================================================================

class TFTrainingDataset(IterableDataset):
    """
    Uses the TensorFlow data pipeline (with augmentation) to yield training samples.
    
    The data_pipeline.prepare_dataset() function (from data_pipeline.py) is used to construct
    a tf.data.Dataset, which is then unbatched. Each sample is converted from a TF tensor
    (with values in [-1,1]) to a PyTorch tensor (converted to [0,1] then normalized).
    """
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.num_samples = dp.count_files(training_dir)
        
    def __iter__(self):
        # Build TF dataset (with augmentation) from the training directory.
        ds = dp.prepare_dataset(self.training_dir)  # returns a batched tf.data.Dataset
        ds = ds.unbatch()  # Make it yield individual (image, label) examples
        
        for image, label in ds:
            # Convert tf.Tensor (image is [H,W,3] in [-1,1]) to numpy
            img_np = image.numpy()
            # Convert from [-1,1] to [0,1]
            img_np = (img_np + 1.0) / 2.0
            # Transpose from H x W x C to C x H x W for PyTorch
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).float()
            # Normalize using ImageNet mean and std
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
            yield img_tensor, int(label.numpy())
            
    def __len__(self):
        return self.num_samples

# Helper function (for validation) without augmentation
def process_path_no_aug(file_path, lookup_table):
    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    label = lookup_table.lookup(class_name)
    img = tf.io.read_file(file_path)
    img = dp.decode_img(img)  # Resizes to dp.IMG_SIZE (224,224)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

class TFValidationDataset(IterableDataset):
    """
    Builds a TF data pipeline for validation (no augmentation) using the Test directory.
    """
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.num_samples = len(glob.glob(os.path.join(test_dir, "*", "*.jpg")))
        # For the lookup table, use the (sorted) class names from the test directory.
        self.class_names = sorted([
            entry for entry in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, entry))
        ])
        keys = tf.constant(self.class_names)
        values = tf.constant(list(range(len(self.class_names))), dtype=tf.int32)
        self.lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1)
        
    def __iter__(self):
        pattern = os.path.join(self.test_dir, "*", "*.jpg")
        list_ds = tf.data.Dataset.list_files(pattern, shuffle=False)
        ds = list_ds.map(lambda x: process_path_no_aug(x, self.lookup_table),
                         num_parallel_calls=dp.AUTOTUNE)
        ds = ds.batch(1)
        ds = ds.prefetch(buffer_size=dp.AUTOTUNE)
        ds = ds.unbatch()
        for image, label in ds:
            img_np = image.numpy()
            img_np = (img_np + 1.0) / 2.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).float()
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
            yield img_tensor, int(label.numpy())
            
    def __len__(self):
        return self.num_samples

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

def main():
    # Use the TF-based pipeline for data loading.
    # Download the dataset using the function from data_pipeline.py.
    dataset_dir = dp.download_dataset()
    training_dir = dp.get_training_dir(dataset_dir)
    test_dir = os.path.join(dataset_dir, 
                            "fruits-360_dataset_100x100",
                            "fruits-360",
                            "Test")
    
    # Show the class mapping (from the training directory)
    temp_classes = sorted(os.listdir(training_dir))
    print("\nClass mapping:")
    for idx, class_name in enumerate(temp_classes):
        print(f"{idx}: {class_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Reduce batch size to save memory
    batch_size = 16 if torch.cuda.is_available() else 8
    
    # Create our custom TF-based datasets
    train_dataset = TFTrainingDataset(training_dir)
    val_dataset   = TFValidationDataset(test_dir)
    
    # When using IterableDataset, DataLoader cannot shuffle. We let the TF pipeline internally shuffle.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=0
    )
    
    # Define the model: load EfficientNet B3 from torchvision and replace the classifier.
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, len(temp_classes))
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
    num_epochs = 10
    early_stopping_patience = 3
    no_improve_count = 0
    best_val_loss = float("inf")
    
    grad_accum_steps = 4  # Accumulate gradients over 4 batches
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Update save paths for model and metrics
    model_name = "efficientnet_b3_fruits"
    save_path = os.path.join("models", f"{model_name}.pth")
    metrics_path = os.path.join("models", f"{model_name}_metrics.json")
    
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
                with torch.amp.autocast(device_type='cuda'):
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
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
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
            no_improve_count = 0
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"Saved best model to {save_path}")
            
            # Calculate and save metrics for the best model
            y_pred, y_true = evaluate_model(model, val_loader, device)
            metrics = calculate_metrics(y_true, y_pred)
            
            # Add validation loss and accuracy
            metrics['val_loss'] = float(val_loss)
            metrics['val_accuracy'] = float(val_acc)
            
            # Save metrics to JSON file
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
    
    # Load the best model and compute final metrics
    print("\nComputing final metrics using best model...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate final metrics on validation set
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