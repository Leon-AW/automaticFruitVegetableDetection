import os
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits images from source_dir into train, validation, and test sets based on the provided ratios.
    
    Args:
        source_dir (str): Directory containing the images to split.
        output_dir (str): Directory where the split folders ('train', 'val', 'test') will be created.
        train_ratio (float): Proportion of images for the training set.
        val_ratio (float): Proportion of images for the validation set.
        test_ratio (float): Proportion of images for the test set.
        seed (int): Random seed for reproducibility.
    """
    # Set the seed for reproducibility and create destination folders.
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get the list of valid image files
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)]
    total_files = len(files)
    print(f"Found {total_files} images in {source_dir}")
    
    # Shuffle files randomly
    random.shuffle(files)
    
    # Compute split indexes based on the total number of files
    train_count = int(train_ratio * total_files)
    val_count = int(val_ratio * total_files)
    test_count = total_files - train_count - val_count
    
    print(f"Splitting dataset into {train_count} training, {val_count} validation, and {test_count} test images.")
    
    # Split the file list
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    
    # Copy the images to their respective folders
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))
    
    print("Dataset splitting complete.")

def main():
    # Source directory: normalized images from the preprocessing step.
    source_dir = "fruit_veg_detection_finetuning/data/processed/normalized"
    # Output directory: we'll place train/val/test folders within the same normalized folder.
    output_dir = "fruit_veg_detection_finetuning/data/processed/normalized"
    
    # Perform the splitting (default: 80% train, 10% val, 10% test)
    split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

if __name__ == "__main__":
    main() 