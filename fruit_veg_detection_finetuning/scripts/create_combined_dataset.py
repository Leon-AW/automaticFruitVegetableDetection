import os
import shutil
import zipfile
import kagglehub

def download_and_extract(dataset_slug, extract_to="fruit_veg_detection_finetuning/data/raw"):
    """
    Downloads and extracts a dataset from KaggleHub.
    
    Args:
        dataset_slug (str): The Kaggle dataset slug (e.g., "moltean/fruits").
        extract_to (str): Directory where to extract the dataset.
    
    Returns:
        str: The path to the extracted dataset folder.
    """
    print(f"Downloading dataset: {dataset_slug}")
    # Download the latest version of the dataset.
    # This returns the path to the downloaded zip file or directory.
    dataset_path = kagglehub.dataset_download(dataset_slug)
    print(f"Path to dataset file: {dataset_path}")

    # Create a specific folder for this dataset
    extraction_folder = os.path.join(extract_to, dataset_slug.replace("/", "_"))
    os.makedirs(extraction_folder, exist_ok=True)

    if os.path.isdir(dataset_path):
        # Dataset is already a directory, so copy its contents.
        print("Downloaded path is a directory; copying contents...")
        for item in os.listdir(dataset_path):
            s = os.path.join(dataset_path, item)
            d = os.path.join(extraction_folder, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        # It's a zip file, extract its contents.
        print("Extracting files from zip archive...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)
    print(f"Extracted {dataset_slug} to {extraction_folder}")
    return extraction_folder

def combine_datasets(source_dirs, target_dir="fruit_veg_detection_finetuning/data/combined"):
    """
    Merges files from multiple source directories into a single target directory.
    
    Args:
        source_dirs (list): List of directories containing extracted datasets.
        target_dir (str): Directory to store the combined dataset.
    """
    os.makedirs(target_dir, exist_ok=True)
    for source in source_dirs:
        # Walk through the source directory and copy every file.
        for root, _, files in os.walk(source):
            for file in files:
                file_source_path = os.path.join(root, file)
                file_target_path = os.path.join(target_dir, file)
                # If the file exists, rename to prevent overwriting.
                if os.path.exists(file_target_path):
                    base, ext = os.path.splitext(file)
                    file_target_path = os.path.join(
                        target_dir, f"{base}_{os.path.basename(root)}{ext}"
                    )
                shutil.copy(file_source_path, file_target_path)
    print("All datasets have been combined successfully into:", target_dir)

def main():
    # Directories setup
    raw_data_dir = "fruit_veg_detection_finetuning/data/raw"
    combined_dir = "fruit_veg_detection_finetuning/data/combined"
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # List of datasets to download
    datasets = [
        "moltean/fruits",
        "kvnpatel/fruits-vegetable-detection-for-yolov4"
    ]
    
    extracted_folders = []
    for dataset in datasets:
        folder = download_and_extract(dataset, extract_to=raw_data_dir)
        extracted_folders.append(folder)
    
    # Include the local dataset folder in the merge.
    local_dataset = "fruit_veg_detection_finetuning/data/Fruits and vegetables.v1-zone-verif.yolov8"
    if os.path.exists(local_dataset):
        print(f"Including local dataset: {local_dataset}")
        extracted_folders.append(local_dataset)
    else:
        print("Local dataset folder not found:", local_dataset)
    
    # Combine extracted datasets into one unified dataset folder.
    combine_datasets(extracted_folders, target_dir=combined_dir)

if __name__ == "__main__":
    main() 