import os
import glob
import time
import tensorflow as tf
import kagglehub
from tqdm import tqdm

# Global configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def download_dataset():
    """
    Downloads the latest fruits dataset using kagglehub and returns the dataset directory path.
    
    The downloaded dataset is expected to contain (at least) two subdirectories: 'Training' and 'Test'.
    We will use the 'Training' subset for fine-tuning.
    """
    path = kagglehub.dataset_download("moltean/fruits")
    print("Dataset downloaded to:", path)
    return path

def get_training_dir(dataset_dir):
    """
    Returns the path to the training set directory.
    
    Updated to reflect the dataset structure downloaded from fruits-360_dataset_100x100:
    The folder structure is:
      dataset_dir/
         fruits-360_dataset_100x100/
             fruits-360/
                 Training/
                 Test/
                 Readme.md
                 LICENSE
    """
    # The full path to the training directory using the new dataset structure.
    training_dir = os.path.join(dataset_dir, 
                                "fruits-360_dataset_100x100",
                                "fruits-360",
                                "Training")
    if not os.path.isdir(training_dir):
        raise ValueError(f"'Training' directory not found in {dataset_dir}. Check the dataset structure.")
    print("Training directory found at:", training_dir)
    return training_dir

def get_class_names(training_dir):
    """
    Retrieves a sorted list of class names from the training directory.
    
    Each subdirectory of the training folder is assumed to correspond to one class.
    """
    class_names = [
        entry for entry in os.listdir(training_dir)
        if os.path.isdir(os.path.join(training_dir, entry))
    ]
    class_names = sorted(class_names)
    return class_names

def decode_img(img):
    """
    Decodes a JPEG image, resizes it to IMG_SIZE, and returns the image tensor.
    """
    # Decode the image to a uint8 tensor with 3 channels.
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize to the desired dimensions.
    img = tf.image.resize(img, IMG_SIZE)
    return img

def process_path(file_path, lookup_table):
    """
    Given a file path, loads and preprocesses the image and extracts its label.
    
    - Assumes a file structure: TrainingDir/class_name/image_filename
    - Uses a lookup table to convert the class name (extracted from the folder name) to an integer label.
    - Applies EfficientNet-specific preprocessing.
    """
    # Split the file path into parts and extract the class name (assumed to be the parent folder)
    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    label = lookup_table.lookup(class_name)
    
    # Read the image file
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # Preprocess using EfficientNet's preprocessing function
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def get_data_augmentation():
    """Instantiate the augmentation pipeline only once."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),   # Horizontal flip
        tf.keras.layers.RandomRotation(0.1),          # Random rotation up to 10%
        tf.keras.layers.RandomZoom(0.1),              # Random zoom by up to 10%
        tf.keras.layers.RandomTranslation(0.1, 0.1)   # Random translation
    ])

def process_path_with_augmentation(file_path, lookup_table, training=True, augmentation=None):
    """
    Given a file path, load and preprocess the image and extract its label.
    Optionally apply augmentation if provided.
    """
    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    label = lookup_table.lookup(class_name)
    
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    if training and augmentation is not None:
        # Use the pre-instantiated augmentation pipeline, rather than creating a new one
        img = augmentation(img, training=True)
    
    return img, label

def prepare_dataset(training_dir):
    """
    Prepares a tf.data.Dataset for fine-tuning:
      - Extracts class names and creates a lookup table for mapping labels to indices.
      - Constructs a file pattern for all jpg files.
      - Maps each file path to a (image, label) tuple.
      - Applies augmentation (if training), shuffles, batches and prefetches the data.
    """
    # Get sorted class names from the training directory.
    class_names = get_class_names(training_dir)
    num_classes = len(class_names)
    print("Found classes:", class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create a lookup table to map class names to integer labels.
    keys = tf.constant(class_names)
    values = tf.constant(list(range(num_classes)), dtype=tf.int32)
    lookup_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1)
    
    # Create the augmentation pipeline once.
    augmentation = get_data_augmentation()
    
    # Construct a file pattern for all jpg files.
    pattern = os.path.join(training_dir, "*", "*.jpg")
    list_ds = tf.data.Dataset.list_files(pattern, shuffle=True)
    
    # Map each file path to (image, label) using the pre-instantiated augmentation pipeline.
    ds = list_ds.map(
        lambda x: process_path_with_augmentation(x, lookup_table, training=True, augmentation=augmentation),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    # Limit to complete batches.
    ds = ds.take(count_files(training_dir) // BATCH_SIZE)
    
    return ds

def count_files(training_dir):
    """
    Count the total number of image files in the training directory.
    """
    pattern = os.path.join(training_dir, "*", "*.jpg")
    return len(glob.glob(pattern))

def get_data_pipeline():
    """
    Downloads the dataset, selects the training set, and returns a preprocessed tf.data.Dataset
    ready for fine-tuning, along with the path to the training directory.
    """
    dataset_dir = download_dataset()
    training_dir = get_training_dir(dataset_dir)
    ds = prepare_dataset(training_dir)
    return ds, training_dir

if __name__ == "__main__":
    start_time = time.time()
    ds, training_dir = get_data_pipeline()
    
    total_files = count_files(training_dir)
    # Compute the total number of batches given the batch size
    total_batches = total_files // BATCH_SIZE  # Remove the +1 to avoid the partial batch
        
    print(f"Total files: {total_files}, Total batches: {total_batches}")
    
    try:
        # Iterate through the dataset with a progress bar
        for _ in tqdm(ds, total=total_batches, desc="Preprocessing"):
            pass
    except tf.errors.OutOfRangeError:
        print("Reached end of dataset (this is normal)")
    
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")

    # Visualization of augmentation (using an actual image from the dataset)
    import matplotlib.pyplot as plt
    
    # Get the first image file from the training directory
    sample_class = os.listdir(training_dir)[0]  # Get first class directory
    sample_image_dir = os.path.join(training_dir, sample_class)
    sample_image_path = os.path.join(sample_image_dir, os.listdir(sample_image_dir)[0])  # Get first image
    
    print(f"Visualizing augmentation using sample image: {sample_image_path}")
    
    img = tf.io.read_file(sample_image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    # Normalize as needed (either /255 or using model-specific preprocessing)
    img = img / 255.0

    aug = get_data_augmentation()
    augmented_img = aug(tf.expand_dims(img, axis=0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmented_img, axis=0))
    plt.title("Augmented")
    plt.show() 