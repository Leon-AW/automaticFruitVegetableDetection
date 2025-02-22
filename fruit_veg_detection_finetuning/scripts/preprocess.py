import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar

def adjust_color(image):
    """
    Apply color correction by converting the image to HSV,
    equalizing the V (brightness) channel, and converting back to BGR.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    corrected = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return corrected

def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocess the input image by resizing, applying a Gaussian blur for noise removal,
    and performing color correction.
    """
    # Resize image
    resized = cv2.resize(image, target_size)
    # Remove noise using a small Gaussian blur
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    # Apply color correction
    corrected = adjust_color(denoised)
    return corrected

def adjust_brightness(image, beta=50):
    """
    Increase brightness by adding a constant value (beta) to all pixels.
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

def adjust_contrast(image, alpha=1.5):
    """
    Increase contrast by scaling pixel values by alpha.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def scale_image(image, scale=1.2):
    """
    Simulate a zoom effect by cropping the central region and resizing it back.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h / scale), int(w / scale)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
    scaled = cv2.resize(cropped, (w, h))
    return scaled

def augmentations(image):
    """
    Generate augmented images from the given preprocessed image.
    Returns a list of tuples in the form: (augmentation type, augmented image).
    """
    aug_images = []
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    aug_images.append(("flip", flipped))
    
    # Rotations (90, 180, 270 degrees)
    rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    aug_images.append(("rot90", rot90))
    rot180 = cv2.rotate(image, cv2.ROTATE_180)
    aug_images.append(("rot180", rot180))
    rot270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    aug_images.append(("rot270", rot270))
    
    # Brightness adjustment (increase brightness)
    bright = adjust_brightness(image, beta=50)
    aug_images.append(("bright", bright))
    
    # Contrast adjustment (increase contrast)
    contrast = adjust_contrast(image, alpha=1.5)
    aug_images.append(("contrast", contrast))
    
    # Scaling (simulate zoom in)
    scaled = scale_image(image, scale=1.2)
    aug_images.append(("scale", scaled))
    
    return aug_images

def process_and_augment(input_dir="fruit_veg_detection_finetuning/data/combined",
                        normalized_dir="fruit_veg_detection_finetuning/data/processed/normalized",
                        augmented_dir="fruit_veg_detection_finetuning/data/processed/augmented",
                        target_size=(640, 640)):
    """
    Processes images from the combined dataset:
      - Preprocess images (resize, noise removal, color correction)
      - Save preprocessed images in 'normalized_dir'
      - Create and save augmented versions in 'augmented_dir'
    """
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)
    
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    
    # Get list of files, then iterate with a progress bar.
    image_files = [file for file in os.listdir(input_dir)
                   if file.lower().endswith(valid_extensions)]
    
    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Preprocess the image
        preprocessed = preprocess_image(image, target_size=target_size)
        
        # Save the preprocessed image
        base_name, ext = os.path.splitext(filename)
        norm_filename = f"{base_name}_preprocessed{ext}"
        norm_output_path = os.path.join(normalized_dir, norm_filename)
        cv2.imwrite(norm_output_path, preprocessed)
        
        # Apply augmentations to the preprocessed image
        for aug_type, aug_img in augmentations(preprocessed):
            aug_filename = f"{base_name}_{aug_type}{ext}"
            aug_output_path = os.path.join(augmented_dir, aug_filename)
            cv2.imwrite(aug_output_path, aug_img)

def main():
    process_and_augment()

if __name__ == "__main__":
    main() 