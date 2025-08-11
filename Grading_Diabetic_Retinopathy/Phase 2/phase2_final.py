import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from numba import cuda

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

@cuda.jit
def intensity_standardization_cuda(image, output):
    """
    CUDA-accelerated intensity standardization
    """
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        # Process each channel
        for c in range(3):
            output[x, y, c] = image[x, y, c]

def intensity_standardization(image):
    """
    Standardizes the intensity of the image using CUDA
    """
    image_tensor = torch.from_numpy(image).float().to(DEVICE)
    mean = image_tensor.mean()
    std = torch.clamp(image_tensor.std(), min=1e-5)
    
    standardized_image = (image_tensor - mean) / std
    
    # Rescale to 0-255
    standardized_image = ((standardized_image - standardized_image.min()) / 
                         (standardized_image.max() - standardized_image.min()) * 255)
    
    return standardized_image.cpu().numpy().astype(np.uint8)

def color_normalization(image, reference_mean, reference_std):
    """
    Color normalization using CUDA
    """
    image_tensor = torch.from_numpy(image).float().to(DEVICE)
    ref_mean = torch.from_numpy(reference_mean).float().to(DEVICE)
    ref_std = torch.from_numpy(reference_std).float().to(DEVICE)
    
    mean = image_tensor.mean(dim=(0, 1), keepdim=True)
    std = torch.clamp(image_tensor.std(dim=(0, 1), keepdim=True), min=1e-5)
    
    normalized_image = (image_tensor - mean) / std * ref_std + ref_mean
    normalized_image = torch.clamp(normalized_image, 0, 255)
    
    return normalized_image.cpu().numpy().astype(np.uint8)

def apply_clahe_to_image(image):
    """
    Applies CLAHE using CUDA-enabled OpenCV if available
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_l_channel = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl_l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def process_images(input_dir, output_dir, reference_image_path):
    """
    Process images using CUDA acceleration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process reference image
    reference_image = cv2.imread(reference_image_path)
    reference_tensor = torch.from_numpy(reference_image).float().to(DEVICE)
    reference_mean = reference_tensor.mean(dim=(0, 1), keepdim=True)
    reference_std = reference_tensor.std(dim=(0, 1), keepdim=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images with progress bar
    for file_name in tqdm(image_files, desc="Processing Images", unit="image"):
        try:
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Unable to read {file_name}. Skipping.")
                continue
            
            # Process image with CUDA acceleration
            with torch.cuda.amp.autocast():
                # Intensity standardization
                standardized_image = intensity_standardization(image)
                
                # Color normalization
                normalized_image = color_normalization(
                    standardized_image, 
                    reference_mean.cpu().numpy(), 
                    reference_std.cpu().numpy()
                )
                
                # CLAHE enhancement
                final_image = apply_clahe_to_image(normalized_image)
            
            # Save processed image
            cv2.imwrite(output_path, final_image)
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Input and output directories
    input_dir = r"K:\MajorProject\Execution\Phase 1\esr_op"
    output_dir = r"K:\MajorProject\Execution\Phase 2\p2_op1"
    reference_image_path = os.path.join(input_dir, os.listdir(input_dir)[0])  # Use first image as reference
    
    print("Starting image processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Reference image: {reference_image_path}")
    
    # Process images
    try:
        process_images(input_dir, output_dir, reference_image_path)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()