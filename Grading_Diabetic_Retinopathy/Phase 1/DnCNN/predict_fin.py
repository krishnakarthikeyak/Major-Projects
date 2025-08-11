import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")

print(f"Using device: {device}")

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=20):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return x - out

# Load pre-trained weights
try:
    model = DnCNN().to(device)
    state_dict = torch.load(r"K:\MajorProject\Execution\Phase 1\DnCNN\dncnn_color_blind.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

def denoise_image(image_path, model):
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Denoise the image
        with torch.no_grad():
            output_tensor = model(input_tensor)
            # Move to CPU immediately and clear CUDA cache
            output_tensor = output_tensor.squeeze().cpu()
            torch.cuda.empty_cache()
            output_tensor = output_tensor.clamp(0, 1)

        # Convert to image
        output_image = transforms.ToPILImage()(output_tensor)
        
        # Clear some memory
        del input_tensor, output_tensor
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_folder(input_folder, output_folder, model, batch_size=10):
    try:
        os.makedirs(output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if not image_files:
            print(f"No valid image files found in {input_folder}")
            return

        # Process in batches
        successful_count = 0
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            for image_file in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}"):
                try:
                    input_path = os.path.join(input_folder, image_file)
                    output_path = os.path.join(output_folder, f"denoised_{image_file}")

                    denoised_image = denoise_image(input_path, model)
                    if denoised_image is not None:
                        denoised_image.save(output_path)
                        successful_count += 1
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
            
            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Successfully processed {successful_count} out of {len(image_files)} images.")
        print(f"Denoised images saved in: {output_folder}")

    except Exception as e:
        print(f"Error in process_folder: {str(e)}")

def main():
    try:
        # Define input and output folders
        input_folder = r"K:\MajorProject\Execution\archive\resized_train\resized_train"
        output_folder = r"K:\MajorProject\Execution\Phase 1\dn_op"

        # Verify folders exist
        if not os.path.exists(input_folder):
            print(f"Input folder does not exist: {input_folder}")
            return

        # Process images
        process_folder(input_folder, output_folder, model, batch_size=5)  # Reduced batch size for better memory management

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()