import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import gc

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

class ImageEnhancementCNN(nn.Module):
    def __init__(self):
        super(ImageEnhancementCNN, self).__init__()
        
        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            self.conv_block(3, 64, kernel_size=3, stride=1, padding=1),
            self.conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            self.conv_block(128, 256, kernel_size=3, stride=2, padding=1),
        )
        
        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[self.residual_block(256) for _ in range(5)]
        )
        
        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        features = self.encoder(x)
        residual = self.residual_blocks(features)
        features = features + residual
        enhanced = self.decoder(features)
        return enhanced

def enhance_image(model, img_path, transform, device):
    try:
        # Load and preprocess single image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Pad image
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        padding_h = new_h - h
        padding_w = new_w - w
        img_tensor = nn.functional.pad(img_tensor, (0, padding_w, 0, padding_h), mode='reflect')

        # Process image
        with torch.no_grad():
            enhanced_img = model(img_tensor)
            enhanced_img = enhanced_img[:, :, :h, :w]
            
            # Move to CPU and free GPU memory
            enhanced_img = enhanced_img.cpu()
            del img_tensor
            torch.cuda.empty_cache()

        # Post-process
        enhanced_img = (enhanced_img + 1) / 2.0
        enhanced_img = enhanced_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        enhanced_img = Image.fromarray((enhanced_img * 255).astype('uint8'))
        
        if enhanced_img.size != original_size:
            enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

        return enhanced_img

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def enhance_images(model, input_folder, output_folder, device, batch_size=5):
    try:
        os.makedirs(output_folder, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        model.eval()

        # Get list of image files
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No valid images found in {input_folder}")
            return

        # Process images in batches
        successful_count = 0
        total_batches = (len(image_files) - 1) // batch_size + 1
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            
            for filename in tqdm(batch, desc=f"Batch {i//batch_size + 1}/{total_batches}"):
                try:
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, f"enhanced_{filename}")

                    enhanced_img = enhance_image(model, input_path, transform, device)
                    
                    if enhanced_img is not None:
                        enhanced_img.save(output_path)
                        successful_count += 1

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Successfully processed {successful_count} out of {len(image_files)} images.")
        print(f"Enhanced images saved in: {output_folder}")

    except Exception as e:
        print(f"Error in enhance_images: {str(e)}")

def main():
    try:
        # Load model
        print("Initializing model...")
        model = ImageEnhancementCNN().to(device)
        model_path = r"K:\MajorProject\Execution\Phase 1\CntEnh\best_image_enhancement_model.pth"
        
        print("Loading model weights...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")

        # Set directories
        input_folder = r"K:\MajorProject\Execution\Phase 1\dn_op"
        output_folder = r"K:\MajorProject\Execution\Phase 1\CE_op"

        # Verify paths
        if not os.path.exists(input_folder):
            print(f"Input folder does not exist: {input_folder}")
            return

        print(f"Processing images from: {input_folder}")
        print(f"Saving results to: {output_folder}")

        # Process images
        enhance_images(model, input_folder, output_folder, device, batch_size=5)

    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Set CUDA to be synchronous for better error reporting
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    main()