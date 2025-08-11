import torch
import torch.nn as nn
from torchvision import transforms
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

class MemoryEfficientRRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(MemoryEfficientRRDB, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x

class ESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=5):
        super(ESRGANGenerator, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[MemoryEfficientRRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out

def load_model(model_path):
    try:
        model = ESRGANGenerator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def process_image(model, image_path):
    try:
        # Load the image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        low_res = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(low_res)
            
            # Move to CPU and clear CUDA memory
            output = output.cpu()
            del low_res
            torch.cuda.empty_cache()

        # Post-process the output
        output = output.squeeze().clamp_(-1, 1).add_(1).div_(2)
        output = transforms.ToPILImage()(output)
        
        # Resize back to original size if needed
        if output.size != original_size:
            output = output.resize(original_size, Image.LANCZOS)
        
        return output

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_batch(model, image_files, input_folder, output_folder, batch_size=5):
    try:
        successful_count = 0
        total_batches = (len(image_files) - 1) // batch_size + 1

        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            
            for image_file in tqdm(batch, desc=f"Batch {i//batch_size + 1}/{total_batches}"):
                try:
                    input_path = os.path.join(input_folder, image_file)
                    output_path = os.path.join(output_folder, f"sr_{image_file}")
                    
                    # Process the image
                    sr_image = process_image(model, input_path)
                    
                    if sr_image is not None:
                        # Save the super-resolved image
                        sr_image.save(output_path)
                        successful_count += 1

                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue

            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

        return successful_count

    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        return 0

def main():
    try:
        # Specify your paths here
        input_folder = r"K:\MajorProject\Execution\Phase 1\CE_op"
        output_folder = r"K:\MajorProject\Execution\Phase 1\esr_op"
        model_path = r"K:\MajorProject\Execution\Phase 1\ESRGAN\model_epoch_1.pth"

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Verify paths
        if not os.path.exists(input_folder):
            print(f"Input folder does not exist: {input_folder}")
            return

        print("Loading model...")
        model = load_model(model_path)
        if model is None:
            return

        # Get list of image files
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if not image_files:
            print(f"No valid images found in {input_folder}")
            return

        print(f"Processing {len(image_files)} images...")
        successful_count = process_batch(model, image_files, input_folder, output_folder, batch_size=5)

        print(f"Successfully processed {successful_count} out of {len(image_files)} images.")
        print(f"Results saved in {output_folder}")

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