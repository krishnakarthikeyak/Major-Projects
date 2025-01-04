import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar

# Custom Dataset Class
class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, f'sr_enhanced_denoised_{img_name}.jpeg')
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['level']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, upsampled=None):
        x = self.conv(x)
        if upsampled is not None:
            x = x + F.interpolate(upsampled, size=x.shape[-2:], mode='nearest')
        x = self.bn(x)
        x = self.relu(x)
        return x

class HybridUNetCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=5):
        super().__init__()
        
        # U-Net Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # FPN layers
        self.fpn4 = FPNBlock(1024, 256)
        self.fpn3 = FPNBlock(512, 256)
        self.fpn2 = FPNBlock(256, 256)
        self.fpn1 = FPNBlock(128, 256)
        self.fpn0 = FPNBlock(64, 256)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5, 512),  # 5 FPN levels
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # Encoder path
        x0 = self.inc(x)        # 64 channels
        x1 = self.down1(x0)     # 128 channels
        x2 = self.down2(x1)     # 256 channels
        x3 = self.down3(x2)     # 512 channels
        x4 = self.down4(x3)     # 1024 channels

        # FPN path
        p4 = self.fpn4(x4)
        p3 = self.fpn3(x3, p4)
        p2 = self.fpn2(x2, p3)
        p1 = self.fpn1(x1, p2)
        p0 = self.fpn0(x0, p1)

        # Global feature fusion
        f4 = self.avgpool(p4)
        f3 = self.avgpool(p3)
        f2 = self.avgpool(p2)
        f1 = self.avgpool(p1)
        f0 = self.avgpool(p0)

        # Concatenate all features
        out = torch.cat([f0, f1, f2, f3, f4], dim=1)
        
        # Classification
        out = self.classifier(out)
        
        return out

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_class_weights(dataset):
    # Calculate class weights to handle imbalance
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = torch.bincount(torch.tensor(labels))
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.float())
    return weights

def train_model(csv_path, img_dir, save_dir, num_epochs=20, batch_size=16, learning_rate=0.001):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Models will be saved to: {save_dir}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Basic data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Data preparation
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DRDataset(train_df, img_dir, transform=transform)
    val_dataset = DRDataset(val_df, img_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = HybridUNetCNN().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update statistics
                current_loss = loss.item()
                running_loss += current_loss
                current_acc = 100 * correct / total
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
                # Memory management
                del outputs, loss
                inputs = inputs.cpu()
                labels = labels.cpu()
                torch.cuda.empty_cache()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Memory management
                    del outputs, loss
                    inputs = inputs.cpu()
                    labels = labels.cpu()
                    torch.cuda.empty_cache()
            
            avg_val_loss = val_loss/len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Print epoch statistics
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print(f'Training Loss: {running_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Validation Accuracy: {val_acc:.2f}%\n')
            
            # Early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(save_dir, 'best_model1.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Saved new best model with validation loss: {avg_val_loss:.4f}')
            
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Training interrupted: {e}")
        emergency_path = os.path.join(save_dir, 'emergency_save.pth')
        torch.save(model.state_dict(), emergency_path)
        print(f"Emergency save created at: {emergency_path}")
        
    return model

if __name__ == "__main__":
    # Updated paths to use full dataset
    CSV_PATH = r"K:\MajorProject\Execution\archive\trainLabels.csv"  # Original CSV with all 35k images
    IMG_DIR = r"K:\MajorProject\Execution\Phase 2\p2_op1"
    SAVE_DIR = r"K:\MajorProject\Execution\Phase 3\final"
    
    # Memory management settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    try:
        # Using default num_epochs=20
        model = train_model(CSV_PATH, IMG_DIR, SAVE_DIR, num_epochs=20, batch_size=16)
    except Exception as e:
        print(f"Error: {e}")





