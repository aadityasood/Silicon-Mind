"""
model/train.py
Foundation CNN training script for Project Silicon Mind.

This script trains a simple CNN on the MNIST dataset using vanilla PyTorch.
The model is designed to be hardware-friendly:
1. No Batch Normalization (complex to implement in fixed-point HW).
2. Standard Conv2d, ReLU, and MaxPool2d layers.
3. Bit-width compatibility for later INT8 quantization.

The model is trained for 5 epochs to reach >98% accuracy.
Final weights are saved to data/mnist_cnn.pth.

Author: Flash (Silicon Mind Team)
Auditor: Pro (Silicon Mind Team)
Date: 2026-04-18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from typing import Tuple

# --- Hardware & Training Constants ---
# These constants align with the HW-SW interface specification
INPUT_CHANNELS = 1
NUM_CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_SIZE = 28
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Path Management: Assumes execution from project root
# data/ is used for datasets and model checkpoints
DATA_DIR = "data"
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "mnist_cnn.pth")

# --- CNN Architecture Constants ---
# Note: Output channels are chosen to be hardware-friendly (powers of 2 or multiples of ARRAY_SIZE=4)
CONV1_OUT_CHANNELS = 16
CONV2_OUT_CHANNELS = 32
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
POOL_SIZE = 2
FC1_OUT_FEATURES = 128

# Image dimension calculation: MNIST is 28x28
# After Conv1 + Pool1 (28 -> 28 -> 14)
# After Conv2 + Pool2 (14 -> 14 -> 7)
FINAL_MAP_SIZE = 7

class SiliconMindCNN(nn.Module):
    """
    A hardware-friendly CNN for MNIST classification.
    
    Architecture:
    Conv1 (3x3) -> ReLU -> MaxPool (2x2)
    Conv2 (3x3) -> ReLU -> MaxPool (2x2)
    FC1 (Linear) -> ReLU
    FC2 (Linear Output)
    
    This architecture avoids BatchNorm and complex layers to ensure
    easy mapping to the systolic array in hardware.
    """
    def __init__(self) -> None:
        super(SiliconMindCNN, self).__init__()
        
        # Layer 1: Feature Extraction
        # Input: [1, 28, 28] | Output: [16, 14, 14]
        self.conv1 = nn.Conv2d(
            in_channels=INPUT_CHANNELS, 
            out_channels=CONV1_OUT_CHANNELS, 
            kernel_size=KERNEL_SIZE, 
            stride=STRIDE, 
            padding=PADDING
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
        
        # Layer 2: Feature Extraction
        # Input: [16, 14, 14] | Output: [32, 7, 7]
        self.conv2 = nn.Conv2d(
            in_channels=CONV1_OUT_CHANNELS, 
            out_channels=CONV2_OUT_CHANNELS, 
            kernel_size=KERNEL_SIZE, 
            stride=STRIDE, 
            padding=PADDING
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
        
        # Classification Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=CONV2_OUT_CHANNELS * FINAL_MAP_SIZE * FINAL_MAP_SIZE, 
            out_features=FC1_OUT_FEATURES
        )
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=FC1_OUT_FEATURES, 
            out_features=NUM_CLASSES
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference/training.
        """
        # Sequential application of layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Download and prepare MNIST train and test dataloaders.
    
    Includes standard normalization for MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    
    # Download data if not present in DATA_DIR
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> None:
    """
    Standard PyTorch training loop for a fixed number of epochs.
    """
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Periodic logging
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f} ---")

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the test set to determine accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    return accuracy

def test_architecture() -> None:
    """
    Verify model output shapes and architecture logic.
    """
    print("\n[VERIFICATION] Testing model architecture...")
    model = SiliconMindCNN()
    
    # Test 1: Single image forward pass
    # Expected input: [Batch=1, Channel=1, H=28, W=28]
    test_input = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    output = model(test_input)
    
    # Test 2: Output dimension check
    assert output.shape == (1, NUM_CLASSES), f"Error: Output shape mismatch. Expected (1, {NUM_CLASSES}), got {output.shape}"
    
    # Test 3: Layer existence (Sanity check for required layers)
    layers = dict(model.named_modules())
    assert 'conv1' in layers and 'conv2' in layers, "Error: Convolutional layers missing."
    assert 'fc1' in layers and 'fc2' in layers, "Error: Fully connected layers missing."
    
    print("[VERIFICATION] All sanity tests passed.\n")

def main() -> None:
    """
    Orchestrate the training process.
    """
    print("=== Silicon Mind: MNIST CNN Training ===")
    
    # Sanity check architecture first
    test_architecture()
    
    # Ensure environment is ready
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory at {DATA_DIR}...")
        os.makedirs(DATA_DIR)
        
    # Select best available hardware (CUDA/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")
    
    # Initialize components
    model = SiliconMindCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load dataset
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    # Training phase
    print(f"Starting training for {EPOCHS} epochs...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Evaluation phase
    print("Evaluating final model performance...")
    accuracy = evaluate_model(model, test_loader, device)
    
    # Persistence
    print(f"Saving model weights to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Final health check
    if accuracy >= 98.0:
        print("Success: Model exceeds 98% accuracy target.")
    else:
        print("Note: Model accuracy below target. Consider adjusting learning rate or epochs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Silicon Mind MNIST CNN.")
    parser.add_argument('--test-only', action='store_true', help='Only run the architecture tests')
    args = parser.parse_args()
    
    if args.test_only:
        test_architecture()
    else:
        main()
