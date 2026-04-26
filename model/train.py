"""
model/train.py — Foundation CNN Training for Project Silicon Mind

Trains a hardware-friendly CNN on MNIST with Quantization-Aware Training (QAT)
stubs for seamless transition to INT8 quantization.

Architecture constraints (per docs/interface_spec.md):
    - No BatchNorm (complex in fixed-point hardware)
    - Signed INT8 weights/activations (-128 to +127)
    - INT32 accumulators inside each PE
    - Output channels are multiples of ARRAY_SIZE (4) for systolic tiling
    - QuantStub/DeQuantStub bracket the forward pass for QAT readiness
    - Explicit bias=True on all layers (interface_spec.md §1: INT32 bias)

Cross-reference:
    - docs/interface_spec.md §1 (Data Types & Widths)
    - docs/interface_spec.md §3 (Weight File Format)
    - docs/interface_spec.md §5 (Output Format: INT32 → requantized INT8)
    - docs/interface_spec.md §8 (Layer Execution Order)

Author: Aaditya Sood
Date: 2026-04-18
"""

import json
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.ao.quantization as quant
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Hardware & Training Constants ───────────────────────────────────────────
# These constants align with docs/interface_spec.md
INPUT_CHANNELS = 1       # MNIST: single grayscale channel
NUM_CLASSES = 10         # MNIST: digits 0-9
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
RANDOM_SEED = 42
IMAGE_SIZE = 28          # MNIST image dimensions

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Path management: execution assumed from project root
DATA_DIR = "data"
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "mnist_cnn.pth")
MODEL_META_PATH = os.path.join(DATA_DIR, "mnist_cnn_meta.json")

# ─── CNN Architecture Constants ─────────────────────────────────────────────
# Output channels: multiples of ARRAY_SIZE=4 for clean systolic tiling.
# NOTE: FC1 input dim (CONV2_OUT_CHANNELS * 7 * 7 = 1568) is NOT a multiple of
# ARRAY_SIZE=4. The tiling controller will need to pad partial tiles for FC
# layers. This is handled in the export/driver, not here.
CONV1_OUT_CHANNELS = 16
CONV2_OUT_CHANNELS = 32
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1              # padding=1 with kernel=3 preserves spatial dims
POOL_SIZE = 2
FC1_OUT_FEATURES = 128

# Image dimension flow (MNIST 28×28):
#   Conv1 (k=3, s=1, p=1) → 28×28  →  Pool1 (k=2, s=2) → 14×14
#   Conv2 (k=3, s=1, p=1) → 14×14  →  Pool2 (k=2, s=2) → 7×7
FINAL_MAP_SIZE = 7


def _compute_spatial_size(image_size: int) -> int:
    """Programmatically compute the spatial size after conv+pool layers.

    This prevents silent shape mismatches if KERNEL_SIZE, PADDING, STRIDE,
    or POOL_SIZE are ever changed. (GLM Audit Issue #1)

    Returns:
        The spatial dimension after both conv+pool stages.
    """
    # Conv1: preserves spatial (padding=1, kernel=3, stride=1)
    size = (image_size + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    # Pool1: halves spatial
    size = size // POOL_SIZE
    # Conv2: preserves spatial
    size = (size + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    # Pool2: halves spatial
    size = size // POOL_SIZE
    return size


class SiliconMindCNN(nn.Module):
    """Hardware-friendly CNN for MNIST classification.

    Architecture (per docs/interface_spec.md §8):
        Input [B, 1, 28, 28]
        → Conv1 (3×3, 16 filters, bias=True) → ReLU → MaxPool (2×2)
        → Conv2 (3×3, 32 filters, bias=True) → ReLU → MaxPool (2×2)
        → Flatten → FC1 (1568→128, bias=True) → ReLU
        → FC2 (128→10, bias=True)

    QAT readiness (per interface_spec.md §1):
        QuantStub/DeQuantStub bracket the forward pass so that
        torch.ao.quantization.prepare_qat() can insert fake-quantize
        observers without modifying this class.

    Constraints:
        - No BatchNorm (hard to implement in weight-stationary systolic array)
        - Explicit bias=True (interface_spec.md §1: INT32 bias per PE)
        - All channel dims are multiples of ARRAY_SIZE=4
    """

    def __init__(self) -> None:
        super().__init__()

        # Validate FINAL_MAP_SIZE matches actual conv/pool arithmetic
        computed = _compute_spatial_size(IMAGE_SIZE)
        assert computed == FINAL_MAP_SIZE, (
            f"FINAL_MAP_SIZE={FINAL_MAP_SIZE} does not match computed spatial "
            f"size={computed}. Check KERNEL_SIZE, PADDING, STRIDE, POOL_SIZE."
        )

        # QAT stubs — required for torch.ao.quantization workflow
        # (Pro Audit Issue #2, interface_spec.md §1)
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # Layer 1: Feature Extraction
        # Input: [B, 1, 28, 28] → Output: [B, 16, 14, 14]
        self.conv1 = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=CONV1_OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=True  # interface_spec.md §1: INT32 bias
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)

        # Layer 2: Feature Extraction
        # Input: [B, 16, 14, 14] → Output: [B, 32, 7, 7]
        self.conv2 = nn.Conv2d(
            in_channels=CONV1_OUT_CHANNELS,
            out_channels=CONV2_OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=True  # interface_spec.md §1: INT32 bias
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)

        # Classification Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=CONV2_OUT_CHANNELS * FINAL_MAP_SIZE * FINAL_MAP_SIZE,
            out_features=FC1_OUT_FEATURES,
            bias=True  # interface_spec.md §1: INT32 bias
        )
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=FC1_OUT_FEATURES,
            out_features=NUM_CLASSES,
            bias=True  # interface_spec.md §1: INT32 bias
        )

        # Runtime validation: dummy forward to confirm flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            dummy = self.pool1(self.relu1(self.conv1(dummy)))
            dummy = self.pool2(self.relu2(self.conv2(dummy)))
            actual_flatten = dummy.numel()
            expected_flatten = CONV2_OUT_CHANNELS * FINAL_MAP_SIZE * FINAL_MAP_SIZE
            assert actual_flatten == expected_flatten, (
                f"Flatten size mismatch: actual={actual_flatten}, "
                f"expected={expected_flatten}. Architecture constants are inconsistent."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Expected input shape: [B, 1, 28, 28] (batch of grayscale MNIST images)
        Output shape: [B, 10] (logits for 10 digit classes)

        QuantStub converts FP32 → fake-quantized INT8 at the input.
        DeQuantStub converts fake-quantized → FP32 at the output.
        During normal FP32 training these are identity ops.

        Cross-ref: docs/interface_spec.md §8 (Layer Execution Order)
        """
        x = self.quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Download and prepare MNIST train and test DataLoaders.

    Applies standard MNIST normalization. Uses a seeded Generator
    for reproducible batch ordering.

    Note: The test set is used for per-epoch evaluation. This is acceptable
    for Phase 1 (MNIST baseline). A proper validation split should be
    introduced if early stopping or hyperparameter tuning is added later.

    Args:
        batch_size: Number of images per batch.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
) -> Tuple[float, Dict]:
    """Standard PyTorch training loop with per-epoch evaluation.

    Saves the model state dict from the epoch with the highest test accuracy
    (not necessarily the last epoch) to avoid saving an overfitted checkpoint.

    Args:
        model: The CNN to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target device (cuda/cpu).
        epochs: Number of training epochs.

    Returns:
        Tuple of (best_accuracy, training_metadata_dict).
    """
    best_accuracy = 0.0
    best_state_dict = None
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        # ── Training Phase ──
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Guard against empty DataLoader (GLM Audit Issue #8)
        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # ── Evaluation Phase ──
        accuracy = evaluate_model(model, test_loader, device, epoch + 1)

        # Save best model checkpoint (GLM Audit Issue #5)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        history.append({"epoch": epoch + 1, "avg_loss": avg_loss, "test_acc": accuracy})
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}% ---")

    # Restore best model weights into the model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metadata = {
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "history": history,
    }
    return best_accuracy, metadata


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epoch: Optional[int] = None,
) -> float:
    """Evaluate the model on the test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Target device.
        epoch: If provided, printed alongside accuracy for clarity.

    Returns:
        Test accuracy as a percentage (0–100).
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

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    label = f"Epoch {epoch} " if epoch is not None else "Final "
    print(f"{label}Test Accuracy: {accuracy:.2f}%")
    return accuracy


def test_architecture() -> None:
    """Verify model output shapes, layer existence, and QAT stub presence.

    Runs under torch.no_grad() with model.eval() to prevent unnecessary
    graph construction and ensure correct behavior if stateful layers
    (e.g., Dropout) are ever added. (GLM Audit Issue #4)

    Cross-ref: docs/interface_spec.md §1 (Data Types),
               docs/interface_spec.md §8 (Layer Order)
    """
    print("\n[VERIFICATION] Testing model architecture...")
    model = SiliconMindCNN()
    model.eval()

    with torch.no_grad():
        # Test 1: Forward pass produces correct output shape
        # Expected: [B=1, 1, 28, 28] → [B=1, 10]
        test_input = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        output = model(test_input)
        assert output.shape == (1, NUM_CLASSES), (
            f"Output shape mismatch. Expected (1, {NUM_CLASSES}), got {output.shape}"
        )

        # Test 2: Required layers exist
        layer_names = dict(model.named_modules())
        for name in ("conv1", "conv2", "fc1", "fc2"):
            assert name in layer_names, f"Required layer '{name}' missing from model."

        # Test 3: QAT stubs are present (Pro Audit Issue #2)
        assert hasattr(model, "quant") and isinstance(model.quant, quant.QuantStub), (
            "QuantStub missing — required for INT8 QAT pipeline."
        )
        assert hasattr(model, "dequant") and isinstance(model.dequant, quant.DeQuantStub), (
            "DeQuantStub missing — required for INT8 QAT pipeline."
        )

        # Test 4: Bias enabled on all conv/linear layers (interface_spec.md §1)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert module.bias is not None, (
                    f"Layer '{name}' has bias=False. interface_spec.md §1 requires "
                    f"INT32 bias for hardware PE accumulation."
                )

        # Test 5: Post-ReLU activations are non-negative (Kimi Audit Issue #1)
        x = test_input
        x = model.quant(x)
        x = model.relu1(model.conv1(x))
        assert x.min() >= 0.0, f"Post-ReLU1 has negative values: min={x.min().item()}"
        x = model.pool1(x)
        x = model.relu2(model.conv2(x))
        assert x.min() >= 0.0, f"Post-ReLU2 has negative values: min={x.min().item()}"

    print("[VERIFICATION] All architecture tests passed.\n")


def _preflight_checks() -> None:
    """Verify environment is ready before starting training.

    Checks:
        - Data directory exists (creates if needed)
        - Data directory is writable (Kimi Audit Issue #5)

    Raises:
        PermissionError: If the data directory is not writable.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory at {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)

    if not os.access(DATA_DIR, os.W_OK):
        raise PermissionError(
            f"Data directory '{DATA_DIR}' is not writable. "
            f"Cannot save model weights after training."
        )


def _save_model_with_metadata(
    model: nn.Module,
    metadata: Dict,
    lr: float,
    batch_size: int,
) -> None:
    """Save model weights and a JSON metadata sidecar.

    The sidecar file records training configuration, accuracy, and timestamps
    for traceability. (Kimi Audit Issue #4)

    Args:
        model: Trained model (should contain the best-epoch state dict).
        metadata: Training metadata from train_model().
        lr: Learning rate used.
        batch_size: Batch size used.
    """
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")

    meta = {
        "model": "SiliconMindCNN",
        "dataset": "MNIST",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "learning_rate": lr,
        "batch_size": batch_size,
        "random_seed": RANDOM_SEED,
        "architecture": {
            "conv1_out": CONV1_OUT_CHANNELS,
            "conv2_out": CONV2_OUT_CHANNELS,
            "fc1_out": FC1_OUT_FEATURES,
            "kernel_size": KERNEL_SIZE,
            "num_classes": NUM_CLASSES,
        },
        **metadata,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved to {MODEL_META_PATH}")


def main(
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Orchestrate the full training pipeline.

    Steps:
        1. Set deterministic seeds (including cuDNN flags)
        2. Pre-flight environment checks
        3. Architecture verification
        4. Train with per-epoch evaluation
        5. Save best-epoch model + metadata sidecar
    """
    print("=== Silicon Mind: MNIST CNN Training ===")
    print(f"    Epochs: {epochs} | LR: {lr} | Batch Size: {batch_size}")

    # ── Reproducibility: full determinism (GLM Audit Issue #2, Kimi #3) ──
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Pre-flight checks (Kimi Audit Issue #5) ──
    _preflight_checks()

    # ── Architecture sanity ──
    test_architecture()

    # ── Device selection ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # ── Initialize model, loss, optimizer ──
    model = SiliconMindCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Load dataset ──
    train_loader, test_loader = get_dataloaders(batch_size)

    # ── Train ──
    print(f"Starting training for {epochs} epochs...")
    best_accuracy, metadata = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs
    )

    # ── Persist best model + metadata ──
    _save_model_with_metadata(model, metadata, lr, batch_size)

    # ── Final verdict ──
    if best_accuracy >= 98.0:
        print(f"\n✅ Success: Model achieved {best_accuracy:.2f}% accuracy (target: >98%).")
        print(f"   Best epoch: {metadata['best_epoch']}/{metadata['total_epochs']}")
    else:
        print(
            f"\n⚠️  Model accuracy {best_accuracy:.2f}% is below 98% target. "
            f"Try: --epochs 10 or --lr 0.0005"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or test the Silicon Mind MNIST CNN."
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only run the architecture verification tests."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    args = parser.parse_args()

    if args.test_only:
        test_architecture()
    else:
        main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
