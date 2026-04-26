"""
model/quantize.py — INT8 Quantization-Aware Training Pipeline for Silicon Mind

Performs QAT fine-tuning on a pretrained MNIST model to minimize the accuracy
drop when converting from FP32 to signed INT8.

Pipeline:
    1. Load pretrained FP32 model (from train.py)
    2. Evaluate FP32 baseline accuracy
    3. Fuse Conv+ReLU / Linear+ReLU modules
    4. Insert fake-quantization observers (prepare_qat)
    5. Fine-tune for QAT_EPOCHS with fake-quant in the loop
    6. Convert to truly quantized INT8 model
    7. Verify <2% accuracy drop (PROJECT_CONTEXT.md performance targets)

Hardware constraints (per docs/interface_spec.md §1):
    - Precision: Signed INT8 (-128 to 127) for weights and activations
    - Accumulator: INT32 for partial sums
    - Symmetric quantization for weights (zero-point = 0)
    - Asymmetric quantization for activations (post-ReLU: 0 to 127)

Cross-reference:
    - docs/interface_spec.md §1 (Data Types & Widths)
    - docs/interface_spec.md §5 (Output Format: INT32 → requantized INT8)
    - PROJECT_CONTEXT.md: MNIST INT8 accuracy >96% (<2% drop from FP32)

Author: Aaditya Sood
Date: 2026-04-26
"""

import os
from typing import Optional, Tuple

import torch
import torch.ao.quantization as quant
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.train import (
    SiliconMindCNN,
    get_dataloaders,
    evaluate_model,
    DATA_DIR,
    MODEL_SAVE_PATH,
    RANDOM_SEED,
)

# ─── Quantization Constants ──────────────────────────────────────────────────
QUANT_MODEL_SAVE_PATH = os.path.join(DATA_DIR, "mnist_cnn_quant.pth")
QAT_EPOCHS = 2
QAT_LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# Backend: 'fbgemm' for x86 QAT simulation (standard PyTorch backend)
QUANT_BACKEND = "fbgemm"

# Module fusion groups matching SiliconMindCNN layer names
# No BatchNorm in our architecture — only Conv+ReLU and Linear+ReLU fusions.
_FUSION_GROUPS = [
    ["conv1", "relu1"],
    ["conv2", "relu2"],
    ["fc1", "relu3"],
]


def _set_full_determinism() -> None:
    """Set all seeds and flags for full reproducibility.

    Mirrors train.py determinism block. Required for bit-exact QAT results.
   
    """
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained_model(device: torch.device) -> nn.Module:
    """Load the pretrained FP32 model weights.

    Args:
        device: Target device for the model.

    Returns:
        Model with loaded weights in eval mode.

    Raises:
        FileNotFoundError: If pretrained weights don't exist.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Pretrained model not found at {MODEL_SAVE_PATH}. "
            "Run 'python -m model.train' first."
        )

    model = SiliconMindCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded pretrained model from {MODEL_SAVE_PATH}")
    return model


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """Prepare the model for Quantization-Aware Training.

    Steps:
        1. Set quantization backend
        2. Assign qconfig (symmetric weights, asymmetric activations)
        3. Fuse Conv+ReLU / Linear+ReLU modules
        4. Insert fake-quantization observers via prepare_qat

    Args:
        model: Pretrained FP32 model (in eval mode).

    Returns:
        Model ready for QAT fine-tuning (in train mode).

    Cross-ref: docs/interface_spec.md §1 (Data Types)
    """
    torch.backends.quantized.engine = QUANT_BACKEND

    model.qconfig = quant.get_default_qat_qconfig(QUANT_BACKEND)

    model = quant.fuse_modules(model, _FUSION_GROUPS)

    # prepare_qat requires train mode
    model.train()
    model = quant.prepare_qat(model)
    print("Model prepared for QAT (modules fused, fake-quantizers inserted)")

    return model


def fine_tune_qat(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Fine-tune the model with fake-quantization in the training loop.

    CRITICAL: After calling evaluate_model() (which sets model.eval()),
    we must call model.train() before the next epoch. Otherwise fake-quant
    observers freeze and weights stop learning.

    Args:
        model: Model in QAT mode (train mode).
        train_loader: Training data.
        test_loader: Evaluation data.
        device: CPU or CUDA.

    Returns:
        Best test accuracy achieved during QAT fine-tuning.
    """
    optimizer = optim.Adam(model.parameters(), lr=QAT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    print(f"Starting QAT fine-tuning for {QAT_EPOCHS} epochs...")
    model.train()

    for epoch in range(QAT_EPOCHS):
        # ── Training phase ──
        model.train()  # Explicit: undo any eval() from previous epoch's evaluation
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 200 == 0:
                print(
                    f"QAT Epoch [{epoch+1}/{QAT_EPOCHS}], "
                    f"Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # ── Evaluation phase ──
        # evaluate_model() calls model.eval() internally
        accuracy = evaluate_model(model, test_loader, device, epoch + 1)
        best_accuracy = max(best_accuracy, accuracy)

        print(f"--- QAT Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ---")

        # CRITICAL: Restore train mode for next epoch
        model.train()

    return best_accuracy


def finalize_quantization(model: nn.Module) -> nn.Module:
    """Convert the fake-quantized model to a truly quantized INT8 model.

    Must be called with the model on CPU and in eval mode.

    Args:
        model: Fine-tuned QAT model.

    Returns:
        Quantized model with real INT8 weights.

    Cross-ref: docs/interface_spec.md §1 (INT8 weights/activations)
    """
    model.eval()
    model.cpu()
    quantized_model = quant.convert(model, inplace=False)
    print("Model converted to quantized INT8 format.")
    return quantized_model


def verify_quantized_model(
    quantized_model: nn.Module,
    test_loader: DataLoader,
    fp32_accuracy: float,
) -> float:
    """Evaluate the final quantized model and check accuracy drop.

    Args:
        quantized_model: INT8 quantized model.
        test_loader: Test data.
        fp32_accuracy: Baseline FP32 accuracy for drop comparison.

    Returns:
        Quantized model accuracy percentage.
    """
    print("\n[VERIFICATION] Evaluating final quantized INT8 model...")
    accuracy = evaluate_model(quantized_model, test_loader, torch.device("cpu"))

    drop = fp32_accuracy - accuracy
    print(f"FP32 baseline: {fp32_accuracy:.2f}% | INT8: {accuracy:.2f}% | Drop: {drop:.2f}%")

    if drop > 2.0:
        print(
            f"⚠️  Accuracy drop {drop:.2f}% exceeds 2% target. "
            f"Consider: more QAT epochs, lower LR, or adjusted qconfig."
        )

    return accuracy


def main() -> None:
    """Orchestrate the full quantization pipeline.

    Steps:
        1. Set full determinism
        2. Load data
        3. Load pretrained FP32 model
        4. Evaluate FP32 baseline (for drop comparison)
        5. Prepare QAT → fine-tune → convert to INT8
        6. Verify quantized accuracy
        7. Save quantized model
    """
    print("=== Silicon Mind: INT8 Quantization Pipeline ===")

    # 1. Full determinism
    _set_full_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 2. Load data
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # 3. Load pretrained model
    model = load_pretrained_model(device)

    # 4. FP32 baseline accuracy
    print("\n[BASELINE] Evaluating pretrained FP32 model...")
    fp32_accuracy = evaluate_model(model, test_loader, device)
    print(f"FP32 baseline accuracy: {fp32_accuracy:.2f}%")

    # 5. Prepare for QAT
    model = prepare_qat_model(model)

    # 6. Fine-tune with fake-quantization
    qat_accuracy = fine_tune_qat(model, train_loader, test_loader, device)

    # 7. Convert to true INT8
    model.cpu()
    quantized_model = finalize_quantization(model)

    # 8. Verify
    final_accuracy = verify_quantized_model(quantized_model, test_loader, fp32_accuracy)

    # 9. Save
    torch.save(quantized_model.state_dict(), QUANT_MODEL_SAVE_PATH)
    print(f"Quantized model saved to {QUANT_MODEL_SAVE_PATH}")

    # 10. Final verdict
    if final_accuracy >= 96.0:
        print(f"\n✅ Success: Quantized model accuracy {final_accuracy:.2f}% meets target (>96%).")
    else:
        print(f"\n⚠️  Warning: Quantized model accuracy {final_accuracy:.2f}% is below 96% target.")


if __name__ == "__main__":
    main()
