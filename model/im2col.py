"""
model/im2col.py — Image-to-Column Transformation for Silicon Mind

Implements the im2col transformation that maps 2D convolutions onto the
systolic array as matrix multiplications.

Transformation:
    - Weights: [C_out, C_in, K_h, K_w] → [C_out, C_in * K_h * K_w]
    - Input:   [C, H, W]               → [C * K_h * K_w, H_out * W_out]
    - Conv becomes: Output = W_col @ X_col  (matmul on systolic array)

Hardware constraints:
    - INT8 precision only (signed, two's complement)
    - Row-major output layout (per interface_spec.md §3, §4)
    - All values must be in [-128, +127] range

Cross-reference:
    - docs/interface_spec.md §3 (Weight File Format)
    - docs/interface_spec.md §4 (Input Activation Format)
    - docs/interface_spec.md §7 (Tiling Strategy)

Author: Aaditya Sood
Date: 2026-04-26
"""

import numpy as np
from typing import Tuple

# ─── Constants ───────────────────────────────────────────────────────────────
ARRAY_SIZE = 4  # Systolic array dimension (per interface_spec.md §2)


def im2col(
    input_data: np.ndarray,
    kernel_size: int,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Perform im2col transformation on a single input activation map.

    Extracts all receptive-field patches from the input and arranges them
    as columns of a matrix, converting convolution into matrix multiplication.

    CRITICAL: This function operates on ACTUAL INPUT DATA, not shape tuples.
    Passing a shape tuple instead of an ndarray is a bug that will produce
    all-zero outputs.

    Args:
        input_data: Input activation map of shape (C, H, W), dtype int8.
        kernel_size: Spatial dimension of the square kernel.
        padding: Zero-padding added to both sides of the spatial dims.
        stride: Stride of the convolution.

    Returns:
        Matrix of shape (C * K * K, H_out * W_out) where each column
        is a flattened receptive field. dtype matches input.

    Raises:
        ValueError: If input_data is not a 3D array.

    Cross-ref: docs/interface_spec.md §4 (Input Activation Format)
    """
    if input_data.ndim != 3:
        raise ValueError(
            f"im2col expects 3D input (C, H, W), got shape {input_data.shape}. "
            f"If you have a batch dim, index into it first: input_data[0]."
        )

    C, H, W = input_data.shape

    # Output spatial dimensions
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    # Pad the input (zero-padding matches hardware behavior)
    if padding > 0:
        img_padded = np.pad(
            input_data,
            ((0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        img_padded = input_data

    # Pre-allocate output matrix
    # Shape: [C * K * K, H_out * W_out]
    col = np.zeros(
        (C * kernel_size * kernel_size, H_out * W_out),
        dtype=input_data.dtype,
    )

    # Extract patches — this loop structure mirrors how hardware reads
    # memory strides for systolic feeding
    col_idx = 0
    for i in range(0, H + 2 * padding - kernel_size + 1, stride):
        for j in range(0, W + 2 * padding - kernel_size + 1, stride):
            patch = img_padded[:, i : i + kernel_size, j : j + kernel_size]
            col[:, col_idx] = patch.flatten()
            col_idx += 1

    return col


def weight2col(weight_tensor: np.ndarray) -> np.ndarray:
    """Reshape convolution weights for systolic matrix multiplication.

    Args:
        weight_tensor: Weight tensor of shape (C_out, C_in, K_h, K_w).

    Returns:
        Matrix of shape (C_out, C_in * K_h * K_w).

    Raises:
        ValueError: If weight tensor is not 4D.

    Cross-ref: docs/interface_spec.md §3 (Weight File Format — row-major)
    """
    if weight_tensor.ndim != 4:
        raise ValueError(
            f"weight2col expects 4D tensor (C_out, C_in, K_h, K_w), "
            f"got shape {weight_tensor.shape}."
        )
    C_out, C_in, K_h, K_w = weight_tensor.shape
    return weight_tensor.reshape(C_out, C_in * K_h * K_w)


def verify_im2col() -> None:
    """Verify im2col with known test cases including negative INT8 values.

    Tests the ACTUAL im2col function (not a local copy).
    Includes negative INT8 values to exercise two's complement paths.

    Cross-ref: docs/interface_spec.md §1 (signed INT8: -128 to +127)
    """
    print("[VERIFICATION] Testing im2col implementation...")

    # ── Test 1: Basic 3×3 image, 2×2 kernel, no padding ──
    # Image (1 channel):
    #   1  2  3
    #   4  5  6
    #   7  8  9
    img1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int8)

    result1 = im2col(img1, kernel_size=2, padding=0, stride=1)

    # Expected: 4 patches of size 4 each → shape (4, 4)
    assert result1.shape == (4, 4), f"Shape mismatch: expected (4, 4), got {result1.shape}"

    # Patch 0 (top-left 2×2): [1, 2, 4, 5]
    expected_patch0 = np.array([1, 2, 4, 5], dtype=np.int8)
    assert np.array_equal(result1[:, 0], expected_patch0), (
        f"Patch 0 mismatch: expected {expected_patch0}, got {result1[:, 0]}"
    )

    # Patch 3 (bottom-right 2×2): [5, 6, 8, 9]
    expected_patch3 = np.array([5, 6, 8, 9], dtype=np.int8)
    assert np.array_equal(result1[:, 3], expected_patch3), (
        f"Patch 3 mismatch: expected {expected_patch3}, got {result1[:, 3]}"
    )

    print("  Test 1 (basic 3×3, k=2, p=0) passed.")

    # ── Test 2: Negative INT8 values ──
    # Tests two's complement behavior critical for hardware
    img2 = np.array([[[-128, 127, -1], [0, 1, -2], [64, -64, 0]]], dtype=np.int8)

    result2 = im2col(img2, kernel_size=2, padding=0, stride=1)

    assert result2.shape == (4, 4), f"Shape mismatch: expected (4, 4), got {result2.shape}"

    # Patch 0: [-128, 127, 0, 1]
    expected_neg_patch = np.array([-128, 127, 0, 1], dtype=np.int8)
    assert np.array_equal(result2[:, 0], expected_neg_patch), (
        f"Negative patch mismatch: expected {expected_neg_patch}, got {result2[:, 0]}"
    )

    print("  Test 2 (negative INT8 values) passed.")

    # ── Test 3: Padding=1 preserves spatial dimensions ──
    img3 = np.ones((1, 4, 4), dtype=np.int8) * 5
    result3 = im2col(img3, kernel_size=3, padding=1, stride=1)

    # With p=1, k=3, s=1: H_out = (4+2-3)//1+1 = 4, W_out = 4
    assert result3.shape == (9, 16), f"Shape mismatch: expected (9, 16), got {result3.shape}"

    print("  Test 3 (padding=1 preserves spatial) passed.")

    # ── Test 4: Multi-channel ──
    img4 = np.arange(2 * 3 * 3, dtype=np.int8).reshape(2, 3, 3)
    result4 = im2col(img4, kernel_size=2, padding=0, stride=1)

    # 2 channels × 2×2 kernel = 8 rows, 2×2 output = 4 columns
    assert result4.shape == (8, 4), f"Shape mismatch: expected (8, 4), got {result4.shape}"

    print("  Test 4 (multi-channel) passed.")

    # ── Test 5: weight2col ──
    weights = np.arange(16 * 1 * 3 * 3, dtype=np.int8).reshape(16, 1, 3, 3)
    w_col = weight2col(weights)
    assert w_col.shape == (16, 9), f"weight2col shape mismatch: expected (16, 9), got {w_col.shape}"

    print("  Test 5 (weight2col) passed.")

    # ── Test 6: Error handling ──
    try:
        im2col(np.zeros((3, 3), dtype=np.int8), kernel_size=2)
        assert False, "Should have raised ValueError for 2D input"
    except ValueError:
        pass

    print("  Test 6 (error handling) passed.")

    print("[VERIFICATION] ✅ All im2col tests passed.\n")


if __name__ == "__main__":
    verify_im2col()
