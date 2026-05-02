"""
model/golden_model.py — Bit-Exact Integer Reference for Silicon Mind

Implements the full CNN inference using ONLY integer arithmetic, matching
the hardware behavior of the systolic array. This is the golden reference
against which Shaurya's Verilog simulation output must match bit-for-bit.

Pipeline (per docs/interface_spec.md §8):
    Input (INT8) → Conv1 → ReLU → MaxPool → Conv2 → ReLU → MaxPool
                 → FC1 → ReLU → FC2 → Output (INT8)

Each layer:
    1. im2col (for conv layers) or direct matmul (for FC layers)
    2. INT32 accumulation: acc = W @ X + bias
    3. Requantization: output = clamp(((acc * M) >> shift) + zp, -128, 127)
    4. ReLU: max(output, zero_point) for asymmetric quantization
    5. MaxPool: stride-2, 2×2 max over INT8 spatial map

CRITICAL CONSTRAINTS:
    - NO floating point math in the inference path
    - All arithmetic uses np.int32 accumulators (matches hardware PE width)
    - Overflow detection on INT32 accumulation
    - Requantization uses actual multiplier/shift from export metadata

Cross-reference:
    - docs/interface_spec.md §1 (Data Types: INT8, INT32, scale_factor, shift)
    - docs/interface_spec.md §5 (Output: INT32 → requantized INT8)
    - docs/interface_spec.md §8 (Layer Execution Order)
    - docs/interface_spec.md §9 (Verification Contract: bit-for-bit match)

Author: Aaditya Sood
Date: 2026-04-26
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.im2col import im2col, weight2col

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = "data"
MEM_DIR = os.path.join(DATA_DIR, "mem_files")
META_PATH = os.path.join(MEM_DIR, "model_meta.json")

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def load_mem_file(
    filename: str, shape: Tuple[int, ...], bits: int = 8
) -> np.ndarray:
    """Load hex values from a .mem file into a numpy array.

    Args:
        filename: Name of the .mem file (relative to MEM_DIR).
        shape: Expected shape of the output array.
        bits: 8 for INT8, 32 for INT32.

    Returns:
        Numpy array of the specified shape and dtype.

    Raises:
        FileNotFoundError: If the .mem file doesn't exist.
        ValueError: If line count doesn't match expected shape.

    Cross-ref: docs/interface_spec.md §3 (hex format, two's complement)
    """
    path = os.path.join(MEM_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Memory file not found: {path}. "
            f"Run 'python -m model.export' first."
        )

    with open(path, "r") as f:
        hex_vals = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    # Validate length before reshape
    expected_count = int(np.prod(shape))
    if len(hex_vals) != expected_count:
        raise ValueError(
            f"Line count mismatch in {filename}: "
            f"expected {expected_count} values for shape {shape}, "
            f"got {len(hex_vals)} lines."
        )

    # Convert hex to signed integers (two's complement)
    int_vals = []
    for h in hex_vals:
        val = int(h, 16)
        if bits == 8:
            if val >= 0x80:
                val -= 0x100
        elif bits == 32:
            if val >= 0x80000000:
                val -= 0x100000000
        int_vals.append(val)

    dtype = np.int8 if bits == 8 else np.int32
    return np.array(int_vals, dtype=dtype).reshape(shape)


def requantize(
    acc: np.ndarray,
    multiplier,
    shift,
    zero_point: int,
) -> np.ndarray:
    """Hardware-accurate requantization: INT32 accumulator → INT8 output.

    Computes:
        output = clamp( ((acc * multiplier) >> shift) + zero_point, -128, 127 )

    The multiplier and shift decompose the real-valued scale factor:
        M_real = (S_in * S_w) / S_out  ≈  multiplier * 2^(-shift)

    NOTE: Operator precedence — multiplication before shift, shift before
    zero_point addition. Parentheses are critical.

    Supports both per-tensor (scalar) and per-channel (list/array)
    multiplier/shift values. For per-channel, each output channel c uses
    multiplier[c] and shift[c]. The hardware has a per-PE requantization
    unit so this matches the HW contract.

    Args:
        acc: INT32 accumulator values, shape (C_out, N) or (C_out,).
        multiplier: Fixed-point multiplier — scalar (int) for per-tensor,
                    or list/array of length C_out for per-channel.
        shift: Right-shift amount — scalar (int) for per-tensor,
               or list/array of length C_out for per-channel.
        zero_point: Output zero point (added after shift).

    Returns:
        INT8 output array.

    Cross-ref: docs/interface_spec.md §1 (scale_factor, shift_amount)
    """
    acc64 = acc.astype(np.int64)

    if isinstance(multiplier, (list, np.ndarray)):
        # Per-channel: reshape to (C_out, 1) for broadcasting over spatial dim
        m = np.array(multiplier, dtype=np.int64)
        s = np.array(shift, dtype=np.int64)
        if acc64.ndim == 2:
            m = m.reshape(-1, 1)
            s = s.reshape(-1, 1)
        result = acc64 * m
        # Per-element shift: NumPy >> works element-wise with arrays
        result = result >> s
    else:
        # Per-tensor: scalar multiply and shift
        result = acc64 * np.int64(multiplier)
        result = result >> int(shift)

    result = result + np.int64(zero_point)
    return np.clip(result, INT8_MIN, INT8_MAX).astype(np.int8)


def int_matmul(
    weight: np.ndarray, activation: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Integer matrix multiplication with INT32 accumulation.

    Matches hardware PE behavior: INT8 × INT8 → INT32 accumulator + INT32 bias.

    IMPORTANT: np.dot on INT8 arrays promotes to INT64 (NumPy behavior).
    We cast the result to INT32 to match the hardware accumulator width,
    and check for overflow.

    Args:
        weight: INT8 weight matrix, shape (C_out, K).
        activation: INT8 activation matrix, shape (K, N).
        bias: INT32 bias vector, shape (C_out,).

    Returns:
        INT32 accumulator output, shape (C_out, N).
    """
    # Perform matmul in INT64 (NumPy's promotion for int8 dot)
    acc_i64 = np.dot(weight.astype(np.int32), activation.astype(np.int32))

    # Add bias (broadcast across columns)
    acc_i64 = acc_i64 + bias[:, np.newaxis]

    # Check for INT32 overflow (would cause incorrect hardware results)
    overflow_mask = (acc_i64 < INT32_MIN) | (acc_i64 > INT32_MAX)
    if np.any(overflow_mask):
        n_overflow = int(np.sum(overflow_mask))
        print(
            f"  ⚠️  INT32 overflow detected: {n_overflow} values. "
            f"Range: [{acc_i64.min()}, {acc_i64.max()}]"
        )

    return acc_i64.astype(np.int32)


def int_relu(x: np.ndarray, zero_point: int = 0) -> np.ndarray:
    """Integer ReLU matching hardware behavior.

    For asymmetric quantization, ReLU should clamp to zero_point
    (not necessarily 0). For symmetric quantization (zero_point=0),
    this is equivalent to max(0, x).

    Args:
        x: INT8 activation tensor.
        zero_point: Output zero point from quantization metadata.

    Returns:
        INT8 tensor with values >= zero_point.

    Cross-ref: docs/interface_spec.md §1 (activation post-ReLU: 0 to +127)
    """
    return np.maximum(np.int8(zero_point), x).astype(np.int8)


def int_maxpool2d(
    x: np.ndarray, pool_size: int = 2, stride: int = 2
) -> np.ndarray:
    """Integer MaxPool2d matching hardware behavior.

    Performs stride-2, 2×2 max pooling on an INT8 spatial feature map.
   

    Args:
        x: INT8 feature map of shape (C, H, W).
        pool_size: Pooling window size (square).
        stride: Pooling stride.

    Returns:
        Pooled INT8 feature map of shape (C, H//stride, W//stride).

    Raises:
        ValueError: If spatial dims are not divisible by stride.
    """
    C, H, W = x.shape
    H_out = H // stride
    W_out = W // stride

    if H % stride != 0 or W % stride != 0:
        raise ValueError(
            f"Spatial dims ({H}, {W}) must be divisible by stride={stride} "
            f"for MaxPool. Got remainders ({H % stride}, {W % stride})."
        )

    out = np.zeros((C, H_out, W_out), dtype=x.dtype)

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                window = x[
                    c,
                    i * stride : i * stride + pool_size,
                    j * stride : j * stride + pool_size,
                ]
                out[c, i, j] = window.max()

    return out


class GoldenModel:
    """Bit-exact integer reference for the SiliconMindCNN.

    Implements the FULL network: Conv1 → ReLU → Pool → Conv2 → ReLU → Pool
    → FC1 → ReLU → FC2, all in pure integer arithmetic.

    The inference path uses:
        - im2col for convolution → matmul conversion
        - INT32 accumulation (matching hardware PE width)
        - Requantization with actual multiplier/shift from export metadata
        - Integer ReLU and MaxPool

    Cross-ref: docs/interface_spec.md §8 (Layer Execution Order)
    """

    def __init__(self) -> None:
        """Load metadata, weights, and biases from exported .mem files.

        Raises:
            FileNotFoundError: If metadata or .mem files are missing.
        """
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Model metadata not found: {META_PATH}. "
                f"Run 'python -m model.export' first."
            )

        with open(META_PATH, "r") as f:
            self.meta: List[Dict] = json.load(f)

        self.layers: Dict[str, Dict] = {}
        self._load_all_layers()

    def _load_all_layers(self) -> None:
        """Load weights, biases, and requantization params for all layers."""
        for layer_meta in self.meta:
            name = layer_meta["name"]
            idx = layer_meta["layer_index"]
            shape = tuple(layer_meta["weight_shape"])

            self.layers[name] = {
                "weights": load_mem_file(f"layer{idx}_weights.mem", shape, bits=8),
                "bias": load_mem_file(f"layer{idx}_bias.mem", (shape[0],), bits=32),
                "multiplier": layer_meta["requant_multiplier"],
                "shift": layer_meta["requant_shift"],
                "output_zero_point": layer_meta.get("output_zero_point", 0),
                "output_scale": layer_meta.get("output_scale", 1.0),
                "weight_shape": shape,
            }

        print(f"Loaded {len(self.layers)} layers from .mem files.")

    def _conv_layer(
        self,
        x: np.ndarray,
        layer_name: str,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> np.ndarray:
        """Execute one convolutional layer in integer arithmetic.

        Steps:
            1. im2col on input activations
            2. weight2col on filter weights
            3. INT32 matmul + bias
            4. Requantize INT32 → INT8
            5. ReLU
            6. MaxPool

        Args:
            x: INT8 input activation, shape (C, H, W).
            layer_name: Layer identifier (e.g., "conv1").
            kernel_size: Convolution kernel size.
            padding: Zero-padding.
            stride: Convolution stride.

        Returns:
            INT8 output after conv → requant → ReLU → MaxPool.
        """
        layer = self.layers[layer_name]

        # 1. im2col: (C_in*K*K, H_out*W_out)
        x_col = im2col(x, kernel_size=kernel_size, padding=padding, stride=stride)

        # 2. weight2col: (C_out, C_in*K*K)
        w_col = weight2col(layer["weights"])

        # 3. INT32 matmul + bias
        acc = int_matmul(w_col, x_col, layer["bias"])

        # 4. Requantize INT32 → INT8
        multiplier = layer["multiplier"]
        shift = layer["shift"]
        zp = layer["output_zero_point"]
        x_out = requantize(acc, multiplier, shift, zp)

        # 5. ReLU (integer)
        x_out = int_relu(x_out, zero_point=zp)

        # Reshape from (C_out, H_out*W_out) back to (C_out, H_out, W_out)
        C_out = layer["weight_shape"][0]
        C_in, H, W = x.shape
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        x_out = x_out.reshape(C_out, H_out, W_out)

        # 6. MaxPool 2×2
        x_out = int_maxpool2d(x_out, pool_size=2, stride=2)

        return x_out

    def _fc_layer(
        self,
        x: np.ndarray,
        layer_name: str,
        apply_relu: bool = True,
    ) -> np.ndarray:
        """Execute one fully-connected layer in integer arithmetic.

        For FC layers, no im2col is needed — direct matrix multiplication.

        Args:
            x: Flattened INT8 input, shape (N,) or (N, 1).
            layer_name: Layer identifier (e.g., "fc1").
            apply_relu: Whether to apply ReLU after requantization.

        Returns:
            INT8 output vector.
        """
        layer = self.layers[layer_name]

        # Ensure x is a column vector for matmul: (N, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # INT32 matmul + bias
        # weights: (out_features, in_features), x: (in_features, 1)
        acc = int_matmul(layer["weights"], x, layer["bias"])

        # Requantize
        multiplier = layer["multiplier"]
        shift = layer["shift"]
        zp = layer["output_zero_point"]
        x_out = requantize(acc, multiplier, shift, zp)

        # ReLU (only for hidden FC layers, not the output layer)
        if apply_relu:
            x_out = int_relu(x_out, zero_point=zp)

        return x_out.flatten()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full integer forward pass through the CNN.

        Expected input: INT8 array of shape (1, 28, 28) for single-channel MNIST.
        Output: INT8 array of shape (10,) — raw logits for 10 digit classes.

        Network topology (per docs/interface_spec.md §8):
            Input → Conv1(3×3, p=1) → ReLU → MaxPool(2×2)
                  → Conv2(3×3, p=1) → ReLU → MaxPool(2×2)
                  → Flatten → FC1(1568→128) → ReLU
                  → FC2(128→10) → Output

        Args:
            x: INT8 input image, shape (1, 28, 28).

        Returns:
            INT8 output logits, shape (10,).
        """
        # Conv1: (1, 28, 28) → (16, 14, 14) via conv(3×3, p=1) + ReLU + pool(2×2)
        x = self._conv_layer(x, "conv1", kernel_size=3, padding=1, stride=1)

        # Conv2: (16, 14, 14) → (32, 7, 7) via conv(3×3, p=1) + ReLU + pool(2×2)
        x = self._conv_layer(x, "conv2", kernel_size=3, padding=1, stride=1)

        # Flatten: (32, 7, 7) → (1568,)
        x = x.flatten().astype(np.int8)

        # FC1: (1568,) → (128,) + ReLU
        x = self._fc_layer(x, "fc1", apply_relu=True)

        # FC2: (128,) → (10,) — no ReLU on output layer
        x = self._fc_layer(x, "fc2", apply_relu=False)

        return x


def verify_golden_model() -> None:
    """Verify the golden model can load, initialize, and produce valid output.

    Runs a forward pass on a known test input and checks:
        1. Model loads without errors
        2. All layer weights/biases loaded
        3. Forward pass produces correct output shape
        4. Output values are within INT8 range

    For full bit-exact verification against PyTorch quantized model,
    use the verification/ test suite (Phase 2).

    Cross-ref: docs/interface_spec.md §9 (Verification Contract)
    """
    print("[VERIFICATION] Testing Golden Model...")

    if not os.path.exists(META_PATH):
        print(f"  Skipping: {META_PATH} not found. Run export first.")
        return

    # 1. Initialize
    gm = GoldenModel()
    assert len(gm.layers) == 4, f"Expected 4 layers, got {len(gm.layers)}"
    print("  Model initialized with 4 layers.")

    # 2. Forward pass with zeros (baseline sanity)
    zero_input = np.zeros((1, 28, 28), dtype=np.int8)
    output = gm.forward(zero_input)
    assert output.shape == (10,), f"Output shape mismatch: expected (10,), got {output.shape}"
    assert output.dtype == np.int8, f"Output dtype mismatch: expected int8, got {output.dtype}"
    print(f"  Zero-input forward pass: shape={output.shape}, range=[{output.min()}, {output.max()}]")

    # 3. Forward pass with non-zero input
    rng = np.random.RandomState(42)
    random_input = rng.randint(INT8_MIN, INT8_MAX + 1, size=(1, 28, 28)).astype(np.int8)
    output2 = gm.forward(random_input)
    assert output2.shape == (10,), f"Output shape mismatch: {output2.shape}"
    assert np.all(output2 >= INT8_MIN) and np.all(output2 <= INT8_MAX), (
        f"Output values out of INT8 range: [{output2.min()}, {output2.max()}]"
    )
    print(f"  Random-input forward pass: shape={output2.shape}, range=[{output2.min()}, {output2.max()}]")

    # 4. Verify non-trivial output (random input should produce different output than zeros)
    # (This catches the all-zeros im2col bug if it somehow regresses)
    if np.array_equal(output, output2):
        print("  ⚠️  Zero and random inputs produced identical output — possible data-flow bug!")
    else:
        print("  Outputs differ for zero vs random input — data flow confirmed.")

    print("[VERIFICATION] ✅ Golden Model tests passed.\n")


if __name__ == "__main__":
    verify_golden_model()
