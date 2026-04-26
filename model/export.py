"""
model/export.py — Weight, Bias & Metadata Export for Project Silicon Mind

Exports quantized model weights and biases to .mem files for Verilog
$readmemh, and generates a JSON metadata file with per-layer quantization
parameters needed for hardware requantization.

Formats (per docs/interface_spec.md):
    - Weights: INT8 hex (2 chars), one value per line, row-major (§3)
    - Bias: INT32 hex (8 chars), one value per line (§1)
    - Metadata: JSON with scales, zero-points, multiplier, shift (§1)

File naming convention (per interface_spec.md §3):
    - layer0_weights.mem, layer0_bias.mem  (conv1)
    - layer1_weights.mem, layer1_bias.mem  (conv2)
    - layer2_weights.mem, layer2_bias.mem  (fc1)
    - layer3_weights.mem, layer3_bias.mem  (fc2)

Cross-reference:
    - docs/interface_spec.md §1 (Data Types: INT8 weight, INT32 bias/accumulator)
    - docs/interface_spec.md §3 (Weight File Format: hex, two's complement)
    - docs/interface_spec.md §5 (Output Format: requantization parameters)

Author: Aaditya Sood
Date: 2026-04-26
"""

import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.ao.quantization as quant

# ─── Constants ───────────────────────────────────────────────────────────────
ARRAY_SIZE = 4
DATA_DIR = "data"
QUANT_MODEL_PATH = os.path.join(DATA_DIR, "mnist_cnn_quant.pth")
EXPORT_DIR = os.path.join(DATA_DIR, "mem_files")

# Layer name → interface_spec.md layer index mapping
_LAYER_INDEX = {
    "conv1": 0,
    "conv2": 1,
    "fc1": 2,
    "fc2": 3,
}


def _ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def _val_to_hex8(val: int) -> str:
    """Convert a signed integer to 2-character hex (two's complement).

    Cross-ref: docs/interface_spec.md §3 (Weight File Format)
    """
    return f"{val & 0xFF:02X}"


def _val_to_hex32(val: int) -> str:
    """Convert a signed integer to 8-character hex (two's complement).

    Cross-ref: docs/interface_spec.md §1 (INT32 bias)
    """
    return f"{val & 0xFFFFFFFF:08X}"


def export_tensor_to_mem(
    tensor: np.ndarray, filepath: str, bits: int = 8
) -> int:
    """Export a flattened numpy array to a .mem file for Verilog $readmemh.

    Args:
        tensor: Integer numpy array to export.
        filepath: Output .mem file path.
        bits: 8 for INT8 weights/activations, 32 for INT32 biases.

    Returns:
        Number of values written.

    Raises:
        ValueError: If bits is not 8 or 32.
    """
    if bits not in (8, 32):
        raise ValueError(f"bits must be 8 or 32, got {bits}")

    hex_fn = _val_to_hex8 if bits == 8 else _val_to_hex32
    flat = tensor.flatten()

    with open(filepath, "w") as f:
        for val in flat:
            f.write(hex_fn(int(val)) + "\n")

    print(f"  Exported {len(flat)} values ({bits}-bit) → {filepath}")
    return len(flat)


def _compute_requant_params(
    input_scale: float,
    weight_scale: float,
    output_scale: float,
) -> Tuple[int, int]:
    """Compute fixed-point requantization multiplier and shift.

    The real-valued scale factor M = (S_in * S_w) / S_out is decomposed
    into an integer multiplier M0 and a right-shift amount:
        M ≈ M0 * 2^(-shift)

    The hardware computes: output = ((acc * M0) >> shift) + zero_point

    Args:
        input_scale: Activation quantization scale (S_in).
        weight_scale: Weight quantization scale (S_w).
        output_scale: Output activation quantization scale (S_out).

    Returns:
        Tuple of (multiplier: int, shift: int) for hardware requantization.

    Cross-ref: docs/interface_spec.md §1 (scale_factor, shift_amount)
    """
    if output_scale == 0.0:
        raise ValueError("output_scale cannot be zero for requantization.")

    M = (input_scale * weight_scale) / output_scale

    # Find shift such that M0 = round(M * 2^shift) fits in 31 bits
    # Standard approach: M0 should be in [2^30, 2^31) for maximum precision
    if M == 0.0:
        return 0, 0

    shift = 0
    while M * (1 << shift) < (1 << 30) and shift < 31:
        shift += 1

    multiplier = int(round(M * (1 << shift)))

    # Clamp to 32-bit unsigned range (interface_spec.md §1: scale_factor is 32-bit)
    multiplier = min(multiplier, 0xFFFFFFFF)

    return multiplier, shift


def _scale_bias_to_accumulator(
    fp_bias: np.ndarray,
    input_scale: float,
    weight_scale: float,
) -> np.ndarray:
    """Scale FP32 bias to INT32 accumulator domain.

    In the hardware accumulator, partial sums have an implicit scale of
    S_in * S_w. The bias must be in the same domain:
        int_bias = round(fp_bias / (S_in * S_w))

    CRITICAL: Simply rounding FP32 bias to INT32 (the old code's approach)
    produces wrong results because it ignores the quantization scale.
   

    Args:
        fp_bias: FP32 bias tensor from the quantized model.
        input_scale: Input activation scale (S_in).
        weight_scale: Weight scale (S_w) — scalar for per-tensor,
                       array for per-channel.

    Returns:
        INT32 bias array scaled to accumulator domain.

    Cross-ref: docs/interface_spec.md §1 (bias: "Pre-scaled bias added
               after accumulation")
    """
    accumulator_scale = input_scale * weight_scale
    if np.any(accumulator_scale == 0.0):
        raise ValueError(
            "Accumulator scale (S_in * S_w) is zero. "
            "Check that quantization was performed correctly."
        )

    int_bias = np.round(fp_bias / accumulator_scale).astype(np.int32)

    # Overflow check
    overflow_mask = (int_bias < np.iinfo(np.int32).min) | (int_bias > np.iinfo(np.int32).max)
    if np.any(overflow_mask):
        n_overflow = int(np.sum(overflow_mask))
        print(f"  ⚠️  {n_overflow} bias values overflow INT32 range!")

    return int_bias


def _load_quantized_model() -> torch.nn.Module:
    """Reconstruct and load the quantized model.

    Recreates the model structure, applies the same fusion/conversion
    pipeline as quantize.py, then loads the saved state dict.

    Raises:
        FileNotFoundError: If quantized model weights don't exist.
        RuntimeError: If state dict keys don't match the reconstructed model.
    """
    if not os.path.exists(QUANT_MODEL_PATH):
        raise FileNotFoundError(
            f"Quantized model not found at {QUANT_MODEL_PATH}. "
            "Run 'python -m model.quantize' first."
        )

    from model.train import SiliconMindCNN

    torch.backends.quantized.engine = "fbgemm"
    model = SiliconMindCNN()
    model.eval()

    model.qconfig = quant.get_default_qat_qconfig("fbgemm")
    model = quant.fuse_modules(
        model, [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]]
    )
    model.train()
    model = quant.prepare_qat(model)
    model = quant.convert(model)

    # Load weights — verify keys match
    state_dict = torch.load(QUANT_MODEL_PATH, map_location="cpu", weights_only=True)
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())

    if model_keys != saved_keys:
        missing = model_keys - saved_keys
        unexpected = saved_keys - model_keys
        msg = "State dict key mismatch between reconstructed model and saved weights."
        if missing:
            msg += f"\n  Missing keys: {missing}"
        if unexpected:
            msg += f"\n  Unexpected keys: {unexpected}"
        raise RuntimeError(msg)

    model.load_state_dict(state_dict)
    print(f"Loaded quantized model from {QUANT_MODEL_PATH}")

    return model


def _get_input_scale(
    model: torch.nn.Module, layer_name: str, metadata_so_far: List[Dict],
) -> float:
    """Get the input activation scale for a given layer.

    For the first layer (conv1), the input scale comes from the QuantStub.
    For subsequent layers, it's the output scale of the previous layer.

    Args:
        model: The quantized model.
        layer_name: Name of the current layer.
        metadata_so_far: Metadata for previously-processed layers.

    Returns:
        Input activation scale (float).
    """
    if layer_name == "conv1":
        # Input scale comes from QuantStub
        if hasattr(model, "quant") and hasattr(model.quant, "scale"):
            return float(model.quant.scale)
        return 1.0
    else:
        # Input scale = previous layer's output scale
        if metadata_so_far:
            return metadata_so_far[-1].get("output_scale", 1.0)
        return 1.0


def export_quantized_model() -> None:
    """Load quantized model and export weights, biases, and metadata.

    For each layer:
        1. Extract INT8 weights → layer{N}_weights.mem
        2. Scale FP32 bias to INT32 accumulator domain → layer{N}_bias.mem
        3. Compute requantization multiplier+shift → metadata JSON
        4. Record input_scale for downstream layers

    Cross-ref:
        - docs/interface_spec.md §1 (bias: pre-scaled INT32)
        - docs/interface_spec.md §3 (weight .mem format)
        - docs/interface_spec.md §5 (requantization)
    """
    print("=== Silicon Mind: Exporting Quantized Model ===")
    _ensure_dir(EXPORT_DIR)

    model = _load_quantized_model()

    metadata: List[Dict[str, Any]] = []

    # Layer iteration order matches network topology
    layer_names = ["conv1", "conv2", "fc1", "fc2"]

    for name in layer_names:
        layer = getattr(model, name)
        layer_idx = _LAYER_INDEX[name]
        print(f"\nProcessing layer: {name} (layer{layer_idx})")

        # ── 1. Extract INT8 weights ──
        q_weight = layer.weight()
        int_weight = q_weight.int_repr().detach().cpu().numpy()

        weight_path = os.path.join(EXPORT_DIR, f"layer{layer_idx}_weights.mem")
        export_tensor_to_mem(int_weight, weight_path, bits=8)

        # ── 2. Extract weight quantization parameters ──
        qscheme = q_weight.qscheme()
        if qscheme == torch.per_tensor_affine:
            weight_scale = float(q_weight.q_scale())
            weight_zero_point = int(q_weight.q_zero_point())
        elif qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            weight_scale = q_weight.q_per_channel_scales().numpy().tolist()
            weight_zero_point = q_weight.q_per_channel_zero_points().numpy().tolist()
        else:
            weight_scale = 1.0
            weight_zero_point = 0

        # ── 3. Get output scale and zero point ──
        output_scale = float(getattr(layer, "scale", 1.0))
        output_zero_point = int(getattr(layer, "zero_point", 0))

        # ── 4. Get input scale ──
        input_scale = _get_input_scale(model, name, metadata)

        # ── 5. Scale bias to accumulator domain ──
        fp_bias = layer.bias().detach().cpu().numpy()

        # For per-tensor quantization, weight_scale is a scalar
        # For per-channel, use per-channel scales
        if isinstance(weight_scale, list):
            ws_array = np.array(weight_scale, dtype=np.float64)
        else:
            ws_array = np.float64(weight_scale)

        int_bias = _scale_bias_to_accumulator(fp_bias, input_scale, ws_array)
        bias_path = os.path.join(EXPORT_DIR, f"layer{layer_idx}_bias.mem")
        export_tensor_to_mem(int_bias, bias_path, bits=32)

        # ── 6. Compute requantization parameters ──
        if isinstance(weight_scale, list):
            # Per-channel: compute per-channel multiplier/shift
            multipliers = []
            shifts = []
            for ws in weight_scale:
                m, s = _compute_requant_params(input_scale, ws, output_scale)
                multipliers.append(m)
                shifts.append(s)
        else:
            m, s = _compute_requant_params(input_scale, weight_scale, output_scale)
            multipliers = m
            shifts = s

        # ── 7. Build layer metadata ──
        layer_info: Dict[str, Any] = {
            "name": name,
            "layer_index": layer_idx,
            "weight_shape": list(int_weight.shape),
            "qscheme": str(qscheme),
            "weight_scale": weight_scale,
            "weight_zero_point": weight_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "output_zero_point": output_zero_point,
            "requant_multiplier": multipliers,
            "requant_shift": shifts,
        }
        metadata.append(layer_info)

    # ── Save metadata ──
    meta_path = os.path.join(EXPORT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\nExported metadata to {meta_path}")


def verify_export() -> None:
    """Verify exported .mem files for correctness.

    Checks:
        1. Metadata JSON exists and has 4 layers
        2. Every referenced .mem file exists
        3. .mem files are non-empty
        4. Every line is valid hex of correct width
        5. Weight .mem line count matches flattened weight shape
        6. INT8 hex values are in valid range (00-FF)

    Cross-ref: docs/interface_spec.md §3 (hex format, one value per line)
    """
    print("\n[VERIFICATION] Checking exported files...")

    meta_path = os.path.join(EXPORT_DIR, "model_meta.json")
    if not os.path.exists(meta_path):
        print("❌ Metadata file missing. Run export first.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert len(meta) == 4, f"Expected 4 layers in metadata, got {len(meta)}"

    for layer in meta:
        idx = layer["layer_index"]
        name = layer["name"]

        # Check weight .mem
        weight_path = os.path.join(EXPORT_DIR, f"layer{idx}_weights.mem")
        assert os.path.exists(weight_path), f"Missing: {weight_path}"

        with open(weight_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        expected_count = 1
        for dim in layer["weight_shape"]:
            expected_count *= dim

        assert len(lines) == expected_count, (
            f"layer{idx} weights: expected {expected_count} lines, got {len(lines)}"
        )

        # Validate hex format (2 chars for INT8)
        for i, line in enumerate(lines):
            assert len(line) == 2 and all(c in "0123456789ABCDEF" for c in line), (
                f"layer{idx} weights line {i}: invalid hex '{line}'"
            )

        # Check bias .mem
        bias_path = os.path.join(EXPORT_DIR, f"layer{idx}_bias.mem")
        assert os.path.exists(bias_path), f"Missing: {bias_path}"

        with open(bias_path, "r") as f:
            bias_lines = [l.strip() for l in f if l.strip()]

        expected_bias_count = layer["weight_shape"][0]  # C_out
        assert len(bias_lines) == expected_bias_count, (
            f"layer{idx} bias: expected {expected_bias_count} lines, got {len(bias_lines)}"
        )

        # Validate hex format (8 chars for INT32)
        for i, line in enumerate(bias_lines):
            assert len(line) == 8 and all(c in "0123456789ABCDEF" for c in line), (
                f"layer{idx} bias line {i}: invalid hex '{line}'"
            )

        # Check requantization parameters exist
        assert "requant_multiplier" in layer, f"layer{idx}: missing requant_multiplier"
        assert "requant_shift" in layer, f"layer{idx}: missing requant_shift"
        assert "input_scale" in layer, f"layer{idx}: missing input_scale"

        print(f"  ✅ layer{idx} ({name}): weights={len(lines)}, bias={len(bias_lines)}, meta OK")

    print("[VERIFICATION] ✅ All export checks passed.\n")


if __name__ == "__main__":
    if not os.path.exists(QUANT_MODEL_PATH):
        print(f"Error: Quantized model not found at {QUANT_MODEL_PATH}")
        print("Run 'python -m model.quantize' first.")
    else:
        export_quantized_model()
        verify_export()
