# 🔌 Hardware-Software Interface Specification
<!-- VERSION: 1.0 | LAST UPDATED: 2026-04-18 -->
<!-- This document is the CONTRACT between Adi (software) and Shaurya (hardware) -->
<!-- BOTH sides must agree on changes. Update version number on every change. -->

## 1. Data Types & Widths

| Signal | Width | Type | Range | Notes |
|:-------|:------|:-----|:------|:------|
| `weight` | 8 bits | Signed INT8 | -128 to +127 | Quantized CNN weights |
| `activation` | 8 bits | Signed INT8 | -128 to +127 | Quantized activations (post-ReLU: 0 to +127) |
| `accumulator` | 32 bits | Signed INT32 | -2³¹ to 2³¹-1 | Internal PE accumulator (prevents overflow) |
| `bias` | 32 bits | Signed INT32 | — | Pre-scaled bias added after accumulation |
| `scale_factor` | 32 bits | Unsigned | — | Fixed-point multiplier for requantization (M = S_act × S_weight / S_output) |
| `shift_amount` | 8 bits | Unsigned | 0 to 31 | Right-shift for requantization |

## 2. Array Dimensions

```
Parameter ARRAY_SIZE = 4;  // Start with 4, scale to 8
// Total PEs = ARRAY_SIZE × ARRAY_SIZE
// All Verilog modules MUST use this parameter, not hardcoded values
```

## 3. Weight File Format (.mem)

### Convention
- **File naming:** `layer{L}_weights.mem` (e.g., `layer0_weights.mem`)
- **Format:** Hexadecimal, one value per line, 2 hex digits per INT8 value
- **Encoding:** Two's complement for signed INT8
- **Layout:** Row-major order

### Example (4×4 weight matrix)
```
# layer0_weights.mem
# Row 0
03    # +3
FE    # -2 (two's complement)
01    # +1
00    # +0
# Row 1
05    # +5
FC    # -4
02    # +2
FF    # -1
# ... (rows 2-3)
```

### Python Export Code (Adi writes this)
```python
def export_weights_to_mem(weight_tensor, filepath):
    """Export INT8 weight tensor to .mem file for Verilog $readmemh"""
    flat = weight_tensor.flatten().numpy().astype(np.int8)
    with open(filepath, 'w') as f:
        for val in flat:
            # Convert signed int8 to unsigned hex (two's complement)
            f.write(f"{val & 0xFF:02X}\n")
```

### Verilog Load Code (Shaurya writes this)
```verilog
reg [7:0] weights [0:ARRAY_SIZE*ARRAY_SIZE-1];
initial $readmemh("layer0_weights.mem", weights);
```

## 4. Input Activation Format

- **Same format as weights:** `.mem` files, hex, INT8, row-major
- **File naming:** `layer{L}_input.mem`
- **Skewing:** The SOFTWARE exports flat matrices. The HARDWARE controller handles input skewing for systolic feeding. Software does NOT pre-skew.

## 5. Output Format

- **From hardware:** INT32 accumulated results, row-major
- **Requantization:** Done IN HARDWARE (requantize.v) — multiply by scale, right-shift, clamp to INT8
- **To software:** INT8 results for next layer

## 6. Control Signals (High-Level)

| Signal | Direction | Width | Description |
|:-------|:----------|:------|:------------|
| `clk` | → HW | 1 | System clock |
| `rst_n` | → HW | 1 | Active-low reset |
| `load_weights` | → HW | 1 | Asserted when loading weights into PEs |
| `start_compute` | → HW | 1 | Begin matrix multiplication |
| `done` | ← HW | 1 | Asserted when computation is complete |
| `input_valid` | → HW | 1 | Input data is valid on input bus |
| `output_valid` | ← HW | 1 | Output data is valid on output bus |

## 7. Tiling Strategy

When matrix dimensions exceed ARRAY_SIZE:
1. Software tiles the matrices into ARRAY_SIZE × ARRAY_SIZE blocks
2. Software sends tiles in order: column of A × row of B
3. Hardware accumulates partial sums across tiles
4. Software manages the tile sequencing via the driver

```
Example: 8×8 matmul on 4×4 array
Matrix A (8×8) → 4 tiles of 4×4
Matrix B (8×8) → 4 tiles of 4×4
Result C = sum of partial products from 2 tile passes per output block
Total output blocks: 4 (each 4×4)
```

## 8. Layer Execution Order

For CNN inference, layers execute sequentially:
```
Input Image → [Conv1 → ReLU → Pool] → [Conv2 → ReLU → Pool] → [FC1 → ReLU] → [FC2] → Output
                  ↑                        ↑                       ↑              ↑
            Systolic Array           Systolic Array          Systolic Array   Systolic Array
            (im2col'd)               (im2col'd)             (direct matmul)  (direct matmul)
```

Each layer's output becomes the next layer's input. The ARM CPU (or software in simulation) orchestrates this sequence.

## 9. Verification Contract

**Adi provides:** Test vectors in `.mem` format + expected output in `.mem` format
**Shaurya verifies:** Verilog simulation output matches expected output exactly (bit-for-bit)

Test vector naming:
```
verification/test_vectors/
├── test_matmul_4x4_input_a.mem
├── test_matmul_4x4_input_b.mem
├── test_matmul_4x4_expected_output.mem
├── test_layer0_input.mem
├── test_layer0_weights.mem
└── test_layer0_expected_output.mem
```

## 10. Changelog

| Version | Date | Change | Author |
|:--------|:-----|:-------|:-------|
| 1.0 | 2026-04-18 | Initial specification | Adi + Antigravity |
