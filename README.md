# Silicon Mind — Custom FPGA Systolic Array Accelerator

> A custom-designed FPGA-based systolic array accelerator for CNN inference, built from scratch in PyTorch (software) and Verilog (hardware).

## 🎯 Project Overview

**Silicon Mind** is a hardware-software co-design project that takes a Convolutional Neural Network, quantizes it to INT8 precision, and runs inference on a custom systolic array implemented on an FPGA.

| | |
|:---|:---|
| **Software** | PyTorch CNN → INT8 Quantization → Weight Export |
| **Hardware** | Verilog Systolic Array → FPGA Bitstream |
| **Bridge** | im2col transform + cocotb verification |
| **Target** | PYNQ-Z2 / Zynq SoC (TBD) |

## 📁 Project Structure

```
silicon-mind/
├── model/              # CNN training, quantization, weight export
├── rtl/                # Verilog RTL source code
├── verification/       # cocotb testbenches and test vectors
├── driver/             # PYNQ deployment driver
├── docs/               # Architecture docs, interface spec
├── scripts/            # Build and automation scripts
└── results/            # Performance benchmarks
```

## 🚀 Quick Start

```bash
# Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib cocotb

# Train the CNN
python model/train.py

# Run hardware simulation tests
cd verification && make
```

## 🏗️ Architecture

```
Input Image → im2col → Tiled MatMul → Systolic Array (FPGA) → Requantize → Output
                                            ↑
                                    INT8 weights loaded
                                    from .mem files
```

## 👥 Team

- **Adi** — AI/ML, software stack, verification
- **Shaurya** — Hardware design, Verilog RTL, FPGA synthesis

## 📄 License

MIT License
