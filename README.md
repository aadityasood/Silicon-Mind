# Silicon Mind - Custom RISC-V Neural Accelerator

Silicon Mind is a hardware-software co-design project that builds a custom
RISC-V SoC with an INT8 neural-network accelerator. The software stack trains
and exports quantized neural-network weights; the hardware stack implements the
accelerator and processor-side integration for FPGA deployment.

## Project Overview

| Area | Scope |
|:--|:--|
| Software | PyTorch CNN, INT8 quantization, im2col, `.mem` export, golden model |
| Hardware | Verilog systolic array, RV32I core, testbenches, synthesis flow |
| Interface | Memory-mapped accelerator registers and `.mem` test vectors |
| Target | Simulation first, then FPGA board selection and deployment |

## Architecture

```text
PyTorch CNN
  -> INT8 quantization
  -> im2col / tiled matrix form
  -> .mem weight and test-vector export
  -> Verilog simulation
  -> RISC-V C driver
  -> FPGA accelerator execution
```

The hardware path uses signed INT8 activations and weights, signed INT32
accumulators, and fixed-point requantization back to INT8.

## Repository Structure

```text
silicon-mind/
├── model/          Python model, quantization, export, golden reference
├── data/           Exported weights, metadata, and test data
├── docs/           Architecture and hardware-software interface docs
├── rtl/            Verilog RTL modules
├── sim/            Verilog simulation testbenches
├── synth/          Synthesis scripts and constraints
├── verification/   cocotb tests and generated test vectors
├── driver/         RISC-V C driver
└── results/        Simulation, synthesis, and benchmark outputs
```

Some directories are introduced as the project reaches later phases.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib cocotb
```

Verify exported model artifacts:

```bash
.venv/bin/python -c 'from model.export import verify_export; verify_export()'
.venv/bin/python -m model.golden_model
```

Train or regenerate model artifacts when needed:

```bash
.venv/bin/python model/train.py
.venv/bin/python model/quantize.py
.venv/bin/python model/export.py
```

## Team

- Aaditya Sood - AI/ML, software stack, verification, driver integration
- Shaurya Gupta - Verilog RTL, RISC-V hardware, FPGA synthesis

## Development Policy

- `main` contains reviewed, working project milestones.
- `dev` contains Aaditya's software work.
- `hw-dev` contains Shaurya's hardware work.
- Draft scaffolding is acceptable on development branches, but merge to `main`
  requires reviewed implementation and passing verification.

## License

MIT License
