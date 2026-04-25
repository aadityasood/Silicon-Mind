# 🧠 PROJECT CONTEXT — Silicon Mind
<!-- LAST UPDATED: 2026-04-25 -->
<!-- UPDATE THIS FILE after every significant coding session -->
<!-- ANY agent can read this file to understand the full project state -->

## Project Identity
- **Name:** Silicon Mind  
- **Goal:** Custom RISC-V SoC with Neural Network Accelerator on FPGA
- **Team:** Aaditya Sood (CS/AI/ML — software stack) + Shaurya Gupta (ECE — hardware/Verilog/FPGA)
- **Purpose:** Portfolio project for Master's applications (TUM, RWTH Aachen) — deadline Feb–May 2027

## Current Phase
- **Phase:** 1 — Foundation
- **Sprint:** CNN Pipeline (Adi) + Systolic Array (Shaurya)
- **Status:** 🟡 train.py written, awaiting quantization pipeline. Shaurya has placeholders.

## Architecture Summary
```
┌─────────────────────────────── FPGA ───────────────────────────────┐
│                                                                     │
│  RISC-V Core (RV32I)  ◄── Memory-Mapped I/O ──►  Systolic Array   │
│  (Shaurya, from scratch)     Control + Data      (4×4, scales to    │
│  Runs C driver (Adi)                               8×8)             │
│         │                                                           │
│    Instruction +                                                    │
│    Data Memory (BRAM)  ← .mem weight files loaded here             │
└─────────────────────────────────────────────────────────────────────┘

SW Pipeline (runs on Adi's laptop, x86):
CNN (PyTorch) → INT8 Quantization (QAT) → im2col → Weight Export (.mem)
                                                         ↓
                                            Loaded into FPGA memory
                                                         ↓
HW Pipeline (runs on FPGA):
RISC-V Core executes C driver → orchestrates systolic array → inference result
```

| Component | Spec |
|:---|:---|
| CNN Model | TinyCNN for MNIST (Phase 1), then CIFAR-10 (Phase 5) |
| Precision | INT8 weights & activations, INT32 accumulator |
| Array Size | Start 4×4, scale to 8×8 (parameterized) |
| Dataflow | Weight-Stationary |
| Processor | RISC-V RV32I (Shaurya, from scratch) |
| ISA Extensions | Custom accelerator instructions (Phase 5) |
| Target Board | TBD — any FPGA works (no longer need ARM/PYNQ) |
| Verification | cocotb + Verilator (simulation-first approach) |
| Driver | C compiled with riscv32-unknown-elf-gcc |

## Key Decisions Made
| # | Decision | Rationale | Date |
|:--|:---------|:----------|:-----|
| D1 | Start with MNIST, then CIFAR-10 | Establish flow first, then increase complexity | 2026-04-18 |
| D2 | Start with 4×4 array, scale to 8×8 | Easier debugging, parameterized code makes scaling trivial | 2026-04-18 |
| D3 | Simulation-first, deploy later | Remove all bugs before touching hardware | 2026-04-18 |
| D4 | Hand-crafted Verilog (no HLS/FINN/Vitis AI) | Shows deeper understanding for portfolio | 2026-04-18 |
| D5 | Board TBD | All simulation work is board-independent | 2026-04-18 |
| D6 | Multi-agent coding workflow | Flash writes → Pro audits → Opus final review | 2026-04-18 |
| D7 | Accept Shaurya's folder structure (sim/, synth/) | Standard FPGA/ASIC conventions, no conflict with verification/ | 2026-04-25 |
| D8 | 3-branch git workflow (main, dev, hw-dev) | Clear ownership prevents merge conflicts | 2026-04-25 |
| D9 | RISC-V SoC architecture (RV32I core from scratch) | More impressive for portfolio. Shaurya has MIPS32 experience. Custom ISA extensions later. | 2026-04-25 |

## Completed Work
- [x] Project plan created (implementation_plan.md)
- [x] Directory structure set up
- [x] Context management system built
- [x] Agent prompt system designed
- [x] model/train.py written (MNIST CNN)
- [x] Dual-branch weekly workflow (v2) — reviews dev + hw-dev
- [x] Git strategy updated for 3-branch model
- [x] Architecture upgraded to RISC-V SoC
- [ ] Train the model (run train.py)
- [ ] model/quantize.py — QAT pipeline
- [ ] model/im2col.py — im2col transform
- [ ] model/export.py — weight export to .mem
- [ ] model/golden_model.py — bit-exact integer reference
- [ ] driver/main.c — C driver for RISC-V (Phase 3)
- [ ] ... (update as work progresses)

## Active Files
<!-- List files currently being worked on -->
| File | Status | Owner | Branch | Description |
|:-----|:-------|:------|:-------|:------------|
| `model/train.py` | 🟢 Written | Aaditya | `dev` | CNN training script |
| `model/quantize.py` | 🔴 Not started | Aaditya | `dev` | QAT quantization pipeline |
| `rtl/mac.v` | 🟡 Placeholder | Shaurya | `hw-dev` | MAC unit |
| `rtl/pe.v` | 🔴 Not started | Shaurya | `hw-dev` | Processing Element |
| `rtl/systolic_array.v` | 🔴 Not started | Shaurya | `hw-dev` | NxN systolic array |
| `rtl/riscv/` | 🔴 Not started | Shaurya | `hw-dev` | RISC-V core (Phase 2) |
| `sim/testbench.v` | 🟡 Placeholder | Shaurya | `hw-dev` | Verilog testbenches |
| `synth/synth.ys` | 🟢 Written | Shaurya | `hw-dev` | Yosys synthesis script |
| `driver/main.c` | 🔴 Not started | Aaditya | `dev` | C driver for RISC-V (Phase 3) |

## Folder Ownership
```
model/            ← Adi (PyTorch training, quantization, export)
verification/     ← Adi (cocotb Python tests)
driver/           ← Adi (C driver for RISC-V, compiled with riscv32-gcc)
scripts/          ← Adi (utility scripts)
rtl/              ← Shaurya (Verilog: PE, array, controller, RISC-V core)
sim/              ← Shaurya (Verilog testbenches)
synth/            ← Shaurya (Yosys/Vivado synthesis scripts)
data/             ← Shared (.mem weight files, test vectors)
docs/             ← Shared (interface_spec.md, etc.)
results/          ← Shared (simulation results, benchmarks)
```

## Interface Specification (HW-SW Contract)
<!-- This is the CRITICAL bridge between Aaditya and Shaurya's work -->
- **Data Width:** 8 bits (signed INT8: -128 to +127)
- **Accumulator Width:** 32 bits (signed INT32)
- **Weight Format:** Row-major, one weight per PE, loaded via control signal
- **Input Format:** Row-major INT8, skewed for systolic feeding
- **Output Format:** INT32 partial sums, requantized to INT8 by hardware
- **Memory Files:** `.mem` format (hex), loaded via `$readmemh` in Verilog
- **Tiling:** Matrices larger than NxN are tiled; controller handles sequencing

### RISC-V ↔ Accelerator Interface (Memory-Mapped I/O)
```
Base Address: 0x10000000
+0x00  ACCEL_CTRL        (W)  — bit0: start, bit1: reset
+0x04  ACCEL_STATUS      (R)  — bit0: done, bit1: busy
+0x08  ACCEL_WEIGHT_ADDR (W)  — base address for weight data
+0x0C  ACCEL_INPUT_ADDR  (W)  — base address for input data
+0x10  ACCEL_OUTPUT_ADDR (R)  — base address for output data
+0x14  ACCEL_ARRAY_SIZE  (W)  — array dimension (4 or 8)
+0x18  ACCEL_TILE_COUNT  (W)  — number of tiles to process
```
> NOTE: This is a proposed interface. Must be agreed between Adi and Shaurya.

## Blockers & Open Items
- [ ] FPGA board selection (any FPGA works now — no ARM needed)
- [ ] Shaurya: RISC-V core scope agreement (RV32I minimum ISA)
- [ ] Agree on memory-mapped accelerator interface addresses

## Performance Targets
| Metric | Target | Notes |
|:-------|:-------|:------|
| MNIST Accuracy (FP32) | >98% | Baseline |
| MNIST Accuracy (INT8) | >96% | <2% drop from quantization |
| Clock Frequency | ≥100 MHz | On target FPGA |
| Array Utilization | >80% DSP usage | For chosen array size |
| Inference Latency | TBD | Benchmark on FPGA |
| Custom ISA Speedup | TBD | vs base RV32I driver (Phase 5) |

## Agent Workflow
- **Primary Agent (Opus):** Architecture decisions, complex logic, final audit
- **Code Writer (Flash):** Boilerplate, initial implementations
- **Code Auditor (Pro):** Review, improve, catch bugs
- **Context File:** THIS FILE — always read first, always update last
