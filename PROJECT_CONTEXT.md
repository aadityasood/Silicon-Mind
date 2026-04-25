# 🧠 PROJECT CONTEXT — Silicon Mind
<!-- LAST UPDATED: 2026-04-25 -->
<!-- UPDATE THIS FILE after every significant coding session -->
<!-- ANY agent can read this file to understand the full project state -->

## Project Identity
- **Name:** Silicon Mind  
- **Goal:** Custom FPGA-Based Systolic Array Accelerator for CNN Inference
- **Team:** Aaditya Sood (CS/AI/ML — software stack) + Shaurya Gupta (ECE — hardware/Verilog/FPGA)
- **Purpose:** Portfolio project for Master's applications (TUM, RWTH Aachen)

## Current Phase
- **Phase:** 1 — Foundation
- **Sprint:** Pre-coding setup
- **Status:** 🟡 Setting up project infrastructure and context management

## Architecture Summary
```
CNN (PyTorch) → INT8 Quantization (QAT) → im2col → Weight Export (.mem)
                                                         ↓
PYNQ Driver (Python) ← AXI Bus ← Systolic Array (Verilog RTL) ← Weights
```

| Component | Spec |
|:---|:---|
| CNN Model | TinyCNN for MNIST (Phase 1), then CIFAR-10 (Phase 2) |
| Precision | INT8 weights & activations, INT32 accumulator |
| Array Size | Start 4×4, scale to 8×8 (parameterized) |
| Dataflow | Weight-Stationary |
| Target Board | TBD (likely PYNQ-Z2 or Zybo Z7) |
| Verification | cocotb + Verilator (simulation-first approach) |

## Key Decisions Made
| # | Decision | Rationale | Date |
|:--|:---------|:----------|:-----|
| D1 | Start with MNIST, then CIFAR-10 | Establish flow first, then increase complexity | 2026-04-18 |
| D2 | Start with 4×4 array, scale to 8×8 | Easier debugging, parameterized code makes scaling trivial | 2026-04-18 |
| D3 | Simulation-first, deploy later | Remove all bugs before touching hardware | 2026-04-18 |
| D4 | Hand-crafted Verilog (no HLS/FINN/Vitis AI) | Shows deeper understanding for portfolio | 2026-04-18 |
| D5 | Board TBD (2-3 days) | All simulation work is board-independent | 2026-04-18 |
| D6 | Multi-agent coding workflow | Flash writes → Pro audits → Opus final review | 2026-04-18 |
| D7 | Accept Shaurya's folder structure (sim/, synth/) | Standard FPGA/ASIC conventions, no conflict with verification/ | 2026-04-25 |
| D8 | 3-branch git workflow (main, dev, hw-dev) | Clear ownership prevents merge conflicts | 2026-04-25 |

## Completed Work
- [x] Project plan created (implementation_plan.md)
- [x] Directory structure set up
- [x] Context management system built
- [x] Agent prompt system designed
- [x] model/train.py written (MNIST CNN)
- [x] Dual-branch weekly workflow (v2) — reviews dev + hw-dev
- [x] Git strategy updated for 3-branch model
- [ ] Train the model (run train.py)
- [ ] model/quantize.py — QAT pipeline
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
| `sim/testbench.v` | 🟡 Placeholder | Shaurya | `hw-dev` | Verilog testbenches |
| `synth/synth.ys` | 🟢 Written | Shaurya | `hw-dev` | Yosys synthesis script (SKY130) |

## Interface Specification (HW-SW Contract)
<!-- This is the CRITICAL bridge between Aaditya and Shaurya's work -->
- **Data Width:** 8 bits (signed INT8: -128 to +127)
- **Accumulator Width:** 32 bits (signed INT32)
- **Weight Format:** Row-major, one weight per PE, loaded via control signal
- **Input Format:** Row-major INT8, skewed for systolic feeding
- **Output Format:** INT32 partial sums, requantized to INT8 by hardware
- **Memory Files:** `.mem` format (hex), loaded via `$readmemh` in Verilog
- **Tiling:** Matrices larger than NxN are tiled; controller handles sequencing

## Blockers & Open Items
- [ ] FPGA board selection (waiting on Shaurya, ETA: 2-3 days)
- [ ] Vivado installation on Shaurya's machine

## Performance Targets
| Metric | Target | Notes |
|:-------|:-------|:------|
| MNIST Accuracy (FP32) | >98% | Baseline |
| MNIST Accuracy (INT8) | >96% | <2% drop from quantization |
| Clock Frequency | ≥100 MHz | On target FPGA |
| Array Utilization | >80% DSP usage | For chosen array size |

## Agent Workflow
- **Primary Agent (Opus):** Architecture decisions, complex logic, final audit
- **Code Writer (Flash):** Boilerplate, initial implementations
- **Code Auditor (Pro):** Review, improve, catch bugs
- **Context File:** THIS FILE — always read first, always update last
