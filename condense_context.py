#!/usr/bin/env python3
"""
🧊 Context Condenser — Silicon Mind Project
============================================
Reads all project state files and generates a compact, token-efficient
context blob that can be pasted into ANY new AI chat to bring the agent
up to speed instantly.

Usage:
    python condense_context.py              # Print condensed context to stdout
    python condense_context.py --clipboard  # Copy to clipboard (requires xclip)
    python condense_context.py --output FILE # Write to file

The output is designed to be pasted as the FIRST message in a new chat
with any AI model (Flash, Pro, Opus, etc.)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent
CONTEXT_FILE = PROJECT_ROOT / "PROJECT_CONTEXT.md"
INTERFACE_SPEC = PROJECT_ROOT / "docs" / "interface_spec.md"
DECISIONS_LOG = PROJECT_ROOT / "DECISIONS_LOG.jsonl"

def read_file_safe(path):
    """Read a file, return empty string if not found."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return ""

def get_file_tree(root, max_depth=3, prefix=""):
    """Generate a compact file tree string."""
    lines = []
    root = Path(root)
    
    # Skip hidden dirs, __pycache__, .venv, etc.
    skip = {'.git', '__pycache__', '.venv', 'node_modules', '.vivado', 'checkpoints'}
    
    items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    for i, item in enumerate(items):
        if item.name in skip or item.name.startswith('.'):
            continue
        
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        if item.is_dir():
            lines.append(f"{prefix}{connector}{item.name}/")
            if max_depth > 1:
                extension = "    " if is_last else "│   "
                lines.extend(get_file_tree(item, max_depth - 1, prefix + extension))
        else:
            size = item.stat().st_size
            size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
            lines.append(f"{prefix}{connector}{item.name} ({size_str})")
    
    return lines

def get_recent_decisions():
    """Load recent decisions from the JSONL log."""
    decisions = []
    if DECISIONS_LOG.exists():
        for line in DECISIONS_LOG.read_text().strip().split('\n'):
            if line.strip():
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return decisions[-10:]  # Last 10 decisions

def get_code_status():
    """Check which key files exist and their status."""
    key_files = {
        "model/train.py": "CNN training script",
        "model/quantize.py": "Quantization pipeline",
        "model/golden_model.py": "Bit-exact reference model",
        "model/im2col.py": "Conv→MatMul transform",
        "model/export_weights.py": "Weight serializer",
        "rtl/pe.v": "Processing Element",
        "rtl/systolic_array.v": "Systolic Array",
        "rtl/controller.v": "FSM Controller",
        "rtl/axi_wrapper.v": "AXI Interface",
        "rtl/top.v": "Top-level module",
        "verification/cocotb_tests/test_pe.py": "PE testbench",
        "verification/cocotb_tests/test_systolic_array.py": "Array testbench",
        "driver/overlay_driver.py": "PYNQ driver",
    }
    
    status = []
    for path, desc in key_files.items():
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            size = full_path.stat().st_size
            lines = len(full_path.read_text().split('\n'))
            status.append(f"  ✅ {path} ({lines} lines) — {desc}")
        else:
            status.append(f"  ⬜ {path} — {desc}")
    
    return "\n".join(status)

def condense():
    """Generate the condensed context blob."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Read project context
    context = read_file_safe(CONTEXT_FILE)
    
    # Extract key sections from context (parse markdown)
    current_phase = "Unknown"
    decisions = []
    for line in context.split('\n'):
        if line.startswith('- **Phase:**'):
            current_phase = line.split(':**')[1].strip()
        if line.startswith('| D'):
            decisions.append(line)
    
    # Build condensed output
    output = f"""=== SILICON MIND PROJECT CONTEXT (Auto-condensed: {timestamp}) ===

PROJECT: Custom FPGA-Based Systolic Array Accelerator for CNN Inference
TEAM: Adi (CS/AI/ML — software) + Shaurya (ECE — hardware/Verilog)
PURPOSE: Portfolio for Master's applications (TUM, RWTH Aachen)

CURRENT PHASE: {current_phase}

ARCHITECTURE:
  CNN (PyTorch) → INT8 Quantization → im2col → Weight Export (.mem files)
  → Systolic Array (Verilog RTL) → FPGA (board TBD)
  
  Array: {get_array_size()}×{get_array_size()} Weight-Stationary Systolic Array
  Precision: INT8 weights/activations, INT32 accumulator
  Dataset: MNIST (Phase 1) → CIFAR-10 (Phase 2)
  Verification: Simulation-first (cocotb + Verilator)

KEY DECISIONS:
{chr(10).join(f'  • {d}' for d in decisions) if decisions else '  (see PROJECT_CONTEXT.md)'}

FILE STATUS:
{get_code_status()}

FILE TREE:
{chr(10).join(get_file_tree(PROJECT_ROOT, max_depth=2))}

INTERFACE SPEC (HW-SW Contract):
  - Data: Signed INT8 (8-bit), Accumulator: INT32 (32-bit)
  - Weight files: .mem format (hex, two's complement, row-major)
  - Skewing: Done by HARDWARE, not software
  - Requantization: Done by HARDWARE (multiply by scale, shift, clamp)
  - Tiling: Software tiles matrices > ARRAY_SIZE, hardware processes tiles

CODING RULES:
  1. All Verilog must be PARAMETERIZED (use `parameter ARRAY_SIZE`)
  2. All Python must include type hints and docstrings
  3. Golden model must use INTEGER arithmetic only (no floats in HW path)
  4. Test vectors: .mem format, matching interface_spec.md conventions
  5. Every function needs a unit test

CONTEXT FILES (read these for full details):
  - PROJECT_CONTEXT.md — Full project state (this is the condensed version)
  - docs/interface_spec.md — HW-SW data format contract
  - DECISIONS_LOG.jsonl — All architectural decisions with rationale

=== END CONTEXT ==="""
    
    return output

def get_array_size():
    """Get current array size from context."""
    context = read_file_safe(CONTEXT_FILE)
    if "4×4" in context:
        return "4"
    elif "8×8" in context:
        return "8"
    return "N"

def main():
    output = condense()
    
    if "--clipboard" in sys.argv:
        try:
            import subprocess
            proc = subprocess.Popen(['xclip', '-selection', 'clipboard'], 
                                   stdin=subprocess.PIPE)
            proc.communicate(output.encode())
            print("✅ Condensed context copied to clipboard!")
            print(f"   ({len(output)} chars, ~{len(output)//4} tokens)")
        except FileNotFoundError:
            print("❌ xclip not found. Install with: sudo apt install xclip")
            print(output)
    elif "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            Path(sys.argv[idx + 1]).write_text(output)
            print(f"✅ Written to {sys.argv[idx + 1]}")
        else:
            print("❌ --output requires a filename")
    else:
        print(output)
    
    # Always show stats
    print(f"\n--- Stats: {len(output)} chars | ~{len(output)//4} tokens ---")

if __name__ == "__main__":
    main()
