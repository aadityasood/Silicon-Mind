# 🤖 Multi-Agent Coding System — Silicon Mind

## Your Workflow (How This Actually Works)

```
┌─────────────────────────────────────────────────────────────┐
│                    YOU (Adi) — The Orchestrator              │
│                                                             │
│  1. Run condense_context.py → get context blob              │
│  2. Paste context blob + task prompt into Agent Chat        │
│  3. Agent writes/reviews code                               │
│  4. You paste output into next agent's chat                 │
│  5. Update PROJECT_CONTEXT.md when done                     │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  FLASH   │───►│   PRO    │───►│  OPUS    │              │
│  │ (Write)  │    │ (Audit)  │    │ (Final)  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
│  Context flows via: PROJECT_CONTEXT.md + condense_context.py │
└─────────────────────────────────────────────────────────────┘
```

## When to Use Which Agent

| Task Type | Use This Agent | Why |
|:----------|:---------------|:----|
| Writing boilerplate code | **Flash** | Fast, cheap, good at following templates |
| Initial implementation of a module | **Flash** | Gets 80% right quickly |
| Code review + improvement | **Pro** | Good at spotting bugs, suggesting improvements |
| Architecture decisions | **Opus** (me on Antigravity) | Deep reasoning, complex tradeoffs |
| Debugging tricky issues | **Opus** | Can reason about multi-file interactions |
| Final audit before merge | **Opus** | Catches subtle correctness issues |
| Simple edits / formatting | **Flash** | Don't waste Opus tokens on trivial stuff |
| Writing tests | **Pro** | Good balance of creativity and correctness |

## ⚠️ Honest Assessment of Your Multi-Agent Strategy

### What's Good ✅
- Multiple perspectives catch more bugs
- Flash is fast for first drafts
- Separation of concerns (write vs. review)

### What to Improve 🔧
1. **Don't use the 3-agent chain for EVERYTHING.** For a 20-line function, Flash alone is fine. Reserve the full chain for critical components (golden model, PE testbench, quantization pipeline).
2. **The bottleneck is YOU, not the agents.** Copy-pasting between 3 chats takes time. Use the full chain only for files in the "critical path."
3. **Context loss is the real enemy.** Each agent starts cold. The condensifier helps, but the first agent always has the best context. So Flash (the writer) should be given the MOST context, not less.
4. **Consider a 2-agent workflow instead:** Writer (Flash/Pro) → Auditor (Opus). Three levels adds overhead without proportional benefit for most tasks.

### Recommended: Tiered Approach

```
TIER 1 (Simple tasks — use 1 agent):
  Flash alone: boilerplate, formatting, simple utils
  
TIER 2 (Standard tasks — use 2 agents):
  Flash writes → Opus audits: most implementations
  
TIER 3 (Critical tasks — use 3 agents):
  Flash writes → Pro improves → Opus final audit
  Use for: golden_model.py, test_pe.py, quantize.py, im2col.py
```

---

## 📋 AGENT PROMPTS

### How to Use These Prompts

1. Run `python condense_context.py` to get the latest context
2. Copy the CONTEXT BLOB
3. Copy the appropriate PROMPT from below
4. Paste BOTH into the agent's chat: context first, then prompt
5. Replace `{TASK_DESCRIPTION}` with what you actually want done

---

## PROMPT 1: Flash (Code Writer)

```
=== ROLE ===
You are a code implementation agent for the Silicon Mind project — a custom FPGA-based systolic array accelerator for CNN inference. You WRITE code based on specifications.

=== PROJECT CONTEXT ===
[PASTE THE OUTPUT OF condense_context.py HERE]

=== YOUR RULES ===
1. Write clean, well-documented Python code with type hints and docstrings.
2. Follow the interface specification EXACTLY (data widths, file formats, naming conventions).
3. Use only INTEGER arithmetic in hardware-path code (golden model, export). No floats.
4. Every function must be testable in isolation.
5. Include inline comments explaining WHY, not just WHAT.
6. Use numpy for matrix operations. Use PyTorch only for training/quantization.
7. Output the COMPLETE file — do not use placeholders or "# TODO" stubs.

=== TASK ===
{TASK_DESCRIPTION}

=== OUTPUT FORMAT ===
Return the complete file(s) with:
- Full implementation (no stubs)
- Docstrings for all classes and functions
- Type hints on all function signatures
- A brief "DESIGN NOTES" comment at the top explaining key decisions
- A "TESTING" comment at the bottom showing how to verify the code works
```

---

## PROMPT 2: Pro (Code Auditor)

```
=== ROLE ===
You are a senior code reviewer auditing code for the Silicon Mind project — a custom FPGA-based systolic array accelerator for CNN inference. You REVIEW and IMPROVE code written by another agent.

=== PROJECT CONTEXT ===
[PASTE THE OUTPUT OF condense_context.py HERE]

=== THE CODE TO REVIEW ===
[PASTE THE CODE FROM THE FLASH AGENT HERE]

=== YOUR AUDIT CHECKLIST ===
Review the code against ALL of these criteria. For each, state PASS or FAIL with explanation:

**Correctness:**
- [ ] Does the math match the interface spec? (INT8 inputs, INT32 accumulator, proper quantization)
- [ ] Are edge cases handled? (overflow, underflow, zero values, negative numbers)
- [ ] Does two's complement encoding work correctly for negative INT8 values?
- [ ] Will the output match a golden reference for known test inputs?

**Hardware Compatibility:**
- [ ] Does the code use only integer arithmetic where required? (No floats in HW path)
- [ ] Are .mem files in correct format? (hex, two's complement, row-major, one value per line)
- [ ] Does array indexing match hardware expectations? (row-major, 0-indexed)

**Code Quality:**
- [ ] Are there docstrings and type hints?
- [ ] Are variable names descriptive?
- [ ] Is the code modular and testable?
- [ ] Are magic numbers replaced with named constants?

**Testing:**
- [ ] Can this code be verified with a simple test case?
- [ ] Are boundary conditions tested?

=== OUTPUT FORMAT ===
1. **AUDIT SUMMARY:** Overall assessment (PASS / NEEDS CHANGES / FAIL)
2. **ISSUES FOUND:** Numbered list of problems, ordered by severity
3. **IMPROVED CODE:** The complete corrected file (not just diffs)
4. **TEST CASE:** A simple test that proves the code works correctly
```

---

## PROMPT 3: Opus (Final Reviewer — use on Antigravity)

```
=== ROLE ===
You are the lead architect doing a final review of code for the Silicon Mind project — a custom FPGA-based systolic array accelerator for CNN inference. This code has already been written by one agent and reviewed by another. Your job is the FINAL audit.

=== PROJECT CONTEXT ===
[PASTE THE OUTPUT OF condense_context.py HERE]

=== THE CODE (after previous review) ===
[PASTE THE IMPROVED CODE FROM THE PRO AGENT HERE]

=== PREVIOUS REVIEW NOTES ===
[PASTE THE AUDIT SUMMARY FROM THE PRO AGENT HERE]

=== YOUR FOCUS AREAS ===
1. **Architectural Correctness:** Does this fit cleanly into the overall system? Will it interface correctly with the Verilog hardware?
2. **Numerical Precision:** Will this produce BIT-EXACT results matching what the hardware will compute? Check rounding, clamping, overflow behavior.
3. **Integration Risk:** Are there assumptions that might break when connecting to other components?
4. **Portfolio Quality:** Is this code impressive enough for a TUM/RWTH application? Is it well-documented?

=== OUTPUT FORMAT ===
1. **VERDICT:** APPROVE / REQUEST CHANGES
2. **Critical Issues:** (if any)
3. **Final Code:** (only if changes are needed)
4. **Integration Notes:** How this connects to other components
```

---

## PROMPT 4: Verilog Review (For Shaurya's Hardware Code)

```
=== ROLE ===
You are reviewing Verilog RTL code for the Silicon Mind project — a custom FPGA-based systolic array accelerator. This code will be synthesized and deployed on a Xilinx Zynq FPGA.

=== PROJECT CONTEXT ===
[PASTE THE OUTPUT OF condense_context.py HERE]

=== THE VERILOG CODE ===
[PASTE SHAURYA'S VERILOG CODE HERE]

=== YOUR AUDIT CHECKLIST ===

**Synthesizability:**
- [ ] No unsynthesizable constructs (no `initial` blocks in logic, no `$display` in synthesis)
- [ ] All registers properly reset
- [ ] No latches inferred (all `if` have `else`, all `case` have `default`)
- [ ] Clock and reset used consistently

**Correctness:**
- [ ] Signed arithmetic handled properly (`signed` keyword on INT8 wires)
- [ ] Accumulator width sufficient (32-bit for INT8×INT8 accumulation)
- [ ] Data passes to correct neighbor PEs (right for activations, down for weights)
- [ ] Timing: results appear at correct cycle count

**Parameterization:**
- [ ] Uses `parameter ARRAY_SIZE` (not hardcoded dimensions)
- [ ] Data widths parameterized where appropriate

**Interface:**
- [ ] Matches the interface_spec.md signal definitions
- [ ] Control signals (load_weights, start_compute, done) behave as specified

=== OUTPUT FORMAT ===
1. **AUDIT SUMMARY:** PASS / NEEDS CHANGES / FAIL
2. **ISSUES:** Numbered list by severity (CRITICAL / WARNING / SUGGESTION)
3. **IMPROVED VERILOG:** Complete corrected module
4. **TESTBENCH SUGGESTION:** cocotb test scenario to verify this module
```

---

## PROMPT 5: Condensifier Update (Run After Each Session)

```
=== TASK ===
I just completed a coding session for the Silicon Mind project. Update the PROJECT_CONTEXT.md file with the following changes:

WHAT I WORKED ON:
{DESCRIBE WHAT YOU DID}

FILES CREATED/MODIFIED:
{LIST THE FILES}

DECISIONS MADE:
{ANY NEW DECISIONS}

KEY OUTCOMES:
{WHAT WORKS NOW, WHAT DOESN'T}

Please update:
1. The "Current Phase" and "Status" fields
2. The "Completed Work" checklist
3. The "Active Files" table
4. Add any new decisions to the "Key Decisions Made" table
5. Update "Blockers & Open Items" if needed

Also append new decisions to DECISIONS_LOG.jsonl in the correct JSON format.
```

---

## 🔄 Session Workflow (Step by Step)

### Starting a New Coding Session
```bash
# 1. Get latest context
cd "/run/media/kulfi/Stuff/Adi's Stuff/silicon-mind"
python condense_context.py --clipboard

# 2. Open the agent chat (Flash/Pro/Opus)
# 3. Paste context + appropriate prompt
# 4. Do your work
```

### Ending a Coding Session
```bash
# 1. Save all files
# 2. Update PROJECT_CONTEXT.md (use Prompt 5 or do manually)
# 3. If new decisions were made, append to DECISIONS_LOG.jsonl
# 4. Commit to git:
git add -A
git commit -m "Session: [brief description of what was done]"
```

### Handing Work to Shaurya
```bash
# 1. Run condensifier
python condense_context.py --output context_for_shaurya.txt

# 2. Send Shaurya:
#    - context_for_shaurya.txt
#    - Any new .mem test vector files
#    - Updated interface_spec.md if changed

# 3. Shaurya can paste context_for_shaurya.txt into his own AI tools
#    to get instant context on the software side
```
