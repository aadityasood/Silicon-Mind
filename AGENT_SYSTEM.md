# 🤖 4-Agent Coding System — Silicon Mind

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOU (Aaditya) — The Router                    │
│                                                                 │
│  1. Run condense_context.py → get context blob                  │
│  2. Paste context + task into Flash chat                        │
│  3. Copy Flash output → paste into Pro chat                     │
│  4. Copy Pro output → paste into Opus Coder chat                │
│  5. Opus Coder writes final code to project files               │
│  6. Commit to dev: git add -A && git commit && git push         │
│  7. Weekend: run ./weekly → paste into Opus Overseer            │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  FLASH   │─►│   PRO    │─►│  OPUS    │  │    OPUS      │   │
│  │ (Writer) │  │(Auditor) │  │ (Coder)  │  │ (Overseer)   │   │
│  │          │  │          │  │          │  │              │   │
│  │ Gemini   │  │ Gemini   │  │ Claude/  │  │ THIS CHAT    │   │
│  │ Flash    │  │ Pro      │  │ Antigrav │  │ (Antigravity)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                 │
│  Code Flow: Flash → Pro → Opus Coder → project files           │
│  Management: Opus Overseer (weekly review, git, planning)       │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Responsibilities

| Agent | Does | Does NOT |
|:------|:-----|:---------|
| **Flash** | Writes first draft code, follows specs | Review code, make architecture decisions |
| **Pro** | Audits Flash's code, finds bugs, improves | Write from scratch, make architecture decisions |
| **Opus Coder** | Final code review, writes to project files | Manage project, handle git, track progress |
| **Opus Overseer** | Weekly review, git merges, planning, Shaurya sync | Write application code |

## When to Use the Full Chain vs. Skip Steps

| Task | Chain | Why |
|:-----|:------|:----|
| Simple utility function | Flash → Opus Coder | Pro audit unnecessary for trivial code |
| CNN training script | Flash → Pro → Opus Coder | Standard, benefits from review |
| Golden model / quantization | Flash → Pro → Opus Coder | 🔴 Critical — must be bit-exact |
| im2col implementation | Flash → Pro → Opus Coder | 🔴 Critical — hardware interface |
| Test vectors / cocotb tests | Flash → Pro → Opus Coder | 🔴 Critical — verifies hardware |
| Documentation updates | Flash alone or Opus Overseer | Low risk |
| Bug fix (known issue) | Opus Coder alone | Already knows the system |
| Architecture decision | Opus Overseer | Needs project-wide context |

## Context Flow

```
PROJECT_CONTEXT.md ──┐
DECISIONS_LOG.jsonl ──┼──► condense_context.py ──► ~808 token blob
interface_spec.md ────┘                                │
                                                       ▼
                                            Paste into ANY agent chat
                                            (gives instant project context)
```

## Commands Reference

### Daily
```bash
# Get context for agents
python3 condense_context.py

# Commit daily work
git add -A && git commit -m "feat: description" && git push origin dev
```

### Weekend
```bash
# Generate weekly review
./weekly

# Paste output into Opus Overseer chat
# After Overseer approves:
git checkout main && git merge dev && git push origin main
git checkout dev
```

## Detailed Prompt Templates

All copy-paste-ready prompts for each agent are in:
**[docs/CHAT_SETUP.md](docs/CHAT_SETUP.md)**

That file contains:
- Flash setup prompt (paste as first message)
- Pro setup prompt (paste as first message)
- Opus Coder setup prompt (paste as first message)
- Opus Overseer description (this chat, no setup needed)
- Step-by-step daily workflow
- Step-by-step weekend workflow

## Verilog Review Prompt (For Shaurya's Code)

When Shaurya sends you his Verilog, use this prompt in any Opus chat:

```
=== VERILOG REVIEW ===

Review this Verilog module for the Silicon Mind project (FPGA systolic array accelerator).

[PASTE condense_context.py output HERE]

=== VERILOG CODE ===
[PASTE SHAURYA'S CODE HERE]

=== CHECK ===
1. Synthesizability: no latches, proper resets, no initial blocks in logic
2. Signed arithmetic: `signed` keyword on INT8 wires
3. Accumulator: 32-bit wide for INT8×INT8 MAC
4. Data flow: activations go RIGHT, partial sums go DOWN (weight-stationary)
5. Parameterization: uses `parameter ARRAY_SIZE`, not hardcoded
6. Interface: matches interface_spec.md signal names and timing
7. Timing: results at correct cycle count for ARRAY_SIZE pipeline depth

Output: VERDICT + ISSUES + IMPROVED VERILOG + cocotb test suggestion
```

## Session Handoff Protocol

### Ending a Session
```
1. Save all files
2. Tell Opus Overseer (this chat): "Session done — I worked on [X]"
   → I update PROJECT_CONTEXT.md and DECISIONS_LOG.jsonl
3. git add -A && git commit -m "type: description"
4. git push origin dev
```

### Starting a New Session
```
1. cd "/run/media/kulfi/Stuff/Adi's Stuff/silicon-mind"
2. python3 condense_context.py  (copy the output)
3. Open Flash/Pro/Opus Coder chat
4. Paste context blob + task
5. Start coding
```

### Handing Context to Shaurya
```bash
python3 condense_context.py > context_for_shaurya.txt
# Send him: context_for_shaurya.txt + any new .mem files + interface_spec.md
```
