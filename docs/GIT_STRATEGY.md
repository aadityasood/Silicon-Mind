# 🌿 Git Strategy — Silicon Mind
<!-- A beginner-friendly Git workflow designed for this project -->

## TL;DR — When to Commit

| When | What to Commit | Branch |
|:-----|:---------------|:-------|
| **After Opus approves code** | The approved file(s) | `main` |
| **End of Adi's coding session** | WIP progress (even if incomplete) | `dev` |
| **End of Shaurya's coding session** | WIP hardware progress | `hw-dev` |
| **Weekend review** | Cleanup + merge approved work | `main` (merge from dev & hw-dev) |
| **Major milestone** | Tag with version | `main` |

## Branching Strategy (3-Branch Model)

```
main ──────●────────────────●────────────────●─────────── (stable, reviewed)
            \              / \              /
dev ─────────●───●───●────   │              │   (Adi: model/, verification/)
                              \            /
hw-dev ────────────●───●───●──────●───●────    (Shaurya: rtl/, sim/, synth/)
```

**THREE branches:**
- **`main`** — Contains only Opus-reviewed, working code from BOTH contributors. This is what goes on your GitHub profile. Every commit here should compile/run without errors.
- **`dev`** — Adi's daily software work branch. Commit freely here, even broken code.
- **`hw-dev`** — Shaurya's daily hardware work branch. Commit freely here, even broken code.

### Who Owns What

| Branch | Owner | Contents | Pushes |
|:-------|:------|:---------|:-------|
| `main` | Both (via Adi's merge) | Reviewed, stable code | Only after `./weekly` review |
| `dev` | Adi | `model/`, `verification/`, `driver/`, `scripts/`, docs | Adi pushes daily |
| `hw-dev` | Shaurya | `rtl/`, `sim/`, `synth/` | Shaurya pushes daily |

### Folder Ownership (No Conflicts)

```
silicon-mind/
├── model/            ← Adi (PyTorch training, quantization)
├── verification/     ← Adi (cocotb Python tests)
├── driver/           ← Adi (PYNQ driver)
├── scripts/          ← Adi (utility scripts)
├── rtl/              ← Shaurya (Verilog modules: pe.v, systolic_array.v)
├── sim/              ← Shaurya (Verilog testbenches)
├── synth/            ← Shaurya (Yosys/Vivado synthesis scripts, constraints)
├── data/             ← Shared (.mem weight files, test vectors)
├── docs/             ← Shared (interface_spec.md, etc.)
└── results/          ← Shared (simulation results, synthesis reports)
```

> **Rule:** Adi never touches `rtl/`, `sim/`, `synth/`. Shaurya never touches `model/`, `verification/`, `driver/`. Shared folders (`data/`, `docs/`, `results/`) are coordinated via the weekly review.

## Step-by-Step Git Guide

### Initial Setup — Adi (Done ✅)
```bash
cd "/run/media/kulfi/Stuff/Adi's Stuff/silicon-mind"
git init
git add -A && git commit -m "Initial project structure"
git checkout -b dev
git remote add origin https://github.com/aadityasood/Silicon-Mind.git
git push -u origin main
git push -u origin dev
```

### Initial Setup — Shaurya (Done ✅)
```bash
git clone https://github.com/aadityasood/Silicon-Mind.git
cd Silicon-Mind
git checkout -b hw-dev
git push -u origin hw-dev
# Now work in rtl/, sim/, synth/ and push to hw-dev
```

### Daily Workflow — Adi
```bash
# Make sure you're on dev
git checkout dev

# After a coding session
git add -A
git commit -m "feat: description"
git push origin dev
```

### Daily Workflow — Shaurya
```bash
# Make sure you're on hw-dev
git checkout hw-dev

# After a coding session
git add -A
git commit -m "feat: description"
git push origin hw-dev
```

### Weekend Review Workflow (After Opus Reviews Both Branches)
```bash
# Step 1: Run the weekly review generator
./weekly

# Step 2: Paste output into the Opus Overseer (Antigravity) chat
# Opus will review both branches and tell you which to merge

# Step 3: If Opus approves dev:
git checkout main
git merge origin/dev
git push origin main

# Step 4: If Opus approves hw-dev:
git checkout main
git merge origin/hw-dev
git push origin main

# Step 5: Switch back to dev for next week
git checkout dev

# Step 6: Tell Shaurya to rebase his branch (if main was updated)
# Shaurya runs: git checkout hw-dev && git rebase main && git push -f origin hw-dev
```

### Milestone Tags
```bash
# After completing a major milestone
git tag -a v0.1 -m "Phase 1: CNN trained and quantized"
git push origin --tags
```

## Commit Message Convention

Use this format:
```
type: brief description

Examples:
feat: add MNIST CNN training script
feat: implement MAC unit in Verilog
fix: correct INT8 overflow in golden model
fix: add signed keyword to PE accumulator
docs: update interface specification
test: add PE unit tests with cocotb
test: add Verilog testbench for MAC
synth: update Yosys synthesis script
refactor: parameterize array size in golden model
chore: update .gitignore for Vivado files
```

Types:
- `feat` — New feature or file
- `fix` — Bug fix
- `docs` — Documentation changes
- `test` — Adding or updating tests
- `synth` — Synthesis scripts and constraints
- `refactor` — Code restructuring (no behavior change)
- `chore` — Build scripts, configs, maintenance
- `wip` — Work in progress (dev/hw-dev branches only)

## What NOT to Commit
These are in `.gitignore` but worth remembering:
- ❌ Vivado project files (`.xpr`, `.runs/`, `.cache/`)
- ❌ Bitstream files (`.bit`) — too large
- ❌ Yosys output netlists (`*_gate.v`, `*.blif`)
- ❌ Python `__pycache__/` and `.pyc` files
- ❌ Virtual environment (`.venv/`)
- ❌ Large datasets (use Git LFS if needed)
- ❌ Model checkpoints (`.pth` files) — use Git LFS
- ❌ Simulation waveforms (`.vcd`, `.fst`)

## Milestone Schedule

| Milestone | Tag | When | What Should Be Working |
|:----------|:----|:-----|:-----------------------|
| **v0.1** | `v0.1-cnn-trained` | After Phase 1 | CNN trains on MNIST, INT8 quantization works |
| **v0.2** | `v0.2-golden-model` | After Phase 2a | im2col + golden model produce correct matmul results |
| **v0.3** | `v0.3-pe-verified` | After Phase 2b | Verilog PE matches golden model bit-for-bit |
| **v0.4** | `v0.4-array-verified` | After Phase 2c | 4×4 systolic array passes all test vectors |
| **v0.5** | `v0.5-full-system` | After Phase 3 | Controller + memory + top-level integrated |
| **v0.6** | `v0.6-synthesis` | After Phase 4a | Vivado synthesis passes, timing met |
| **v0.7** | `v0.7-fpga-demo` | After Phase 4b | End-to-end inference on FPGA board |
| **v1.0** | `v1.0-cifar10` | Final | CIFAR-10 model running on FPGA |

## Answer: Commit Immediately or After Review?

**Both, on different branches:**
- **Commit to `dev`/`hw-dev` IMMEDIATELY** after every coding session. Even broken code. This is your safety net — you can always go back.
- **Commit to `main` only AFTER Opus weekend review.** This keeps `main` clean and professional.

Your GitHub profile shows `main` branch commits. So `main` = portfolio quality. `dev`/`hw-dev` = your working notebooks.

> **Pro tip:** GitHub shows your commit activity as green squares on your profile. Committing to `dev` or `hw-dev` daily keeps those squares green, showing consistent work — admissions committees notice this.
