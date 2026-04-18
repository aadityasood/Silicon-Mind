# 🌿 Git Strategy — Silicon Mind
<!-- A beginner-friendly Git workflow designed for this project -->

## TL;DR — When to Commit

| When | What to Commit | Branch |
|:-----|:---------------|:-------|
| **After Opus approves code** | The approved file(s) | `main` |
| **End of each coding session** | WIP progress (even if incomplete) | `dev` |
| **Weekend review** | Any cleanup + review notes | `main` (merge from dev) |
| **Major milestone** | Tag with version | `main` |

## Branching Strategy (Keep It Simple)

```
main ──────●─────────────●─────────────●─────────── (stable, reviewed)
            \           / \           /
dev ─────────●───●───●──   ●───●───●──              (daily work)
```

**Only TWO branches:**
- **`main`** — Contains only Opus-reviewed, working code. This is what goes on your GitHub profile. Every commit here should compile/run without errors.
- **`dev`** — Your daily work branch. Commit freely here, even broken code. Merge into `main` only after Opus weekend review.

## Step-by-Step Git Guide (For a Beginner)

### Initial Setup (Do This Once)
```bash
cd "/run/media/kulfi/Stuff/Adi's Stuff/silicon-mind"

# Initialize the repo
git init

# Create your first commit on main
git add -A
git commit -m "Initial project structure and documentation"

# Create dev branch for daily work
git checkout -b dev

# Connect to GitHub (after creating repo on github.com)
git remote add origin https://github.com/YOUR_USERNAME/silicon-mind.git
git push -u origin main
git push -u origin dev
```

### Daily Workflow
```bash
# Make sure you're on dev branch
git checkout dev

# After a coding session, stage and commit
git add -A
git commit -m "Session: [brief description]"

# Push to GitHub
git push origin dev
```

### Weekend Review Workflow (After Opus Approves)
```bash
# Switch to main
git checkout main

# Merge dev into main
git merge dev

# Push the clean, reviewed code
git push origin main

# Switch back to dev for next week
git checkout dev
```

### Milestone Tags
```bash
# After completing a major milestone (e.g., CNN training works)
git tag -a v0.1 -m "Phase 1: CNN trained and quantized"
git push origin --tags
```

## Commit Message Convention

Use this format:
```
type: brief description

Examples:
feat: add MNIST CNN training script
feat: implement im2col transform
fix: correct INT8 overflow in golden model
docs: update interface specification
test: add PE unit tests with cocotb
refactor: parameterize array size in golden model
chore: update .gitignore for Vivado files
```

Types:
- `feat` — New feature or file
- `fix` — Bug fix
- `docs` — Documentation changes
- `test` — Adding or updating tests
- `refactor` — Code restructuring (no behavior change)
- `chore` — Build scripts, configs, maintenance
- `wip` — Work in progress (dev branch only)

## What NOT to Commit
These are in `.gitignore` but worth remembering:
- ❌ Vivado project files (`.xpr`, `.runs/`, `.cache/`)
- ❌ Bitstream files (`.bit`) — too large
- ❌ Python `__pycache__/` and `.pyc` files
- ❌ Virtual environment (`.venv/`)
- ❌ Large datasets (use Git LFS if needed)
- ❌ Model checkpoints (`.pth` files) — use Git LFS

## Milestone Schedule

| Milestone | Tag | When | What Should Be Working |
|:----------|:----|:-----|:----------------------|
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
- **Commit to `dev` IMMEDIATELY** after every coding session. Even broken code. This is your safety net — you can always go back.
- **Commit to `main` only AFTER Opus weekend review.** This keeps `main` clean and professional.

Your GitHub profile shows `main` branch commits. So `main` = portfolio quality. `dev` = your working notebook.

> **Pro tip:** GitHub shows your commit activity as green squares on your profile. Committing to `dev` daily keeps those squares green, showing consistent work — admissions committees notice this.
