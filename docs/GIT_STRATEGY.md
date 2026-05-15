# Git Strategy - Silicon Mind

This project uses separate development branches so software and hardware work
can move independently while `main` stays portfolio quality.

## Branch Model

| Branch | Owner | Purpose | Merge policy |
|:--|:--|:--|:--|
| `main` | Shared | Stable reviewed milestones | Only reviewed, verified work |
| `dev` | Aaditya | Software, ML, verification, driver work | Reviewed before merge to `main` |
| `hw-dev` | Shaurya | RTL, simulations, synthesis work | Reviewed before merge to `main` |

Development branches may contain work in progress. `main` should not contain
empty placeholders, broken verification, accidental reversions, or unreviewed
shared-file changes.

## Folder Ownership

```text
model/          Aaditya - training, quantization, export, golden model
verification/   Aaditya - cocotb tests and expected outputs
driver/         Aaditya - RISC-V C driver
scripts/        Aaditya - utility scripts
rtl/            Shaurya - Verilog RTL
sim/            Shaurya - Verilog testbenches
synth/          Shaurya - synthesis scripts and constraints
data/           Shared - exported weights, metadata, test vectors
docs/           Shared - architecture and interface documentation
results/        Shared - generated reports and benchmark outputs
```

Shared files should be changed intentionally and reviewed before merge.

## Daily Workflow - Aaditya

```bash
git checkout dev
git pull --ff-only origin dev

# make software changes

git status --short
git add <intentional paths>
git commit -m "feat: short software change"
git push origin dev
```

## Daily Workflow - Shaurya

```bash
git checkout hw-dev
git fetch origin
git rebase origin/main

# make hardware changes in rtl/, sim/, synth/

git status --short
git add rtl sim synth
git commit -m "feat: short hardware change"
git push origin hw-dev
```

If `hw-dev` has already diverged from GitHub, use:

```bash
git fetch origin
git rebase origin/hw-dev
git push origin hw-dev
```

If a rebase was required and Git asks for a force push, use the safer form:

```bash
git push --force-with-lease origin hw-dev
```

Do not use plain `--force`.

## Weekly Review

Before merging to `main`:

```bash
git fetch --prune origin
```

Review:

- local uncommitted work
- commits on `origin/dev` and `origin/hw-dev`
- changed files and deleted files
- edits outside each contributor's owned folders
- software verification output
- hardware implementation status
- public documentation cleanliness

Textual merge conflicts are not the only risk. A branch can merge cleanly while
deleting important files or reverting shared documentation.

## Merge To Main

Merge only branches that passed review:

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff origin/dev -m "merge: software milestone"
git push origin main
git checkout dev
```

For hardware:

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff origin/hw-dev -m "merge: hardware milestone"
git push origin main
git checkout dev
```

If one branch is unsafe, leave it unmerged and merge only the safe branch.

## Commit Message Convention

Use:

```text
type: brief engineering change
```

Examples:

```text
feat: add quantized weight export
fix: repair exported model metadata verification
docs: clarify accelerator register interface
test: add golden model verification case
synth: add yosys synthesis script
chore: update ignore rules
```

Allowed types:

- `feat`
- `fix`
- `docs`
- `test`
- `synth`
- `refactor`
- `chore`
- `wip` on development branches only

## What Not To Commit

- virtual environments
- Python caches
- model checkpoints unless intentionally versioned through large-file storage
- generated waveforms
- generated netlists
- FPGA project build directories
- private local workflow notes
- temporary logs or scratch files

## Main Branch Standard

Before code enters `main`, it should be:

- relevant to the project milestone
- reviewed against folder ownership and interface contracts
- free of accidental shared-file reversions
- verified with the appropriate quick checks
- documented where the hardware-software contract changes
