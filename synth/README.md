# Synthesis

Synthesis scripts, constraints, and notes live here.

This directory should make it clear how the hand-written RTL is intended to be
turned into a gate-level result. Keep scripts reviewable and use relative paths
back to `rtl/` so another checkout can reproduce the same flow.

Conventions:
- Commit scripts and constraints, not generated outputs.
- Keep tool-specific files named clearly, for example `yosys.ys`.
- Write outputs to ignored build/report locations.
- Do not commit generated netlists, logs, reports, or bitstreams.
