# RTL

Synthesizable hardware source for Silicon Mind lives here.

Keep this directory focused on design files: CPU blocks, accelerator blocks,
interconnect glue, and small reusable modules. Testbenches belong in `sim/`;
tool scripts and generated netlists belong outside the RTL source tree.

Conventions:
- One main module per file when possible.
- Name files after the module they define, for example `mac.v` or `pe.v`.
- Prefer small blocks that can be simulated on their own before integration.
- Keep generated files, datasets, and scratch notes out of this folder.
