# Simulation

RTL testbenches and simulation helpers live here.

Use this directory for quick checks while the hardware is still changing:
module-level testbenches first, then integration tests once the pieces start
connecting cleanly.

Conventions:
- Name testbenches after the block under test, for example `tb_mac.v`.
- Keep simulator commands simple and relative to the repo root.
- Commit source testbenches and small input vectors only.
- Do not commit waveform dumps, simulator build folders, or temporary logs.
