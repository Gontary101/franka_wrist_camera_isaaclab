# AGENTS.md

This repository is an Isaac Lab data-collection and evaluation environment for Franka tabletop manipulation.

## Architecture rules

Keep modules small and single-purpose.

Do not put task logic, policy logic, dataset writing, and simulator launching in the same file.

Use configuration files under `configs/` for experiment/task parameters. Do not add CLI arguments unless the value must change per invocation, such as config path, headless mode, device, or output directory.

Do not add broad `try/except` blocks. If a failure should stop data collection, let it fail clearly. Only catch exceptions when the code can recover in a specific, tested way.

Do not add fallback behavior that silently changes semantics. No hidden alternate camera paths, no silent object respawn, no automatic task substitution, no ignored failed resets.

Use dataclasses for typed configs and episode/task records.

Keep all randomization seeded and recorded in episode metadata.

Every episode must record:
- task name
- language instruction
- seed
- success flag
- timestamps
- camera frame paths or arrays
- robot state
- action representation
- object poses
- randomization metadata

## File ownership

`scene/` owns Isaac Lab scene construction and assets.

`tasks/` owns task definitions, reset sampling, language templates, and success checks.

`policies/` owns scripted demonstrators.

`control/` owns IK, gripper control, motion primitives, and trajectory utilities.

`episode/` owns episode schemas, reset orchestration, and recording.

`export/` owns conversion to model-specific formats.

`scripts/` should only load configs and call package code.

## Code quality

Prefer explicit simple code over clever abstractions.

Do not introduce framework-like registries unless there are at least two concrete implementations using them.

No global mutable state except Isaac Sim application objects that must be global.

No hardcoded absolute paths. Use config values or paths relative to repo root.

No print spam in library code. Use concise logging from scripts.

Do not mix debug visualization with data collection logic.

Keep Isaac Sim compatibility patches isolated in `app/launcher.py`.

## Testing expectations

Pure Python modules must be testable without launching Isaac Sim.

Task sampling, language generation, success predicates, episode schema validation, and exporters should have unit tests.

Simulation-dependent tests should be smoke tests only:
- scene launches
- reset runs
- one scripted episode finishes
- one episode writes a valid dataset directory
