# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
make sync       # Install deps with the CUDA 13 torch build (sync-cpu for CPU-only)
```

Running tests and tasks:

```sh
uv run pytest tests/<test_file>.py::<test_name>  # Single test
uv run pytest tests/<test_file>.py -k <pattern>  # Tests matching a pattern
FORCE_CPU=1 uv run pytest -m "not slow"          # CPU-only (same as make test-cpu-fast)

uv run list-envs                                 # All registered task IDs
uv run train <task-id> --env.scene.num-envs 4096 # Any config field is a CLI flag
uv run train <task-id> --gpu-ids None            # Train on CPU
uv run play <task-id> --agent zero               # Sanity-check an MDP without a policy
```

Tests pick the device through `get_test_device()` in `tests/conftest.py` (CUDA if
available, CPU with `FORCE_CPU=1`). Tests marked `slow` are skipped by `make test-fast`.
`train` logs to wandb by default; pass `--agent.logger tensorboard` to stay local.

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# Architecture

mjlab is Isaac Lab's manager-based API on top of MuJoCo Warp. `docs/source/` has a page
per subsystem; `architecture_overview.rst` is the map. The code has two layers.

**Simulation layer** (`entity/`, `actuator/`, `sensor/`, `scene/`, `sim/`, `terrains/`):

- An `EntityCfg` starts from an MJCF loaded by `spec_fn` and layers Python config on
  the `MjSpec`: actuators, collision rules, initial state. The XML is not edited.
- `Scene` merges all entities, terrain and sensors into one spec, prefixing entity
  names (`robot/base_link`). Sensors defined in an entity's XML become scene sensors
  named `<entity>/<sensor>`.
- `Simulation` compiles the spec and steps it in MuJoCo Warp with a leading world
  dimension, one world per environment.

**Manager layer** (`envs/`, `managers/`): `ManagerBasedRlEnvCfg` is a dataclass of
dicts, one per manager (observations, actions, commands, events, rewards, terminations,
curriculum, metrics). Each entry is a term config with a `func` and `params`. Terms
are plain functions, or classes when they cache setup or keep per-episode state.
`SceneEntityCfg` resolves name regexes to ids. Shared terms live in `envs/mdp/`;
task-specific ones in `tasks/<family>/mdp/`.

**Tasks** (`tasks/`): each family (`velocity`, `tracking`, `manipulation`) has a
`make_*_env_cfg()` factory with robot-specific fields left blank. A robot package under
`tasks/<family>/config/<robot>/` calls the factory, fills those in, and registers task
IDs with `register_mjlab_task` in its `__init__.py`. `tasks/__init__.py` imports every
subpackage, so a new config package registers itself.

**Robots** (`asset_zoo/robots/<robot>/`): MJCF and meshes under `xmls/`, plus a
`*_constants.py` that defines actuator groups, keyframe, `CollisionCfg`s,
`get_<robot>_robot_cfg()` and `<ROBOT>_ACTION_SCALE`. Tasks expect collision geoms
named `*_collision`, foot sites, and the `imu_*` / `root_angmom` sensors in the XML.

**Training** (`rl/`, `scripts/`): `train` and `play` wrap the env for RSL-RL. The CLI
is generated from the config dataclasses with tyro.

Things that are easy to get wrong:

- Control step = `sim.mujoco.timestep * decimation`. Rewards are scaled by the control
  step; terms that count physics substeps (contact sensor `history_length`) are not.
- Joint position actions follow joint order in the XML, not actuator ctrl order.
- Every task registers a training config and a play config;
  `load_env_cfg(task_id, play=True)` returns the play one.
