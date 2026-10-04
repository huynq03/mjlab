"""Tests specific to velocity tasks."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  GO1_ACTION_SCALE,
  MINIPI_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def minipi_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Mini-Pi velocity task IDs."""
  return [t for t in velocity_task_ids if "MiniPi" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_minipi_velocity_has_required_sensors(
  minipi_velocity_task_ids: list[str],
) -> None:
  """Mini-Pi velocity tasks should have feet/ground and self collision sensors."""
  assert len(minipi_velocity_task_ids) == 2
  for task_id in minipi_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    for name in ("feet_ground_contact", "self_collision", "foot_height_scan"):
      assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-MiniPi",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-MiniPi",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_minipi_velocity_has_correct_action_scale_and_timing(
  minipi_velocity_task_ids: list[str],
) -> None:
  """Mini-Pi velocity tasks should use MINIPI_ACTION_SCALE at a 50 Hz policy rate."""
  for task_id in minipi_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )
    assert joint_pos_action.scale == MINIPI_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected MINIPI_ACTION_SCALE"
    )

    # The stance chatters at the generic 5 ms timestep; see minipi env_cfgs.
    assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
    assert cfg.decimation == 10


def test_minipi_velocity_command_ranges(minipi_velocity_task_ids: list[str]) -> None:
  """Mini-Pi commands should stay within the fixed Mini-Pi range, with no ramp-up."""
  for task_id in minipi_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    assert twist_cmd.ranges.lin_vel_x == (-0.4, 0.6)
    assert twist_cmd.ranges.lin_vel_y == (-0.45, 0.45)
    assert twist_cmd.ranges.ang_vel_z == (-1.2, 1.2)
    assert twist_cmd.rel_standing_envs == pytest.approx(0.1)
    # Forward-only envs force vx >= 0.3 m/s, only the top of the range; kept off.
    assert twist_cmd.rel_forward_envs == 0.0
    assert "command_vel" not in cfg.curriculum
    assert twist_cmd.command_deadzone == pytest.approx(0.1)


def test_minipi_velocity_locomotion_finetune_config(
  minipi_velocity_task_ids: list[str],
) -> None:
  """Mini-Pi reward weights, pose tolerances, gait clock and reset events."""
  for task_id in minipi_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    rewards = cfg.rewards

    expected_weights = {
      "track_linear_velocity": 1.0,
      "track_angular_velocity": 1.0,
      "foot_gait": 0.75,
      "foot_clearance": -1.0,
      "foot_swing_height": -1.0,
      "foot_slip": -0.25,
      "action_rate_l2": -0.05,
      "stand_still": -1.0,
      "is_terminated": -200.0,
      "joint_acc_l2": -2.5e-7,
      "dof_pos_limits": -1.0,
      "soft_landing": -1e-5,
      "upright": 1.0,
      "angular_momentum": -0.02,
    }
    for name, weight in expected_weights.items():
      assert rewards[name].weight == pytest.approx(weight), name
    assert rewards["track_angular_velocity"].params["xy_weight"] == 0.05
    assert rewards["track_linear_velocity"].params["std"] == pytest.approx(0.2)
    for name in ["foot_clearance", "foot_swing_height"]:
      assert rewards[name].params["target_height"] == pytest.approx(0.05), name

    gait = rewards["foot_gait"].params
    assert gait["period"] == pytest.approx(0.5)
    assert gait["offset"] == [0.0, 0.5]
    assert gait["threshold"] == pytest.approx(0.55)
    assert gait["sensor_name"] == "feet_ground_contact"

    # Every stand/walk switch matches the command deadzone.
    assert rewards["pose"].params["walking_threshold"] == pytest.approx(0.1)
    for name in [
      "foot_gait",
      "stand_still",
      "foot_clearance",
      "foot_swing_height",
      "foot_slip",
      "soft_landing",
    ]:
      assert rewards[name].params["command_threshold"] == pytest.approx(0.1), name

    pose = rewards["pose"].params
    assert pose["std_standing"] == {".*": 0.05}
    assert pose["std_walking"] == {
      r".*_hip_pitch_joint": 0.8,
      r".*_hip_roll_joint": 0.25,
      r".*_thigh_joint": 0.3,
      r".*_calf_joint": 0.7,
      r".*_ankle_pitch_joint": 0.35,
      r".*_ankle_roll_joint": 0.15,
    }

    # One gait clock for the reward and the phase observation, in both groups.
    for group in ("actor", "critic"):
      names = list(cfg.observations[group].terms)
      assert names.count("phase") == 1
      assert names[names.index("command") + 1] == "phase"
      phase_params = cfg.observations[group].terms["phase"].params
      assert phase_params["period"] == gait["period"]
      assert phase_params["command_threshold"] == pytest.approx(0.1)

    # Foot order must match the right-then-left model order of the contact channels
    # and site ids, or foot_clearance pairs one foot's height with the other's speed.
    assert rewards["foot_clearance"].params["asset_cfg"].site_names == (
      "r_foot",
      "l_foot",
    )
    assert rewards["foot_slip"].params["asset_cfg"].site_names == ("r_foot", "l_foot")

    assert "push_robot" not in cfg.events
    assert cfg.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.0)


def test_minipi_velocity_self_collision_matches_g1_per_control_step(
  minipi_velocity_task_ids: list[str],
) -> None:
  """A full-step self-collision should cost Mini-Pi the same as G1 (4 x -1.0)."""
  for task_id in minipi_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    sensor = next(s for s in cfg.scene.sensors or () if s.name == "self_collision")
    assert isinstance(sensor, ContactSensorCfg)
    assert sensor.history_length == cfg.decimation == 10
    weight = cfg.rewards["self_collisions"].weight
    assert weight == pytest.approx(-0.4)
    assert weight * sensor.history_length == pytest.approx(-4.0)


def test_minipi_flat_commands_are_bounded_and_standing_is_zero() -> None:
  """Sampled commands should respect the limits; standing envs get exactly zero."""
  cfg = load_env_cfg("Mjlab-Velocity-Flat-MiniPi")
  cfg.scene.num_envs = 256
  cfg.seed = 0

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()

  term = env.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  command = term.command
  standing = term.is_standing_env

  assert standing.any() and not standing.all()
  assert (command[standing] == 0.0).all()
  assert ((command[:, 0] >= -0.4) & (command[:, 0] <= 0.6)).all()
  assert (command[:, 1].abs() <= 0.45).all()
  assert (command[:, 2].abs() <= 1.2).all()
  # Deadzone: every command is either exactly zero or clearly above 0.1.
  norm = command.norm(dim=1)
  assert ((norm == 0.0) | (norm > 0.1)).all()

  # Policy interface: 48D proprioception + command, plus the 2D gait phase -> 12D.
  obs = env.observation_manager.compute()
  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert actor_obs.shape == (256, 50)
  # Phase is [sin, cos] while moving and exactly zero while standing.
  phase_obs = actor_obs[:, 48:50]  # Right after the 3D command at 45:48.
  assert (phase_obs[norm == 0.0] == 0.0).all()
  moving_phase = phase_obs[norm > 0.0]
  assert torch.allclose(moving_phase.norm(dim=1), torch.ones(len(moving_phase)))
  assert env.action_manager.total_action_dim == 12
  assert "foot_gait" in env.reward_manager.active_terms
  env.close()
