"""HighTorque Mini-Pi velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  MINIPI_ACTION_SCALE,
  get_minipi_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Desired swing-foot clearance/peak height, shared by foot_clearance and
# foot_swing_height. A reward target, not a hard limit. Raised from the 0.04 m
# baseline (about 13% of the 0.31 m leg length) for higher steps.
_FOOT_CLEARANCE = 0.05

# Commands with norm at or below this are zeroed (stand); every remaining command
# drives the locomotion terms. Shared by the sampler and all stand/walk switches.
_COMMAND_THRESHOLD = 0.1

# Gait clock period, shared by the phase observation and the foot_gait reward. The
# legs are half a period apart, so at vx = 0.4 m/s a step covers 0.4 * 0.25 = 0.1 m.
_GAIT_PERIOD = 0.5


def minipi_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create HighTorque Mini-Pi rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # Mini-Pi has no armature, so at the generic 5 ms timestep the joint damping makes
  # the stance chatter and the feet creep apart. 2 ms is clean; the policy stays at
  # 50 Hz.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  cfg.scene.entities = {"robot": get_minipi_robot_cfg()}

  # Set raycast sensor frame to Mini-Pi base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # Right first, matching the XML body/site order. Site ids and contact channels
  # resolve in model order (right, left) while the foot height scan keeps this
  # tuple's order, so any other order cross-pairs the feet in foot_clearance.
  site_names = ("r_foot", "l_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("l", "r") for i in range(1, 6)
  )

  # Wire foot height scan to per-foot sites. The sole is 57-68 mm wide, so a 25 mm
  # ring around the sole center stays inside the footprint.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.025, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Actions follow joint order (right leg then left leg), not actuator ctrl order.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MINIPI_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  # Visualization only: the top of the base is 0.16 m above the base origin.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.3

  # Widened past HighTorque's Mini-Pi RL deployment limits (sim2real
  # walk/dreamwaq.yaml: vx/vy +-0.25/+-0.2 m/s, yaw 2.0 rad/s): longer forward and
  # lateral steps, faster yaw. Yaw stays inside the vendor limit.
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.6)
  twist_cmd.ranges.lin_vel_y = (-0.45, 0.45)
  twist_cmd.ranges.ang_vel_z = (-1.2, 1.2)
  # Forward-only envs force vx >= 0.3 m/s, only the top of the range; kept off.
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.command_deadzone = _COMMAND_THRESHOLD
  # Fixed command range: the generic curriculum ramps vx up to 3 m/s.
  cfg.curriculum.pop("command_vel", None)

  # First locomotion finetune: no pushes and no reset drop. Robustness training
  # comes back later.
  cfg.events.pop("push_robot", None)
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  # Gait clock right after the command, as in Unitree's velocity task. The critic
  # copied the actor terms at build time, so it gets its own entry.
  phase_obs = ObservationTermCfg(
    func=mdp.phase,
    params={
      "period": _GAIT_PERIOD,
      "command_name": "twist",
      "command_threshold": _COMMAND_THRESHOLD,
    },
  )
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    assert "phase" not in terms
    items = list(terms.items())
    idx = [name for name, _ in items].index("command") + 1
    cfg.observations[group].terms = dict(
      items[:idx] + [("phase", phase_obs)] + items[idx:]
    )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # First-run baseline tolerances mapped from G1's lower body (thigh_joint is the hip
  # yaw, calf_joint is the knee). Not tuned for Mini-Pi.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_pitch_joint": 0.8,
    r".*_hip_roll_joint": 0.25,
    r".*_thigh_joint": 0.3,
    r".*_calf_joint": 0.7,
    r".*_ankle_pitch_joint": 0.35,
    r".*_ankle_roll_joint": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_pitch_joint": 0.5,
    r".*_hip_roll_joint": 0.2,
    r".*_thigh_joint": 0.2,
    r".*_calf_joint": 0.6,
    r".*_ankle_pitch_joint": 0.35,
    r".*_ankle_roll_joint": 0.15,
  }

  # Narrower than the generic tracking width (std 0.5 for 1 m/s commands) for the
  # smaller Mini-Pi range; widened from 0.125 for commands up to 0.6 m/s.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.2

  # Tracking at 1.0 (generic 2.0) so it does not dominate gait formation. Roll and
  # pitch rates count 0.05x so natural walking sway is not punished like yaw error.
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].params["xy_weight"] = 0.05

  # Stand/walk switches match the command deadzone.
  cfg.rewards["pose"].params["walking_threshold"] = _COMMAND_THRESHOLD
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["command_threshold"] = _COMMAND_THRESHOLD
  cfg.rewards["soft_landing"].params["command_threshold"] = _COMMAND_THRESHOLD

  # Alternating gait clock, left/right half a period apart, on the same clock as the
  # phase observation.
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=mdp.feet_gait,
    weight=0.75,
    params={
      "sensor_name": feet_ground_cfg.name,
      "period": _GAIT_PERIOD,
      "offset": [0.0, 0.5],
      "threshold": 0.55,
      "command_name": "twist",
      "command_threshold": _COMMAND_THRESHOLD,
    },
  )

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  for reward_name in ["foot_clearance", "foot_swing_height"]:
    cfg.rewards[reward_name].params["target_height"] = _FOOT_CLEARANCE

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  # Lighter clearance shaping and smoothness, stronger slip penalty, so stepping
  # beats shuffling. foot_swing_height pulls the landing peak toward 5 cm.
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_swing_height"].weight = -1.0
  cfg.rewards["foot_slip"].weight = -0.25
  cfg.rewards["action_rate_l2"].weight = -0.05

  # Unitree-style terms: hold the default pose at zero command, a large penalty for
  # falling (timeouts are truncations, not terminations), and a small joint
  # acceleration regularizer. Joint limits stay at the generic -1.0.
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": _COMMAND_THRESHOLD,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.rewards["is_terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)

  # The cost counts substeps in contact, up to the decimation. -0.4 over 10 substeps
  # matches G1's -1.0 over 4 for a collision lasting a full control step.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.4,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def minipi_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create HighTorque Mini-Pi flat terrain velocity configuration."""
  cfg = minipi_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
