"""HighTorque Mini-Pi constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MINIPI_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "minipi" / "xmls" / "cl_pai.xml"
)
assert MINIPI_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MINIPI_XML))


##
# Actuator config.
##

# PD gains from HighTorque's RL deployment configs (sim2real walk/dreamwaq.yaml and
# clpai_12dof_0905 devel_config/config.yaml). Effort limit, armature, frictionloss and
# viscous damping are left unset: the vendor files give no trustworthy motor specs.
MINIPI_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=60.0,
  damping=2.4,
)
MINIPI_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=40.0,
  damping=0.8,
)
MINIPI_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_joint",),
  stiffness=20.0,
  damping=0.4,
)
MINIPI_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=60.0,
  damping=2.8,
)
MINIPI_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=1.6,
)
MINIPI_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=10.0,
  damping=0.3,
)

##
# Keyframe config.
##

# The MJCF bakes the nominal knee-bent stance into the body frames, so zero joint
# angles is the standing pose. Base height matches HighTorque's pai_12dof.xml.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.347),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[lr]_foot[1-5]_collision$"

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

MINIPI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MINIPI_ACTUATOR_HIP_PITCH,
    MINIPI_ACTUATOR_HIP_ROLL,
    MINIPI_ACTUATOR_THIGH,
    MINIPI_ACTUATOR_CALF,
    MINIPI_ACTUATOR_ANKLE_PITCH,
    MINIPI_ACTUATOR_ANKLE_ROLL,
  ),
)


def get_minipi_robot_cfg() -> EntityCfg:
  """Get a fresh Mini-Pi robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MINIPI_ARTICULATION,
  )


# From HighTorque's RL deployment configs (action_scale: 0.25), not derived from
# effort limits like the other robots.
MINIPI_ACTION_SCALE: float = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_minipi_robot_cfg())

  viewer.launch(robot.spec.compile())
