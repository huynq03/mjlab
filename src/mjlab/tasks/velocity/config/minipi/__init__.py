from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  minipi_flat_env_cfg,
  minipi_rough_env_cfg,
)
from .rl_cfg import minipi_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-MiniPi",
  env_cfg=minipi_rough_env_cfg(),
  play_env_cfg=minipi_rough_env_cfg(play=True),
  rl_cfg=minipi_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MiniPi",
  env_cfg=minipi_flat_env_cfg(),
  play_env_cfg=minipi_flat_env_cfg(play=True),
  rl_cfg=minipi_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
