from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Gait clock as ``[sin, cos]`` of the episode-time phase.

  Zero while the command norm is at or below ``command_threshold``, so standing has
  no free-running clock.

  Returns:
    Tensor of shape [B, 2].
  """
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  angle = global_phase * 2.0 * torch.pi
  clock = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stand_mask = torch.linalg.norm(command, dim=1) <= command_threshold
  return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(clock), clock)
