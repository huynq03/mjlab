"""Export a Mini-Pi checkpoint with metadata and verify ONNX inference.

Usage:
  uv run --locked --extra cu130 scripts/tools/export_minipi_onnx.py CHECKPOINT
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--task", default="Mjlab-Velocity-Flat-MiniPi")
  args = parser.parse_args()
  checkpoint = args.checkpoint.resolve(strict=True)
  output = (args.output or checkpoint.with_suffix(".onnx")).resolve()
  if output == checkpoint:
    parser.error("Output must differ from the checkpoint path.")

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=agent_cfg.clip_actions
  )
  try:
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
    )
    runner.export_policy_to_onnx(str(output.parent), output.name)
    attach_metadata_to_onnx(
      str(output), get_base_metadata(env.unwrapped, str(checkpoint))
    )
    onnx.checker.check_model(str(output))

    model = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    input_names = cast(list[str], model.input_names)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    max_error = 0.0
    with torch.inference_mode():
      for sample in range(10):
        inputs = tuple(
          torch.zeros_like(x)
          if sample == 0
          else torch.randn(x.shape, generator=generator, dtype=x.dtype)
          for x in model.get_dummy_inputs()  # type: ignore[operator]
        )
        expected = model(*inputs).numpy()
        actual = session.run(
          None,
          {name: x.numpy() for name, x in zip(input_names, inputs, strict=True)},
        )[0]
        assert isinstance(actual, np.ndarray), "Expected a dense ONNX action tensor"
        assert np.isfinite(actual).all(), "Non-finite ONNX actions"
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        assert actual.shape == (1, 12), actual.shape
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    print(f"Exported: {output}")
    print(f"ONNX validation passed; 12 actions; max absolute error: {max_error:.3g}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
