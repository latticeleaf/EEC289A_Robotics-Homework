#!/usr/bin/env python3
"""Custom per-direction evaluation script for the Go2 locomotion homework.

Tests the policy across all 6 motion directions at multiple command magnitudes,
providing a more complete picture of policy capability than the 4-episode benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent

# Custom episode definitions: (label, segments)
# Each segment is [vx, vy, yaw_rate]
# Pattern: stand -> command -> stand
CUSTOM_EPISODES = [
    ("forward_slow",    [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ("forward_medium",  [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ("forward_fast",    [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ("backward",        [[0.0, 0.0, 0.0], [-0.4, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ("lateral_left",    [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]),
    ("lateral_right",   [[0.0, 0.0, 0.0], [0.0, -0.15, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]]),
    ("yaw_left_slow",   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]]),
    ("yaw_left_fast",   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]),
    ("yaw_right",       [[0.0, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]]),
    ("combined",        [[0.0, 0.0, 0.0], [0.4, 0.1, 0.3], [0.5, 0.15, 0.4], [0.0, 0.0, 0.0]]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--episode-length-seconds", type=float, default=10.0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def command_for_step(segments, step_idx, total_steps):
    segment_length = max(1, total_steps // len(segments))
    segment_idx = min(len(segments) - 1, step_idx // segment_length)
    return np.asarray(segments[segment_idx], dtype=np.float32)


def _force_command(state, command, jax):
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}

    force_cpu = bool(config.get("force_cpu")) or args.force_cpu
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    episode_length = int(round(args.episode_length_seconds / env_cfg.ctrl_dt))
    env_cfg.episode_length = episode_length

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    rng = jax.random.PRNGKey(int(config["seed"]) + 99)
    per_episode_results = []

    for episode_idx, (label, segments) in enumerate(CUSTOM_EPISODES):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        state = _force_command(state, np.asarray(segments[0], dtype=np.float32), jax)

        cmd_xy_list, meas_xy_list = [], []
        cmd_yaw_list, meas_yaw_list = [], []
        fell_any = False

        for step_idx in range(episode_length):
            command = command_for_step(segments, step_idx, episode_length)
            state = _force_command(state, command, jax)

            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, command, jax)

            cmd_xy_list.append(command[:2])
            meas_xy_list.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
            cmd_yaw_list.append(command[2])
            meas_yaw_list.append(float(np.asarray(env.get_gyro(state.data)[2])))

            if bool(np.asarray(state.done)):
                fell_any = True
                break

        cmd_xy = np.asarray(cmd_xy_list)
        meas_xy = np.asarray(meas_xy_list)
        vel_error = float(np.linalg.norm(cmd_xy - meas_xy, axis=-1).mean())
        yaw_error = float(np.abs(np.asarray(cmd_yaw_list) - np.asarray(meas_yaw_list)).mean())

        result = {
            "episode_id": episode_idx,
            "label": label,
            "velocity_tracking_error": vel_error,
            "yaw_tracking_error": yaw_error,
            "fell": fell_any,
            "num_steps": len(cmd_xy_list),
        }
        per_episode_results.append(result)
        print(f"[{label}] vel_err={vel_error:.4f} yaw_err={yaw_error:.4f} fell={fell_any}")

    # Aggregate
    fall_rate = float(np.mean([r["fell"] for r in per_episode_results]))
    mean_vel_err = float(np.mean([r["velocity_tracking_error"] for r in per_episode_results]))
    mean_yaw_err = float(np.mean([r["yaw_tracking_error"] for r in per_episode_results]))

    output = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "num_episodes": len(CUSTOM_EPISODES),
        "episode_length_seconds": args.episode_length_seconds,
        "aggregate": {
            "mean_velocity_tracking_error": mean_vel_err,
            "mean_yaw_tracking_error": mean_yaw_err,
            "fall_rate": fall_rate,
        },
        "per_episode": per_episode_results,
    }

    out_path = output_dir / "custom_eval.json"
    save_json(out_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
