#!/usr/bin/env python3
"""
Record LunarLander episodes as JPEG frame sequences (Gymnasium; default LunarLander-v3).

Replaces the notebook pipeline (Monitor + ffmpeg + OpenCV) with direct rgb_array
renders for simpler, cross-platform reproducibility.

Requires: pip install -e ".[gym]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect LunarLander frame sequences for training.")
    parser.add_argument("--out", type=Path, default=Path("Train"), help="Output root (episode subfolders).")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to record.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (incremented per episode).")
    parser.add_argument(
        "--env-id",
        default="LunarLander-v3",
        help="Gymnasium environment id (default: LunarLander-v3; v2 is deprecated in recent Gymnasium).",
    )
    args = parser.parse_args()

    try:
        import gymnasium as gym
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print("Missing dependency. Install with: pip install -e \".[gym]\"", file=sys.stderr)
        raise SystemExit(1) from e

    args.out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        ep_dir = args.out / str(ep)
        ep_dir.mkdir(parents=True, exist_ok=True)
        env = gym.make(args.env_id, render_mode="rgb_array")
        env.reset(seed=args.seed + ep)
        frame_idx = 0

        def save_frame(rgb: np.ndarray) -> None:
            nonlocal frame_idx
            img = Image.fromarray(rgb)
            img.save(ep_dir / f"frame{frame_idx}.jpg", quality=92)
            frame_idx += 1

        save_frame(env.render())

        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            _obs, _reward, terminated, truncated, _info = env.step(action)
            save_frame(env.render())

        env.close()
        print(f"Episode {ep}: wrote {frame_idx} frames -> {ep_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
