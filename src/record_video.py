import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from envs import make_env
from evaluate import build_agent, default_model_path, load_model_state, run_eval_episode
from train import METHODS


def choose_best_seed(agent, episodes, seed, max_steps):
    env = make_env(render_mode=None)
    env.action_space.seed(seed)

    candidates = []
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        result = run_eval_episode(agent, env, episode_seed, max_steps)
        result["episode"] = episode
        candidates.append(result)

        print(
            f"candidate | episode={episode:02d} | "
            f"seed={episode_seed} | "
            f"reward={result['reward']:8.2f} | "
            f"steps={result['steps']:4d}"
        )

    env.close()
    return max(candidates, key=lambda row: row["reward"]), candidates


def record_episode(agent, seed, max_steps):
    env = make_env(render_mode="rgb_array")
    env.action_space.seed(seed)

    frames = []
    state, _ = env.reset(seed=seed)
    first_frame = env.render()
    if first_frame is not None:
        frames.append(np.asarray(first_frame))

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for steps in range(1, max_steps + 1):
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    env.close()
    return {
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "frames": frames,
    }


def save_mp4(frames, output_path, fps):
    try:
        import cv2
    except ImportError:
        print("opencv-python is not available; skip mp4 export.")
        return None

    if not frames:
        return None

    output_path = Path(output_path)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    return output_path


def save_key_frames(frames, method, frame_dir):
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        return []

    frame_specs = [
        ("start", 0),
        ("middle", len(frames) // 2),
        ("end", len(frames) - 1),
    ]

    paths = []
    for label, index in frame_specs:
        path = frame_dir / f"{method}_best_{label}.png"
        imageio.imwrite(path, frames[index])
        paths.append(path)

    return paths


def record_model_video(
    method,
    model_path=None,
    episodes=20,
    seed=20260710,
    max_steps=1000,
    hidden_dim=128,
    lr=1e-3,
    gamma=0.99,
    fps=30,
    output_dir="outputs/videos",
    frame_dir="screenshots/inference",
    device=None,
):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHODS)}")

    model_path = Path(model_path) if model_path else default_model_path(method)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(render_mode=None)
    agent = build_agent(method, probe_env, hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)
    load_model_state(agent, model_path)
    probe_env.close()

    best_result, candidates = choose_best_seed(agent, episodes, seed, max_steps)
    print(
        "\nRecording best candidate: "
        f"seed={best_result['seed']}, "
        f"reward={best_result['reward']:.2f}, "
        f"steps={best_result['steps']}"
    )

    recorded = record_episode(agent, best_result["seed"], max_steps)
    frames = recorded.pop("frames")

    gif_path = output_dir / f"{method}_inference_best.gif"
    mp4_path = output_dir / f"{method}_inference_best.mp4"
    summary_path = output_dir / f"{method}_inference_best_summary.json"

    imageio.mimsave(gif_path, frames, fps=fps)
    saved_mp4_path = save_mp4(frames, mp4_path, fps=fps)
    frame_paths = save_key_frames(frames, method, frame_dir)

    summary = {
        "method": method,
        "label": METHODS[method]["label"],
        "model_path": str(model_path),
        "candidate_episodes": episodes,
        "seed_start": seed,
        "selected_seed": best_result["seed"],
        "selected_reward": best_result["reward"],
        "selected_steps": best_result["steps"],
        "recorded_reward": recorded["reward"],
        "recorded_steps": recorded["steps"],
        "gif_path": str(gif_path),
        "mp4_path": str(saved_mp4_path) if saved_mp4_path else "",
        "frame_paths": [str(path) for path in frame_paths],
        "candidates": candidates,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"GIF saved to: {gif_path}")
    if saved_mp4_path:
        print(f"MP4 saved to: {saved_mp4_path}")
    print(f"Key frames saved to: {frame_dir}")
    print(f"Recording summary saved to: {summary_path}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Record a trained LunarLander policy as GIF/MP4.")
    parser.add_argument("--method", choices=METHODS.keys(), default="dueling_ddqn")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/videos")
    parser.add_argument("--frame-dir", default="screenshots/inference")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_model_video(
        method=args.method,
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        fps=args.fps,
        output_dir=args.output_dir,
        frame_dir=args.frame_dir,
        device=args.device,
    )
