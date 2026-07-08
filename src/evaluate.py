import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from agent import DQNAgent
from envs import make_env
from train import METHODS


def set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_model_path(method):
    return Path(METHODS[method]["output_dir"]) / f"{method}_model.pth"


def build_agent(method, env, hidden_dim=128, lr=1e-3, gamma=0.99, device=None):
    method_config = METHODS[method]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    return DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        double_dqn=method_config["double_dqn"],
        dueling=method_config["dueling"],
        prioritized_replay=method_config["prioritized_replay"],
        device=device,
    )


def load_model_state(agent, model_path):
    model_path = Path(model_path)
    state = torch.load(model_path, map_location=agent.device)

    if isinstance(state, dict) and "q_net_state_dict" in state:
        state = state["q_net_state_dict"]

    agent.q_net.load_state_dict(state)
    agent.update_target_network()
    agent.q_net.eval()
    agent.target_q_net.eval()


def run_eval_episode(agent, env, seed, max_steps):
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for steps in range(1, max_steps + 1):
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
    }


def summarize_results(rows, success_threshold):
    rewards = [row["reward"] for row in rows]
    steps = [row["steps"] for row in rows]
    successes = [1 for reward in rewards if reward >= success_threshold]

    return {
        "episodes": len(rows),
        "mean_reward": mean(rewards),
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "success_threshold": success_threshold,
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows),
        "mean_steps": mean(steps),
    }


def save_eval_outputs(rows, summary, method, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{method}_eval_episodes.csv"
    json_path = output_dir / f"{method}_eval_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["episode", "seed", "reward", "steps", "success", "terminated", "truncated"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return csv_path, json_path


def evaluate_model(
    method,
    model_path=None,
    episodes=20,
    seed=20260710,
    max_steps=1000,
    hidden_dim=128,
    lr=1e-3,
    gamma=0.99,
    success_threshold=200.0,
    output_dir="outputs/evaluation",
    device=None,
):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHODS)}")

    set_eval_seed(seed)
    model_path = Path(model_path) if model_path else default_model_path(method)

    env = make_env(render_mode=None)
    env.action_space.seed(seed)
    agent = build_agent(method, env, hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)
    load_model_state(agent, model_path)

    rows = []
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        result = run_eval_episode(agent, env, episode_seed, max_steps)
        result["episode"] = episode
        result["success"] = result["reward"] >= success_threshold
        rows.append(result)

        print(
            f"{METHODS[method]['label']} eval | "
            f"episode={episode:02d} | "
            f"seed={episode_seed} | "
            f"reward={result['reward']:8.2f} | "
            f"steps={result['steps']:4d} | "
            f"success={int(result['success'])}"
        )

    env.close()

    summary = summarize_results(rows, success_threshold)
    summary.update(
        {
            "method": method,
            "label": METHODS[method]["label"],
            "model_path": str(model_path),
            "seed_start": seed,
            "max_steps": max_steps,
        }
    )

    csv_path, json_path = save_eval_outputs(rows, summary, method, output_dir)

    print("\nEvaluation summary")
    print(f"method: {METHODS[method]['label']}")
    print(f"episodes: {summary['episodes']}")
    print(f"mean_reward: {summary['mean_reward']:.2f}")
    print(f"std_reward: {summary['std_reward']:.2f}")
    print(f"min_reward: {summary['min_reward']:.2f}")
    print(f"max_reward: {summary['max_reward']:.2f}")
    print(f"success_rate: {summary['success_rate']:.2%}")
    print(f"mean_steps: {summary['mean_steps']:.1f}")
    print(f"episode results saved to: {csv_path}")
    print(f"summary saved to: {json_path}")

    return rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN variants on LunarLander-v3.")
    parser.add_argument("--method", choices=METHODS.keys(), default="dueling_ddqn")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--success-threshold", type=float, default=200.0)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        method=args.method,
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        success_threshold=args.success_threshold,
        output_dir=args.output_dir,
        device=args.device,
    )
