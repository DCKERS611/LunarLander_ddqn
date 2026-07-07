from pathlib import Path

import matplotlib.pyplot as plt

from plot_results import SOLVED_REWARD, mean_last, read_training_log


def plot_last100_reward_curve(log_path="outputs/dqn/train_log.csv", output_dir="outputs/dqn/figures"):
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, rewards, avg_rewards, losses, epsilons = read_training_log(log_path)
    last_count = min(100, len(episodes))

    last_episodes = episodes[-last_count:]
    last_rewards = rewards[-last_count:]
    last_avg_rewards = avg_rewards[-last_count:]
    last100_avg = mean_last(rewards, last_count)

    plt.figure(figsize=(10, 5))
    plt.plot(last_episodes, last_rewards, color="#9ca3af", linewidth=1, label="Episode reward")
    plt.plot(last_episodes, last_avg_rewards, color="#2563eb", linewidth=2, label="Average reward (last 10)")
    plt.axhline(SOLVED_REWARD, color="#dc2626", linestyle="--", linewidth=1.5, label="Solved threshold")

    plt.title("DQN Reward Curve on LunarLander-v3 (Last 100 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "dqn_reward_curve_last100.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Last 100 reward curve saved to: {output_path}")
    print(f"Average reward (last {last_count}): {last100_avg:.2f}")
    print(f"Solved threshold: {SOLVED_REWARD}, gap: {SOLVED_REWARD - last100_avg:.2f}")


if __name__ == "__main__":
    plot_last100_reward_curve()
