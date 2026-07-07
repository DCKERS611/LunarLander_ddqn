import csv
from pathlib import Path

import matplotlib.pyplot as plt


SOLVED_REWARD = 200


def read_training_log(log_path):
    episodes = []
    rewards = []
    avg_rewards = []
    losses = []
    epsilons = []

    with log_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avg_rewards.append(float(row["avg_reward_10"]))
            losses.append(float(row["loss"]))
            epsilons.append(float(row["epsilon"]))

    return episodes, rewards, avg_rewards, losses, epsilons


def mean_last(values, count):
    recent_values = values[-count:]
    return sum(recent_values) / len(recent_values)


def print_training_summary(rewards, avg_rewards):
    last_100_avg = mean_last(rewards, 100)
    last_50_avg = mean_last(rewards, 50)
    last_10_avg = mean_last(rewards, 10)
    gap = SOLVED_REWARD - last_100_avg

    print("\nDQN 训练结果统计")
    print(f"Total episodes: {len(rewards)}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Worst reward: {min(rewards):.2f}")
    print(f"Final reward: {rewards[-1]:.2f}")
    print(f"Final avg reward (last 10): {avg_rewards[-1]:.2f}")
    print(f"Average reward (last 50): {last_50_avg:.2f}")
    print(f"Average reward (last 100): {last_100_avg:.2f}")

    if last_100_avg >= SOLVED_REWARD:
        print(f"Solved threshold: {SOLVED_REWARD}, reached.")
    else:
        print(f"Solved threshold: {SOLVED_REWARD}, gap: {gap:.2f}")


def plot_reward_curve(log_path="outputs/dqn/train_log.csv", output_dir="outputs/dqn/figures"):
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, rewards, avg_rewards, losses, epsilons = read_training_log(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#2563eb", linewidth=1.2, label="Episode reward")

    plt.title("DQN Reward Curve on LunarLander-v3")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "dqn_reward_curve.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Reward curve saved to: {output_path}")
    print_training_summary(rewards, avg_rewards)


if __name__ == "__main__":
    plot_reward_curve()
