import csv
from pathlib import Path

import matplotlib.pyplot as plt


SOLVED_REWARD = 200

METHOD_LOGS = {
    "double_dqn": {
        "label": "Double DQN",
        "log_path": "outputs/double_dqn/train_log.csv",
        "figure_dir": "outputs/double_dqn/figures",
    },
    "dueling_ddqn": {
        "label": "Dueling Double DQN",
        "log_path": "outputs/dueling_ddqn/train_log.csv",
        "figure_dir": "outputs/dueling_ddqn/figures",
    },
}


def read_training_log(log_path):
    # 统一读取 train.py 保存的 CSV 日志，返回画图和统计所需的序列。
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


def summarize_rewards(rewards, last_count=100):
    # 计算训练结果摘要；最后 100 轮平均奖励常用来表示最终收敛水平。
    actual_count = min(last_count, len(rewards))
    last_avg = mean_last(rewards, actual_count)
    return {
        "episodes": len(rewards),
        "best_reward": max(rewards),
        "final_reward": rewards[-1],
        "last_count": actual_count,
        "last_avg": last_avg,
        "gap_to_solved": SOLVED_REWARD - last_avg,
    }


def mean_last(values, count):
    # 计算最近 count 个 episode 的平均值，用来衡量训练后期表现。
    recent_values = values[-count:]
    return sum(recent_values) / len(recent_values)


def print_training_summary(rewards, avg_rewards):
    # 输出训练结果摘要，方便直接复制到实验记录或报告草稿中。
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
    # 绘制单个算法的 episode reward 曲线。
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


def plot_method_reward_curve(method, log_path=None, output_dir=None):
    # 绘制单个改进算法的完整 reward 曲线，同时显示单局 reward 和最近 10 轮平均 reward。
    method_config = METHOD_LOGS[method]
    label = method_config["label"]
    log_path = Path(log_path or method_config["log_path"])
    output_dir = Path(output_dir or method_config["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, rewards, avg_rewards, losses, epsilons = read_training_log(log_path)
    summary = summarize_rewards(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#9ca3af", linewidth=1, label="Episode reward")
    plt.plot(episodes, avg_rewards, color="#2563eb", linewidth=2, label="Average reward (last 10)")

    plt.title(f"{label} Reward Curve on LunarLander-v3")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{method}_reward_curve.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"{label} reward curve saved to: {output_path}")
    return output_path, summary


def plot_method_last100_reward_curve(method, log_path=None, output_dir=None):
    # 绘制最后 100 轮 reward 曲线，风格与基础 DQN 的 dqn_reward_curve_last100.png 保持一致。
    # 最后 100 轮平均值放在统计框里，避免和 200 分阈值线贴得太近时分辨不清。
    method_config = METHOD_LOGS[method]
    label = method_config["label"]
    log_path = Path(log_path or method_config["log_path"])
    output_dir = Path(output_dir or method_config["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, rewards, avg_rewards, losses, epsilons = read_training_log(log_path)
    summary = summarize_rewards(rewards)
    last_count = summary["last_count"]

    last_episodes = episodes[-last_count:]
    last_rewards = rewards[-last_count:]
    last_avg_rewards = avg_rewards[-last_count:]

    plt.figure(figsize=(10, 5))
    plt.plot(last_episodes, last_rewards, color="#9ca3af", linewidth=1, label="Episode reward")
    plt.plot(last_episodes, last_avg_rewards, color="#2563eb", linewidth=2, label="Average reward (last 10)")
    plt.axhline(SOLVED_REWARD, color="#dc2626", linestyle="--", linewidth=1.5, label="Solved threshold")

    stats_text = (
        f"Last {last_count} avg: {summary['last_avg']:.2f}\n"
        f"Gap to 200: {summary['gap_to_solved']:.2f}"
    )
    plt.gca().text(
        0.02,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )

    plt.title(f"{label} Reward Curve on LunarLander-v3 (Last 100 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{method}_reward_curve_last100.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"{label} last100 reward curve saved to: {output_path}")
    print(f"{label} average reward (last {last_count}): {summary['last_avg']:.2f}")
    print(f"{label} solved threshold gap: {summary['gap_to_solved']:.2f}")
    return output_path, summary


def plot_double_dqn_and_dueling_figures():
    # 生成 4 张图：Double DQN 和 Dueling DDQN 各一张完整 reward 曲线、一张最后 100 轮 reward 曲线。
    outputs = []
    for method in ("double_dqn", "dueling_ddqn"):
        outputs.append(plot_method_reward_curve(method))
        outputs.append(plot_method_last100_reward_curve(method))
    return outputs


def plot_comparison_curve(
    log_paths=None,
    output_dir="outputs/comparison/figures",
    output_name="reward_comparison_dqn_variants.png",
):
    # 绘制 DQN、Double DQN、Dueling DDQN 三组算法的奖励对比曲线。
    # 默认读取 outputs/dqn、outputs/double_dqn 和 outputs/dueling_ddqn。
    if log_paths is None:
        log_paths = {
            "DQN": "outputs/dqn/train_log.csv",
            "Double DQN": "outputs/double_dqn/train_log.csv",
            "Dueling DDQN": "outputs/dueling_ddqn/train_log.csv",
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    colors = {
        "DQN": "#9ca3af",
        "Double DQN": "#2563eb",
        "Dueling DDQN": "#dc8a00",
    }

    for method_name, log_path in log_paths.items():
        log_path = Path(log_path)
        if not log_path.exists():
            # 某个算法还没训练时跳过，不影响已经完成的曲线生成。
            print(f"Skip missing log: {log_path}")
            continue

        episodes, rewards, avg_rewards, losses, epsilons = read_training_log(log_path)
        # 使用 avg_reward_10 而不是原始 reward，可以让早期短实验趋势更清楚。
        plt.plot(
            episodes,
            avg_rewards,
            color=colors.get(method_name),
            linewidth=2,
            label=f"{method_name} avg10",
        )

    plt.axhline(
        SOLVED_REWARD,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label="Solved threshold",
    )

    plt.title("Reward Comparison of DQN Variants on LunarLander-v3")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (last 10)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Comparison reward curve saved to: {output_path}")


if __name__ == "__main__":
    plot_reward_curve()
    plot_comparison_curve()
