import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from envs import make_env


METHODS = {
    # 通过 method 名称集中管理三组实验的差异，训练主循环保持完全一致。
    # 这样报告中可以说明：对比实验只改变算法模块，不改变环境和日志格式。
    "dqn": {
        "label": "DQN",
        "double_dqn": False,
        "dueling": False,
        "output_dir": "outputs/dqn",
    },
    "double_dqn": {
        "label": "Double DQN",
        "double_dqn": True,
        "dueling": False,
        "output_dir": "outputs/double_dqn",
    },
    "dueling_ddqn": {
        "label": "Dueling Double DQN",
        "double_dqn": True,
        "dueling": True,
        "output_dir": "outputs/dueling_ddqn",
    },
}


def set_seed(seed):
    # 固定随机种子，减少不同运行之间的随机波动，方便复现实验曲线。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dqn(
    method="dqn",
    episodes=500,
    max_steps=1000,
    target_update_interval=10,
    output_dir=None,
    seed=42,
):
    # method 决定使用基础 DQN、Double DQN 还是 Dueling Double DQN。
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHODS)}")

    method_config = METHODS[method]

    # 如果没有手动指定 output_dir，就按算法名称保存到 outputs/dqn 等目录。
    output_dir = output_dir or method_config["output_dir"]
    set_seed(seed)

    env = make_env(render_mode=None)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 根据 method_config 中的开关构建对应智能体。
    # double_dqn=True 会启用 Double DQN 目标值；dueling=True 会启用 Dueling 网络。
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=method_config["double_dqn"],
        dueling=method_config["dueling"],
    )

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # rewards 用于计算最近 10 局平均奖励，log_rows 最终写入 CSV 文件。
    rewards = []
    log_rows = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        total_reward = 0
        losses = []

        for step in range(1, max_steps + 1):
            # 先用 epsilon 保持探索，后期逐渐更多依赖网络判断。
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 存储当前 transition，并立即尝试从回放池采样更新网络。
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        # 每隔固定 episode 同步一次 target network，避免目标值剧烈变化。
        if episode % target_update_interval == 0:
            agent.update_target_network()

        # epsilon 逐步衰减，但不低于 epsilon_min，保留少量探索能力。
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        avg_loss = sum(losses) / len(losses) if losses else 0
        rewards.append(total_reward)
        recent_avg_reward = sum(rewards[-10:]) / min(len(rewards), 10)

        # CSV 日志包含 reward、loss、epsilon 和步数，后续画图和报告统计都读这个文件。
        log_rows.append(
            {
                "episode": episode,
                "reward": total_reward,
                "avg_reward_10": recent_avg_reward,
                "loss": avg_loss,
                "epsilon": epsilon,
                "steps": step,
                "method": method,
            }
        )

        print(
            f"{method_config['label']} | "
            f"Episode {episode:03d} | "
            f"reward={total_reward:8.2f} | "
            f"avg10={recent_avg_reward:8.2f} | "
            f"loss={avg_loss:.4f} | "
            f"epsilon={epsilon:.3f} | "
            f"steps={step}"
        )

    env.close()

    log_path = output_path / "train_log.csv"
    model_path = output_path / f"{method}_model.pth"

    # 每种算法都保存为相同的 train_log.csv，便于 plot_results.py 自动读取对比。
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # 保存 online network 参数，用于后续评估、推理或录制视频。
    torch.save(agent.q_net.state_dict(), model_path)

    print(f"\n算法: {method_config['label']}")
    print(f"训练日志已保存: {log_path}")
    print(f"模型权重已保存: {model_path}")
    return log_path, model_path


if __name__ == "__main__":
    # 命令行接口示例：
    # uv run python src/train.py --method double_dqn --episodes 300
    parser = argparse.ArgumentParser(description="Train DQN variants on LunarLander-v3.")
    parser.add_argument("--method", choices=METHODS.keys(), default="dqn")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dqn(
        method=args.method,
        episodes=args.episodes,
        max_steps=args.max_steps,
        target_update_interval=args.target_update_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )
