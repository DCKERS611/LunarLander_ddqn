import csv
from pathlib import Path

import torch

from agent import DQNAgent
from envs import make_env


def train_dqn(
    episodes=500,
    max_steps=1000,
    target_update_interval=10,
    output_dir="outputs/dqn",
):
    env = make_env(render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

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

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        if episode % target_update_interval == 0:
            agent.update_target_network()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        avg_loss = sum(losses) / len(losses) if losses else 0
        rewards.append(total_reward)
        recent_avg_reward = sum(rewards[-10:]) / min(len(rewards), 10)

        log_rows.append(
            {
                "episode": episode,
                "reward": total_reward,
                "avg_reward_10": recent_avg_reward,
                "loss": avg_loss,
                "epsilon": epsilon,
                "steps": step,
            }
        )

        print(
            f"Episode {episode:03d} | "
            f"reward={total_reward:8.2f} | "
            f"avg10={recent_avg_reward:8.2f} | "
            f"loss={avg_loss:.4f} | "
            f"epsilon={epsilon:.3f} | "
            f"steps={step}"
        )

    env.close()

    log_path = output_path / "train_log.csv"
    model_path = output_path / "dqn_model.pth"

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(agent.q_net.state_dict(), model_path)

    print(f"\n训练日志已保存: {log_path}")
    print(f"模型权重已保存: {model_path}")


if __name__ == "__main__":
    train_dqn()
